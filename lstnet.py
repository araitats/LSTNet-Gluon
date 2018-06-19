'''
LSTNet SageMaker  
'''

from __future__ import print_function

import logging
import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn, rnn
import numpy as np
import json
import pickle
import os

logging.basicConfig(level=logging.DEBUG)

class TimeSeriesData(object):
    """
    Reads data from file and creates training and validation datasets
    """
    def __init__(self, file_path, window, horizon, train_ratio=0.8):
        """
        :param str file_path: path to the data file (e.g. electricity.txt)
        """
        # with open(file_path) as f:
        data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        data = self._normalize(data)
        train_data_len = int(len(data) * train_ratio)
        self.num_series = data.shape[1]
        if train_ratio > 0.0:
            self.train = TimeSeriesDataset(data[:train_data_len], window=window, horizon=horizon)
        if train_ratio < 1.0:
            self.val = TimeSeriesDataset(data[train_data_len:], window=window, horizon=horizon)

    def _normalize(self, data):
        """ Normalizes data by maximum value per row (i.e. per time series) and saves the scaling factor

        :param np.ndarray data: input data to be normalized
        :return: normalized data
        :rtype np.ndarray
        """
        self.scale = np.max(data, axis=0)
        return data / self.scale


class TimeSeriesDataset(gluon.data.Dataset):
    """
    Dataset that splits the data into a dense overlapping windows
    """
    def __init__(self, data, window, horizon, transform=None):
        """
        :param np.ndarray data: time-series data in TC layout (T: sequence len, C: channels)
        :param int window: context window size
        :param int horizon: prediction horizon
        :param function transform: data transformation function: fn(data, label)
        """
        super(TimeSeriesDataset, self).__init__()
        self._data = data
        self._window = window
        self._horizon = horizon
        self._transform = transform

    def __getitem__(self, idx):
        """
        :param int idx: index of the item
        :return: single item in 'TC' layout
        :rtype np.ndarray
        """
        assert idx < len(self)
        data = self._data[idx:idx + self._window]
        label = self._data[idx + self._window + self._horizon - 1]
        if self._transform is not None:
            return self._transform(data, label)
        return data, label

    def __len__(self):
        """
        :return: length of the dataset
        :rtype int
        """
        return len(self._data) - self._window - self._horizon


class LSTNet(gluon.Block):
    """
    LSTNet auto-regressive block
    """
    def __init__(self, ctx):
        super(LSTNet, self).__init__()
        self.ctx = ctx
        
    def build_model(self, num_series, conv_hid, gru_hid, skip_gru_hid, skip, ar_window, model_dir = False):
        kernel_size = 6
        dropout_rate = 0.2
        self.skip = skip
        self.ar_window = ar_window
        self.model_dir = model_dir
        self.num_series = num_series
        self.conv_hid = conv_hid
        self.gru_hid = gru_hid
        self.skip_gru_hid = skip_gru_hid
        with self.name_scope():
            self.conv = nn.Conv1D(conv_hid, kernel_size=kernel_size, layout='NCW', activation='relu')
            self.dropout = nn.Dropout(dropout_rate)
            self.gru = rnn.GRU(gru_hid, layout='TNC')
            self.skip_gru = rnn.GRU(skip_gru_hid, layout='TNC')
            self.fc = nn.Dense(num_series)
            self.ar_fc = nn.Dense(1)

    def forward(self, x):
        """
        :param nd.NDArray x: input data in NTC layout (N: batch-size, T: sequence len, C: channels)
        :return: output of LSTNet in NC layout
        :rtype nd.NDArray
        """
        # Convolution
        c = self.conv(x.transpose((0, 2, 1)))  # Transpose NTC to to NCT (a.k.a NCW) before convolution
        c = self.dropout(c)

        # GRU
        r = self.gru(c.transpose((2, 0, 1)))  # Transpose NCT to TNC before GRU
        r = r[-1]  # Only keep the last output
        r = self.dropout(r)  # Now in NC layout

        # Skip GRU
        # Slice off multiples of skip from convolution output
        skip_c = c[:, :, -(c.shape[2] // self.skip) * self.skip:]
        skip_c = skip_c.reshape((c.shape[0], c.shape[1], -1, self.skip))  # Reshape to NCT x skip
        skip_c = skip_c.transpose((2, 0, 3, 1))  # Transpose to T x N x skip x C
        skip_c = skip_c.reshape((skip_c.shape[0], -1, skip_c.shape[3]))  # Reshape to Tx (Nxskip) x C
        s = self.skip_gru(skip_c)
        s = s[-1]  # Only keep the last output (now in (Nxskip) x C layout)
        s = s.reshape((x.shape[0], -1))  # Now in N x (skipxC) layout

        # FC layer
        fc = self.fc(nd.concat(r, s))  # NC layout

        # Autoregressive highway
        ar_x = x[:, -self.ar_window:, :]  # NTC layout
        ar_x = ar_x.transpose((0, 2, 1))  # NCT layout
        ar_x = ar_x.reshape((-1, ar_x.shape[2]))  # (NC) x T layout
        ar = self.ar_fc(ar_x)
        ar = ar.reshape((x.shape[0], -1))  # NC layout

        # Add autoregressive and fc outputs
        res = fc + ar
        return res
    
    def compile_model(self, loss=None, lr = 0.001, clip_g = 10.):
        self.collect_params().initialize(init=mx.init.Xavier(factor_type="in", magnitude=2.34), ctx=self.ctx)
        self.trainer = gluon.Trainer(self.collect_params(),
                                     optimizer='adam',
                                     optimizer_params={'learning_rate': lr, 'clip_gradient': clip_g})
        self.loss = gluon.loss.L1Loss()
    
    def fit(self, ts_data, epochs = 100, batch_size = 128, out_path = 'model.params'):

        ctx = self.ctx
        
        train_data_loader = gluon.data.DataLoader(
            ts_data.train, batch_size=batch_size, shuffle=True, num_workers=16, last_batch='discard')

        scale = nd.array(ts_data.scale, ctx=ctx)

        loss = None
        print("Training Start")
        for e in range(epochs):
            epoch_loss = mx.nd.zeros((1,), ctx=ctx)
            num_iter = 0
            for data, label in train_data_loader:
                data = data.as_in_context(ctx)
                label = label.as_in_context(ctx)
                if loss is not None:
                    loss.wait_to_read()
                with autograd.record():
                    y_hat = self.forward(data)
                    loss = self.loss(y_hat * scale, label * scale)
                loss.backward()
                self.trainer.step(batch_size)
                epoch_loss = epoch_loss + loss.mean()
                num_iter += 1
            print("Epoch {:3d}: loss {:.4}".format(e, epoch_loss.asscalar() / num_iter))

        self.save_params('{}/model_out.params'.format(out_path))
        print("Training End")
        return 0


# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #

def train(channel_input_dirs, output_data_dir, model_dir, hyperparameters, **kwargs):

    # Retrieve the hyperparameters we set in notebook (with some defaults)
    batch_size = hyperparameters.get('batch_size', 100)
    epochs = hyperparameters.get('epochs', 10)
    num_gpus = hyperparameters.get('num_gpus', 0)
    
    # Parametrize the network definition
    conv_hid = hyperparameters.get('conv_hid', 100)
    gru_hid = hyperparameters.get('gru_hid', 100)
    skip_gru_hid = hyperparameters.get('skip_gru_hid', 5)
    skip = hyperparameters.get('skip', 24)
    ar_window = hyperparameters.get('ar_window', 24)
        
    # Load a text file
    text_file = hyperparameters.get('text_file', 'electricity.txt')
    path = channel_input_dirs['training']
    ts_data = load_data(path, text_file)
    
    # context 
    ctx = mx.cpu()
    if num_gpus >= 1:
        ctx = mx.gpu(0)
    model = LSTNet(ctx)
    model.build_model(num_series=ts_data.num_series,
                      conv_hid=conv_hid,
                      gru_hid=gru_hid,
                      skip_gru_hid=skip_gru_hid,
                      skip=skip,
                      ar_window=ar_window,
                      model_dir = model_dir)
    model.compile_model()
    model.fit(ts_data, epochs = epochs, batch_size = batch_size, out_path = output_data_dir)

    return model

def save(net, model_dir):
    net.save_params('{}/model.params'.format(model_dir))
    '''
    These parameters need to be saved.
    '''
    f = open('{}/model.json'.format(model_dir), 'w')
    json.dump({'num_series': net.num_series,
               'conv_hid': net.conv_hid,
               'gru_hid': net.gru_hid,
               'skip_gru_hid':net.skip_gru_hid,
               'skip': net.skip,
               'ar_window':net.ar_window},
              f)
    f.close()

def find_file(root_path, file_name):
    for root, dirs, files in os.walk(root_path):
        if file_name in files:
            return os.path.join(root, file_name)
    
def load_data(path, file_name):
    file_path = find_file(path, file_name)

    ts_data = TimeSeriesData(file_path, window=24*7, horizon=24)

    return ts_data

# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    ctx = mx.cpu()
    f = open('{}/model.json'.format(model_dir), 'r')
    block_params = json.load(f)
    f.close()
    model = LSTNet(ctx)
    model.build_model(num_series=block_params['num_series'],
                      conv_hid=block_params['conv_hid'],
                      gru_hid=block_params['gru_hid'],
                      skip_gru_hid=block_params['skip_gru_hid'],
                      skip=block_params['skip'],
                      ar_window=block_params['ar_window'])
    model.compile_model()
    model.load_params('{}/model.params'.format(model_dir), ctx)
    
    return model
    

def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.#

    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    # we can use content types to vary input/output handling, but
    # here we just assume json for both
    # data: <type 'unicode'>
    
    parsed = json.loads(data) #<type 'list'>

    nda = mx.nd.array(np.array(parsed), ctx = mx.cpu())

    output = net(nda) # calling model.forward()
    
    response_body = json.dumps(output.asnumpy().tolist())
    return response_body, output_content_type