from Layers.Base import BaseLayer
import numpy as np

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        padding = 0  # 'valid padding'
        batch_size = input_tensor.shape[0]
        channel_size = input_tensor.shape[1]
        stride_row = self.stride_shape[0]
        stride_col = self.stride_shape[1]
        pool_row = self.pooling_shape[0]
        pool_col = self.pooling_shape[1]
        row_path_size = int((input_tensor.shape[2] - pool_row + 2 * padding) / stride_row) + 1
        col_path_size = int((input_tensor.shape[3] - pool_col + 2 * padding) / stride_col) + 1
        output = np.zeros((batch_size, channel_size, row_path_size, col_path_size))


    def backward(self, error_tensor):
        pass

