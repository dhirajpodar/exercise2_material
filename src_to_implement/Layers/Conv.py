import numpy as np
from Layers.Base import BaseLayer
from scipy import signal


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.padding_size_x = None
        self.padding_size_y = None
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self._gradient_bias = None
        self._optimizer = None
        self.weights = np.random.rand(num_kernels, *convolution_shape)
        self.bias = np.random.rand(num_kernels)
        self._gradient_weights = np.zeros_like(self.weights)
        self.original_input_tensor = None
        self.batch_size = None
        self.input_channel_size = None

    def padding_calculation(self, param):
        if param % 2 != 0:
            return param // 2, param // 2
        else:
            return param // 2, param // 2 - 1

    def forward(self, input_tensor):
        self.original_input_tensor = np.copy(input_tensor)
        self.batch_size = input_tensor.shape[0]
        self.input_channel_size = input_tensor.shape[1]

        self.padding_size_x = (0, 0)
        padded_input_x = 0
        self.padding_size_y = self.padding_calculation(self.convolution_shape[1])
        out_y = int(
            ((input_tensor.shape[2] - self.convolution_shape[1] + sum(self.padding_size_y)) / self.stride_shape[0]) + 1)

        # if conv is 2D
        self.stride_indx = []
        if len(self.convolution_shape) == 3:
            #  Padding input tensor
            self.padding_size_x = self.padding_calculation(self.convolution_shape[2])
            padded_input_x = sum(self.padding_size_x) + input_tensor.shape[3]
            self.input_tensor = np.zeros(
                (self.batch_size, self.input_channel_size, sum(self.padding_size_y) + input_tensor.shape[2], \
                 padded_input_x))

            out_x = int(((input_tensor.shape[3] - self.convolution_shape[2] + sum(self.padding_size_x)) /
                         self.stride_shape[1]) + 1)
            #  Create empty output tensor
            output_tensor = np.zeros((self.batch_size, self.num_kernels,
                                      out_y, out_x * (len(self.convolution_shape) > 2)))

            # Get index of pixels for stride
            counter = 0
            for i in range(0, input_tensor.shape[3], self.stride_shape[1]):
                counter = i * input_tensor.shape[2]
                for j in range(0, input_tensor.shape[2], self.stride_shape[0]):
                    self.stride_indx.append(counter)
                    counter += self.stride_shape[0]

            # Padding axis y, x
            for b in range(self.batch_size):
                for c in range(self.input_channel_size):
                    self.input_tensor[b, c] = np.pad(input_tensor[b, c], (self.padding_size_y, self.padding_size_x),
                                                     mode='constant')

        # if conv is 1D
        elif len(self.convolution_shape) == 2:
            #  Padding input tensor
            self.input_tensor = np.zeros(
                (self.batch_size, self.input_channel_size, sum(self.padding_size_y) + input_tensor.shape[2]))
            #  Create empty output tensor
            output_tensor = np.zeros((self.batch_size, self.num_kernels, out_y))
            # Get index of pixels for stride
            self.stride_indx = [i for i in range(0, input_tensor.shape[2], self.stride_shape[0])]
            # Padding axis y
            for b in range(self.batch_size):
                for c in range(self.input_channel_size):
                    self.input_tensor[b, c] = np.pad(input_tensor[b, c], self.padding_size_y, mode='constant')
        else:
            raise NotImplementedError(
                f"Convolution size {len(self.convolution_shape)}D is not implemented! Size of 1D or 2D is expected.")

        # Number of batch
        for b in range(self.batch_size):
            # Number of kernel
            for k in range(self.num_kernels):
                # Number of channel
                corr = []
                for c in range(self.weights.shape[1]):
                    # Current weights and input
                    curr_weights = self.weights[k, c]
                    curr_input_tensor = input_tensor[b, c]
                    corr_temp = signal.correlate(curr_input_tensor, curr_weights, mode="same")
                    corr.append(corr_temp)

                corr_staked = np.stack(corr, axis=0)
                corr = corr_staked.sum(axis=0)
                # Get pixels in order to apply stride
                corr = corr.flatten()
                corr = corr[self.stride_indx]
                corr = corr.reshape(output_tensor.shape[2:])
                # Fill the output tensor with calculated values and added bias
                output_tensor[b, k] = corr + self.bias[k]

        return output_tensor

    def backward(self, error_tensor):
        pass

    def initialize(self, weights_initializer, bias_initializer):
        if len(self.convolution_shape) == 3:
            self.weights = weights_initializer.initialize(
                (self.num_kernels, self.convolution_shape[0], self.convolution_shape[1], self.convolution_shape[2]),
                self.convolution_shape[0], self.convolution_shape[1], self.convolution_shape[2],
                self.num_kernels * self.convolution_shape[1] * self.convolution_shape[2])
            self.bias = bias_initializer.initializer(self.num_kernels, 1, self.num_kernels)
            self.bias = self.bias[-1]

        elif len(self.convolution_shape) == 2:
            self.weights = weights_initializer.initialize(
                (self.num_kernels, self.convolution_shape[0], self.convolution_shape[1]),
                self.convolution_shape[0], self.convolution_shape[1],
                self.num_kernels * self.convolution_shape[1])
            self.bias = bias_initializer.initializer((1, self.num_kernels), 1, self.num_kernels)
            self.bias = self.bias[-1]

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, x):
        self._gradient_weights = x

    @property
    def gradient_bias(self):
        return self._gradient_weights

    @gradient_bias.setter
    def gradient_bias(self, x):
        self._gradient_bias = x

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
