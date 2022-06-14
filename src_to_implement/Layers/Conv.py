import numpy as np
from Layers.Base import BaseLayer


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.gradient_weights = None
        self.gradient_bias = None
        self.optimizer = None

    def forward(self, input_tensor):
        pass

    def backward(self, error_tensor):
        pass

    def initialize(self, weights_initializer, bias_initializer):
        pass

    @property
    def gradient_weights(self):
        return self.gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.gradient_weights = gradient_weights

    @property
    def gradient_bias(self):
        return self.gradient_weights

    @gradient_bias.setter
    def gradient_bias(self, gradient_bias):
        self.gradient_bias = gradient_bias

    @property
    def optimizer(self):
        return self.optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.optimizer = optimizer
