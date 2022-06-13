class Constant:
    def __init__(self, weight):
        self.weight = weight

    def initialize(self, weights_shape, fan_in, fan_out):
        pass


class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        pass


class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        pass


class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        pass
