class SgdWithMomentum:
    def calculate_weights(self, weight_tensor, gradient_tensor):
        pass


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        
    def calculate_weights(self, weight_tensor, gradient_tensor):
        pass

