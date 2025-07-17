import numpy as np


class CorrelatedNoise:
    def __init__(self, sigma, tau, dt):
        self.sigma = sigma
        self.tau = tau
        self.dt = dt
        self.last_noise = None

    def __call__(self, x, dt=1.0):
        if self.last_noise is None:
            noise = np.random.normal(0, self.sigma, x.shape[0])
        else:
            dW = np.random.normal(0, 1, x.shape[0]) * np.sqrt(dt)
            noise = (
                self.last_noise
                + (-self.last_noise / self.tau) * dt
                + self.sigma * np.sqrt(2 / self.tau) * dW
            )
        self.last_noise = noise
        return x + noise


class WhiteNoise:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        noise = np.random.normal(0, self.sigma, x.shape[0])
        return x + noise
