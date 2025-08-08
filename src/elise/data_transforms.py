import numpy as np


class ColorNotes:
    def __init__(self, val_min=0.5, val_max=1.5, n_palette=100):
        self.val_min = val_min
        self.val_max = val_max
        self.n_palette = n_palette
        self.rng = np.random.default_rng()

    def __call__(self, arr):
        """
        Color the notes in the given array based on their properties.

        :param arr: 2D numpy array where each column represents a note.
        :return: 2D numpy array with colored values.
        """
        return self.color_notes(
            arr, self.val_min, self.val_max, self.n_palette, self.rng
        )

    def color_notes(self, arr, val_min=0.5, val_max=1.5, n_palette=100, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        arr = arr.astype(np.int32)
        colored = np.zeros_like(arr, dtype=np.float32)
        palette = rng.uniform(val_min, val_max, n_palette)
        for note_idx in range(arr.shape[1]):
            row = arr[:, note_idx]
            in_note = False
            start = 0
            for i, v in enumerate(np.append(row, 0)):
                if v == 1 and not in_note:
                    in_note = True
                    start = i
                elif v == 0 and in_note:
                    in_note = False
                    # Use hash of properties as index into palette
                    key = (note_idx, start, tuple(row[:start]))
                    color_idx = abs(hash(key)) % n_palette
                    value = palette[color_idx]
                    colored[start:i, note_idx] = value

        return colored


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
