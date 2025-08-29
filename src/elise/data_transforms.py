import numpy as np


class ColorNotes:
    def __init__(self, val_min=0.3, val_max=1.2, n_palette=100):
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
        return self.color_notes(arr)

    def color_notes(self, arr):
        arr = arr.astype(np.int32)
        colored = np.zeros_like(arr, dtype=np.float32)
        palette = self.rng.uniform(self.val_min, self.val_max, self.n_palette)
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
                    color_idx = abs(hash(key)) % self.n_palette
                    value = palette[color_idx]
                    colored[start:i, note_idx] = value

        return colored


class ChunkSplit:
    def __init__(self, num_chunks=5):
        self.num_chunks = num_chunks

    def __call__(self, x):
        len_x = x.shape[0]
        width_x = x.shape[1]
        new_pat = np.zeros((len_x, self.num_chunks * width_x))

        chunk_start_idx = np.array(
            np.linspace(0, len_x, self.num_chunks + 1), dtype=int
        )

        for i in range(self.num_chunks):
            row_start = width_x * i
            row_end = width_x * (i + 1)
            column_start = chunk_start_idx[i]
            column_end = chunk_start_idx[i + 1]
            chunk = x[column_start:column_end, :]
            new_pat[column_start:column_end, row_start:row_end] = chunk

        return new_pat


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


class Silence:
    def __init__(self, proportion=1 / 3):
        self.proportion = proportion

    def __call__(self, x):
        """
        Apply silence to the middle 1/proportion of the input.
        """
        len_x = x.shape[0]
        silence_mask = np.zeros(len_x, dtype=bool)
        silence_start = int(len_x * (1 - self.proportion) / 2)
        silence_end = int(len_x * (1 + self.proportion) / 2)
        silence_mask[silence_start:silence_end] = True
        x[silence_mask] = 0.0

        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test silence on 10 x 100 random values
    silence_transform = Silence(proportion=0.5)
    test_data = np.random.rand(20, 10)
    transformed_data = silence_transform(test_data)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(test_data.T, aspect="auto", cmap="viridis")
    ax[0].set_title("Original Data")
    ax[1].imshow(transformed_data.T, aspect="auto", cmap="viridis")
    ax[1].set_title("Transformed Data with Silence")
    plt.tight_layout()
    plt.show()

    print("Original data:\n", test_data)
    print("Transformed data with silence:\n", transformed_data)
