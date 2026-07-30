import numpy as np


def mean_squared_error(original_image, reconstructed_image):
    original_image = np.asarray(original_image, dtype=np.float64)
    reconstructed_image = np.asarray(reconstructed_image, dtype=np.float64)

    if original_image.shape != reconstructed_image.shape:
        raise ValueError("Les deux images doivent avoir les memes dimensions.")

    squared_error = (original_image - reconstructed_image) ** 2
    return float(np.mean(squared_error))


def peak_signal_to_noise_ratio(original_image, reconstructed_image, max_value=255.0):
    mse_value = mean_squared_error(original_image, reconstructed_image)

    if mse_value == 0:
        return float("inf")

    return float(10.0 * np.log10((max_value ** 2) / mse_value))


def compression_ratio(original_bits, compressed_bits):
    if compressed_bits <= 0:
        raise ValueError("La taille compressee doit etre strictement positive.")

    return float(original_bits / compressed_bits)
