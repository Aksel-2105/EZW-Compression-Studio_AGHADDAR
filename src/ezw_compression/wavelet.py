from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pywt


def validate_image_array(image_array):
    image_array = np.asarray(image_array, dtype=np.float64)

    if image_array.size == 0:
        raise ValueError("L'image est vide.")

    if image_array.ndim != 2:
        raise ValueError("Une image en niveaux de gris doit etre une matrice 2D.")

    return image_array


def maximum_wavelet_level(image_array, wavelet_name="haar"):
    image_array = validate_image_array(image_array)
    wavelet = pywt.Wavelet(wavelet_name)
    rows, cols = image_array.shape

    row_level = pywt.dwt_max_level(rows, wavelet.dec_len)
    col_level = pywt.dwt_max_level(cols, wavelet.dec_len)

    return min(row_level, col_level)


def resolve_wavelet_level(image_array, wavelet_name="haar", level=None):
    max_level = maximum_wavelet_level(image_array, wavelet_name)

    if max_level < 1:
        raise ValueError("L'image est trop petite pour une decomposition en ondelettes.")

    if level is None:
        return min(3, max_level)

    if level < 1 or level > max_level:
        raise ValueError(
            f"Niveau invalide : {level}. Niveau maximal autorise pour cette image : {max_level}."
        )

    return level


def decompose_image(image_array, wavelet_name="haar", level=None):
    image_array = validate_image_array(image_array)
    resolved_level = resolve_wavelet_level(image_array, wavelet_name, level)
    coefficients = pywt.wavedec2(image_array, wavelet_name, level=resolved_level)
    return coefficients, resolved_level


def reconstruct_image(coefficients, wavelet_name="haar", output_shape=None):
    reconstructed = pywt.waverec2(coefficients, wavelet_name)

    if output_shape is not None:
        rows, cols = output_shape
        reconstructed = reconstructed[:rows, :cols]

    return np.clip(reconstructed, 0.0, 255.0)


def extract_main_subbands(coefficients):
    if len(coefficients) < 2:
        raise ValueError("Les coefficients doivent contenir au moins un niveau de details.")

    cA = coefficients[0]
    cH, cV, cD = coefficients[1]
    return cA, cH, cV, cD


def normalize_for_display(array, center_zero=False, percentile=99.0):
    array = np.asarray(array, dtype=np.float64)

    if array.size == 0:
        raise ValueError("Impossible de normaliser un tableau vide.")

    if center_zero:
        abs_array = np.abs(array)
        scale = float(np.percentile(abs_array, percentile))

        if np.isclose(scale, 0.0):
            scale = float(np.max(abs_array))

        if np.isclose(scale, 0.0):
            return np.full_like(array, 127.0)

        clipped = np.clip(array, -scale, scale)
        return 255.0 * (clipped + scale) / (2.0 * scale)

    min_value = float(np.percentile(array, 100.0 - percentile))
    max_value = float(np.percentile(array, percentile))

    if np.isclose(min_value, max_value):
        return np.zeros_like(array)

    clipped = np.clip(array, min_value, max_value)
    return 255.0 * (clipped - min_value) / (max_value - min_value)


def build_wavelet_mosaic(coefficients):
    coefficient_array, coeff_slices = pywt.coeffs_to_array(coefficients)
    magnitude_array = np.log1p(np.abs(coefficient_array))
    mosaic = normalize_for_display(magnitude_array, percentile=99.5)
    return mosaic, coeff_slices


def _draw_subband_boundaries(axis, coeff_slices, line_color="white"):
    line_width = 1.2

    def slice_bounds(slice_object):
        start = 0 if slice_object.start is None else slice_object.start
        stop = slice_object.stop
        if stop is None:
            raise ValueError("Les bornes de sous-bandes doivent etre finies pour l'affichage.")
        return start, stop

    approx_slice = coeff_slices[0]
    approx_row_start, approx_row_stop = slice_bounds(approx_slice[0])
    approx_col_start, approx_col_stop = slice_bounds(approx_slice[1])
    axis.add_patch(
        plt.Rectangle(
            (approx_col_start, approx_row_start),
            approx_col_stop - approx_col_start,
            approx_row_stop - approx_row_start,
            fill=False,
            edgecolor=line_color,
            linewidth=1.5,
        )
    )

    for detail_slices in coeff_slices[1:]:
        for subband_name in ("ad", "da", "dd"):
            row_slice, col_slice = detail_slices[subband_name]
            row_start, row_stop = slice_bounds(row_slice)
            col_start, col_stop = slice_bounds(col_slice)
            axis.add_patch(
                plt.Rectangle(
                    (col_start, row_start),
                    col_stop - col_start,
                    row_stop - row_start,
                    fill=False,
                    edgecolor=line_color,
                    linewidth=line_width,
                )
            )


def plot_subbands(cA, cH, cV, cD, save_path=None, show=True):
    figure, axes = plt.subplots(2, 2, figsize=(10, 10))
    figure.suptitle("Sous-bandes de la transformee en ondelettes")

    subband_data = [
        (cA, "cA - approximation", "gray", False),
        (cH, "cH - details horizontaux", "seismic", True),
        (cV, "cV - details verticaux", "seismic", True),
        (cD, "cD - details diagonaux", "seismic", True),
    ]

    for axis, (data, title, cmap, center_zero) in zip(axes.flat, subband_data):
        axis.imshow(normalize_for_display(data, center_zero=center_zero), cmap=cmap)
        axis.set_title(title)
        axis.axis("off")

    figure.text(
        0.5,
        0.02,
        "Pour cH, cV et cD : rouge/bleu indiquent le signe du coefficient, "
        "le contraste a ete renforce pour mieux voir les details.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        figure.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(figure)


def plot_wavelet_mosaic(coefficients, wavelet_name, level, save_path=None, show=True):
    mosaic, coeff_slices = build_wavelet_mosaic(coefficients)

    figure, axis = plt.subplots(figsize=(9, 9))
    axis.imshow(mosaic, cmap="magma")
    _draw_subband_boundaries(axis, coeff_slices)
    axis.set_title(
        f"Vue d'ensemble des coefficients - ondelette {wavelet_name}, niveau {level}\n"
        "Affichage en log(1 + |coefficient|) pour renforcer les faibles details"
    )
    axis.axis("off")
    figure.tight_layout()

    if save_path is not None:
        figure.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(figure)

    return mosaic
