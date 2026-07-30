from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .image_utils import encode_lossless_png_payload, get_file_size_bytes


@dataclass
class EZWPass:
    threshold: float
    dominant_symbols: list[str]
    refinement_bits: list[int]
    dominant_counts: dict[str, int]


@dataclass
class EZWCompressionResult:
    wavelet_name: str
    level: int
    coeff_shapes: dict[tuple[str, int], tuple[int, int]]
    initial_threshold: float
    passes: list[EZWPass]
    original_bits: int
    compressed_bits: int


@dataclass
class LosslessCompressionResult:
    wavelet_name: str
    level: int
    payload: bytes
    original_bits: int
    compressed_bits: int
    original_file_bytes: int
    compressed_file_bytes: int
    source_format: str
    output_format: str
    payload_path: str | None
    user_message: str


def coeffs_to_band_arrays(coefficients):
    level = len(coefficients) - 1

    if level < 1:
        raise ValueError("EZW demande au moins un niveau de decomposition.")

    band_arrays = {("A", level): np.asarray(coefficients[0], dtype=np.float64).copy()}

    for index, detail_triplet in enumerate(coefficients[1:], start=1):
        current_level = level - index + 1
        cH, cV, cD = detail_triplet
        band_arrays[("H", current_level)] = np.asarray(cH, dtype=np.float64).copy()
        band_arrays[("V", current_level)] = np.asarray(cV, dtype=np.float64).copy()
        band_arrays[("D", current_level)] = np.asarray(cD, dtype=np.float64).copy()

    return band_arrays, level


def band_arrays_to_coeffs(band_arrays, level):
    coefficients = [np.asarray(band_arrays[("A", level)], dtype=np.float64).copy()]

    for current_level in range(level, 0, -1):
        coefficients.append(
            (
                np.asarray(band_arrays[("H", current_level)], dtype=np.float64).copy(),
                np.asarray(band_arrays[("V", current_level)], dtype=np.float64).copy(),
                np.asarray(band_arrays[("D", current_level)], dtype=np.float64).copy(),
            )
        )

    return coefficients


def compute_initial_threshold(coefficients):
    band_arrays, _ = coeffs_to_band_arrays(coefficients)
    max_coefficient = 0.0

    for array in band_arrays.values():
        if array.size > 0:
            max_coefficient = max(max_coefficient, float(np.max(np.abs(array))))

    if max_coefficient == 0.0:
        return 0.0

    return float(2 ** np.floor(np.log2(max_coefficient)))


def simple_significance_pass(coefficients, threshold=None):
    if threshold is None:
        threshold = compute_initial_threshold(coefficients)

    band_arrays, _ = coeffs_to_band_arrays(coefficients)
    counts = {"POS": 0, "NEG": 0, "NS": 0}

    for array in band_arrays.values():
        positive_mask = array >= threshold
        negative_mask = array <= -threshold
        non_significant_mask = ~(positive_mask | negative_mask)

        counts["POS"] += int(np.count_nonzero(positive_mask))
        counts["NEG"] += int(np.count_nonzero(negative_mask))
        counts["NS"] += int(np.count_nonzero(non_significant_mask))

    return {"threshold": threshold, "counts": counts}


def build_coeff_shapes(band_arrays):
    return {key: array.shape for key, array in band_arrays.items()}


def create_zero_band_arrays(coeff_shapes):
    return {key: np.zeros(shape, dtype=np.float64) for key, shape in coeff_shapes.items()}


def create_false_masks(coeff_shapes):
    return {key: np.zeros(shape, dtype=bool) for key, shape in coeff_shapes.items()}


def get_children(band, level, row, col, coeff_shapes):
    children = []

    if band == "A":
        for child_band in ("H", "V", "D"):
            child_shape = coeff_shapes.get((child_band, level))
            if child_shape is None:
                continue

            if row < child_shape[0] and col < child_shape[1]:
                children.append((child_band, level, row, col))

        return children

    if level <= 1:
        return children

    child_shape = coeff_shapes.get((band, level - 1))
    if child_shape is None:
        return children

    for delta_row in (0, 1):
        for delta_col in (0, 1):
            child_row = 2 * row + delta_row
            child_col = 2 * col + delta_col

            if child_row < child_shape[0] and child_col < child_shape[1]:
                children.append((band, level - 1, child_row, child_col))

    return children


def subtree_is_insignificant(band, level, row, col, threshold, band_arrays, coeff_shapes):
    current_value = band_arrays[(band, level)][row, col]
    if abs(current_value) >= threshold:
        return False

    for child in get_children(band, level, row, col, coeff_shapes):
        if not subtree_is_insignificant(*child, threshold, band_arrays, coeff_shapes):
            return False

    return True


def is_zerotree_root(band, level, row, col, threshold, band_arrays, coeff_shapes):
    children = get_children(band, level, row, col, coeff_shapes)

    if not children:
        return False

    for child in children:
        if not subtree_is_insignificant(*child, threshold, band_arrays, coeff_shapes):
            return False

    return True


def estimate_compressed_bits(passes):
    header_bits = 64
    stream_bits = 0

    for pass_data in passes:
        stream_bits += 2 * len(pass_data.dominant_symbols)
        stream_bits += len(pass_data.refinement_bits)

    return header_bits + stream_bits


DISPLAY_SYMBOL_MAP = {
    "POS": "P",
    "NEG": "N",
    "IZ": "Z",
    "ZTR": "T",
}

DISPLAY_TO_BINARY_MAP = {
    "P": "11",
    "N": "10",
    "T": "01",
    "Z": "00",
}


def collect_first_pass_symbols(coefficients):
    band_arrays, level = coeffs_to_band_arrays(coefficients)
    coeff_shapes = build_coeff_shapes(band_arrays)
    threshold = compute_initial_threshold(coefficients)

    if threshold == 0.0:
        return threshold, []

    dominant_symbols = []

    def encode_node(band, current_level, row, col):
        current_value = band_arrays[(band, current_level)][row, col]

        if abs(current_value) >= threshold:
            dominant_symbols.append("POS" if current_value >= 0 else "NEG")
            for child in get_children(band, current_level, row, col, coeff_shapes):
                encode_node(*child)
            return

        if is_zerotree_root(
            band,
            current_level,
            row,
            col,
            threshold,
            band_arrays,
            coeff_shapes,
        ):
            dominant_symbols.append("ZTR")
            return

        dominant_symbols.append("IZ")
        for child in get_children(band, current_level, row, col, coeff_shapes):
            encode_node(*child)

    approx_shape = coeff_shapes[("A", level)]
    for row in range(approx_shape[0]):
        for col in range(approx_shape[1]):
            encode_node("A", level, row, col)

    return threshold, dominant_symbols


def reshape_symbol_sequence(symbol_sequence, max_rows=10, max_cols=10):
    if not symbol_sequence:
        return []

    trimmed_sequence = symbol_sequence[: max_rows * max_cols]
    rows = []

    for index in range(0, len(trimmed_sequence), max_cols):
        rows.append(trimmed_sequence[index : index + max_cols])

    return rows


def symbols_to_binary_matrix(symbol_matrix):
    return [[DISPLAY_TO_BINARY_MAP.get(symbol, "") for symbol in row] for row in symbol_matrix]


def generate_bitstream(symbol_matrix):
    return "".join(
        DISPLAY_TO_BINARY_MAP.get(symbol, "")
        for row in symbol_matrix
        for symbol in row
    )


def generate_bitstream_visualization(coefficients, max_rows=10, max_cols=10):
    threshold, dominant_symbols = collect_first_pass_symbols(coefficients)
    display_symbols = [DISPLAY_SYMBOL_MAP.get(symbol, symbol) for symbol in dominant_symbols]
    symbol_matrix = reshape_symbol_sequence(display_symbols, max_rows=max_rows, max_cols=max_cols)
    binary_matrix = symbols_to_binary_matrix(symbol_matrix)
    bitstream = generate_bitstream(symbol_matrix)

    return {
        "threshold": threshold,
        "symbol_matrix": symbol_matrix,
        "binary_matrix": binary_matrix,
        "bitstream": bitstream,
        "displayed_symbol_count": sum(len(row) for row in symbol_matrix),
        "total_symbol_count": len(display_symbols),
    }


def encode_ezw(coefficients, wavelet_name, image_shape, max_passes=6):
    if max_passes < 1:
        raise ValueError("Le nombre de passes EZW doit etre au moins egal a 1.")

    band_arrays, level = coeffs_to_band_arrays(coefficients)
    coeff_shapes = build_coeff_shapes(band_arrays)
    significance_masks = create_false_masks(coeff_shapes)
    reconstructed_bands = create_zero_band_arrays(coeff_shapes)
    discovery_order = []

    initial_threshold = compute_initial_threshold(coefficients)

    if initial_threshold == 0.0:
        compressed_bits = estimate_compressed_bits([])
        return EZWCompressionResult(
            wavelet_name=wavelet_name,
            level=level,
            coeff_shapes=coeff_shapes,
            initial_threshold=0.0,
            passes=[],
            original_bits=int(image_shape[0] * image_shape[1] * 8),
            compressed_bits=compressed_bits,
        )

    passes = []
    threshold = initial_threshold

    for _ in range(max_passes):
        if threshold < 1.0e-9:
            break

        dominant_symbols = []
        dominant_counts = {"POS": 0, "NEG": 0, "IZ": 0, "ZTR": 0}
        refinement_bits = []
        previous_significant = list(discovery_order)

        def encode_node(band, current_level, row, col):
            if significance_masks[(band, current_level)][row, col]:
                for child in get_children(band, current_level, row, col, coeff_shapes):
                    encode_node(*child)
                return

            current_value = band_arrays[(band, current_level)][row, col]

            if abs(current_value) >= threshold:
                symbol = "POS" if current_value >= 0 else "NEG"
                dominant_symbols.append(symbol)
                dominant_counts[symbol] += 1
                significance_masks[(band, current_level)][row, col] = True
                reconstructed_bands[(band, current_level)][row, col] = np.sign(current_value) * (
                    threshold + threshold / 2.0
                )
                discovery_order.append((band, current_level, row, col))

                for child in get_children(band, current_level, row, col, coeff_shapes):
                    encode_node(*child)
                return

            if is_zerotree_root(
                band,
                current_level,
                row,
                col,
                threshold,
                band_arrays,
                coeff_shapes,
            ):
                dominant_symbols.append("ZTR")
                dominant_counts["ZTR"] += 1
                return

            dominant_symbols.append("IZ")
            dominant_counts["IZ"] += 1
            for child in get_children(band, current_level, row, col, coeff_shapes):
                encode_node(*child)

        approx_shape = coeff_shapes[("A", level)]
        for row in range(approx_shape[0]):
            for col in range(approx_shape[1]):
                encode_node("A", level, row, col)

        for band, current_level, row, col in previous_significant:
            exact_value = band_arrays[(band, current_level)][row, col]
            current_estimate = reconstructed_bands[(band, current_level)][row, col]
            refinement_bit = 1 if abs(exact_value) >= abs(current_estimate) else 0
            refinement_bits.append(refinement_bit)

            delta = threshold / 2.0
            sign = 1.0 if current_estimate >= 0 else -1.0

            if refinement_bit == 1:
                reconstructed_bands[(band, current_level)][row, col] += sign * delta
            else:
                reconstructed_bands[(band, current_level)][row, col] -= sign * delta

        passes.append(
            EZWPass(
                threshold=float(threshold),
                dominant_symbols=dominant_symbols,
                refinement_bits=refinement_bits,
                dominant_counts=dominant_counts,
            )
        )

        threshold /= 2.0

    compressed_bits = estimate_compressed_bits(passes)

    return EZWCompressionResult(
        wavelet_name=wavelet_name,
        level=level,
        coeff_shapes=coeff_shapes,
        initial_threshold=initial_threshold,
        passes=passes,
        original_bits=int(image_shape[0] * image_shape[1] * 8),
        compressed_bits=compressed_bits,
    )


def decode_ezw(compression_result):
    coeff_shapes = compression_result.coeff_shapes
    level = compression_result.level
    reconstructed_bands = create_zero_band_arrays(coeff_shapes)
    significance_masks = create_false_masks(coeff_shapes)
    discovery_order = []

    for pass_data in compression_result.passes:
        previous_significant = list(discovery_order)
        symbol_index = 0

        def decode_node(band, current_level, row, col):
            nonlocal symbol_index

            if significance_masks[(band, current_level)][row, col]:
                for child in get_children(band, current_level, row, col, coeff_shapes):
                    decode_node(*child)
                return

            if symbol_index >= len(pass_data.dominant_symbols):
                raise ValueError("Flux EZW incomplet pendant le decodage.")

            symbol = pass_data.dominant_symbols[symbol_index]
            symbol_index += 1

            if symbol in {"POS", "NEG"}:
                sign = 1.0 if symbol == "POS" else -1.0
                significance_masks[(band, current_level)][row, col] = True
                reconstructed_bands[(band, current_level)][row, col] = sign * (
                    pass_data.threshold + pass_data.threshold / 2.0
                )
                discovery_order.append((band, current_level, row, col))

                for child in get_children(band, current_level, row, col, coeff_shapes):
                    decode_node(*child)
                return

            if symbol == "IZ":
                for child in get_children(band, current_level, row, col, coeff_shapes):
                    decode_node(*child)
                return

            if symbol != "ZTR":
                raise ValueError(f"Symbole EZW inconnu : {symbol}")

        approx_shape = coeff_shapes[("A", level)]
        for row in range(approx_shape[0]):
            for col in range(approx_shape[1]):
                decode_node("A", level, row, col)

        for (band, current_level, row, col), refinement_bit in zip(
            previous_significant,
            pass_data.refinement_bits,
        ):
            current_estimate = reconstructed_bands[(band, current_level)][row, col]
            sign = 1.0 if current_estimate >= 0 else -1.0
            delta = pass_data.threshold / 2.0

            if refinement_bit == 1:
                reconstructed_bands[(band, current_level)][row, col] += sign * delta
            else:
                reconstructed_bands[(band, current_level)][row, col] -= sign * delta

    return band_arrays_to_coeffs(reconstructed_bands, level)


def build_lossless_user_message(source_format, original_file_bytes, compressed_file_bytes):
    if compressed_file_bytes <= original_file_bytes:
        reduction_ratio = 100.0 * (1.0 - (compressed_file_bytes / original_file_bytes))
        return (
            "Compression sans perte reussie : le fichier compresse conserve toutes les informations "
            f"et reduit la taille d'environ {reduction_ratio:.2f} %."
        )

    if source_format.upper() in {"JPG", "JPEG"}:
        return (
            "La taille finale est superieure a l'originale, ce qui est normal ici : "
            "le fichier source est deja en JPEG, un format avec perte souvent tres optimise, "
            "alors que la sortie lossless en PNG conserve toutes les informations sans approximation."
        )

    return (
        "La taille finale est superieure a l'originale, ce qui peut rester normal en compression sans perte : "
        "toutes les informations de l'image sont conservees, et certains contenus ou formats sources "
        "sont deja tres compacts ou peu favorables a une reduction suplementaire."
    )


def encode_lossless_wavelet(image_array, wavelet_name, source_path=None, output_path=None):
    """
    Compression sans perte coherente pour l'application :
    l'image grayscale ou RGB est encodee en PNG optimise, sans perte.
    """
    image_array = np.asarray(image_array, dtype=np.float64)
    payload = encode_lossless_png_payload(image_array)

    if source_path is not None:
        source_path = Path(source_path)
        original_file_bytes = get_file_size_bytes(source_path)
        source_format = source_path.suffix.replace(".", "").upper() or "IMAGE"
    else:
        original_file_bytes = int(image_array.size)
        source_format = "IMAGE"

    payload_path = None
    compressed_file_bytes = int(len(payload))
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)
        payload_path = str(output_path)
        compressed_file_bytes = get_file_size_bytes(output_path)

    user_message = build_lossless_user_message(source_format, original_file_bytes, compressed_file_bytes)

    return LosslessCompressionResult(
        wavelet_name=wavelet_name,
        level=0,
        payload=payload,
        original_bits=int(original_file_bytes * 8),
        compressed_bits=int(compressed_file_bytes * 8),
        original_file_bytes=original_file_bytes,
        compressed_file_bytes=compressed_file_bytes,
        source_format=source_format,
        output_format="PNG",
        payload_path=payload_path,
        user_message=user_message,
    )


def decode_lossless_wavelet(compression_result):
    """
    Reconstitue exactement l'image sauvegardee sans perte.
    """
    buffer = io.BytesIO(compression_result.payload)
    with Image.open(buffer) as image:
        if image.mode == "L":
            decoded_image = image.convert("L")
        else:
            decoded_image = image.convert("RGB")

        return np.asarray(decoded_image, dtype=np.float64)
