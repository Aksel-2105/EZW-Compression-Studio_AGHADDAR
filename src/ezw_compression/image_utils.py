from pathlib import Path
import io

import numpy as np
from PIL import Image, UnidentifiedImageError


def load_image_with_mode(path):
    """
    Lit une image locale en preservant son type principal :
    - mode "L" pour une image en niveaux de gris ;
    - mode "RGB" pour une image couleur ;
    - mode "RGBA" converti en "RGB" pour eviter les erreurs sur le canal alpha.

    Retourne :
    - image_array : tableau NumPy float64
    - detected_mode : mode source detecte
    - normalized_mode : mode reel utilise dans l'application ("L" ou "RGB")
    """
    image_path = Path(path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Fichier image introuvable : {image_path}")

    try:
        with Image.open(image_path) as image:
            detected_mode = image.mode or "UNKNOWN"

            if detected_mode == "L":
                normalized_image = image.convert("L")
                normalized_mode = "L"
            elif detected_mode in {"RGB", "RGBA"}:
                normalized_image = image.convert("RGB")
                normalized_mode = "RGB"
            elif len(image.getbands()) == 1:
                normalized_image = image.convert("L")
                normalized_mode = "L"
            else:
                normalized_image = image.convert("RGB")
                normalized_mode = "RGB"

            image_array = np.asarray(normalized_image, dtype=np.float64)
    except UnidentifiedImageError as error:
        raise ValueError(f"Format d'image invalide ou non supporte : {image_path}") from error

    if image_array.size == 0:
        raise ValueError("L'image chargee est vide.")

    if normalized_mode == "L" and image_array.ndim != 2:
        raise ValueError("La conversion en niveaux de gris a echoue.")

    if normalized_mode == "RGB" and (image_array.ndim != 3 or image_array.shape[2] != 3):
        raise ValueError("La conversion en image couleur RGB a echoue.")

    return image_array, detected_mode, normalized_mode


def load_grayscale_image(path):
    """
    Lit une image locale, la convertit en niveaux de gris
    et retourne un tableau NumPy de type float64.
    """
    image_array, _, _ = load_image_with_mode(path)
    if image_array.ndim == 3:
        image_array = np.asarray(Image.fromarray(ensure_uint8_image(image_array), mode="RGB").convert("L"), dtype=np.float64)
    return image_array


def get_image_info(image_array):
    """
    Retourne les informations principales de l'image sous forme de dictionnaire.
    """
    if image_array.size == 0:
        raise ValueError("Impossible de decrire une image vide.")

    return {
        "shape": image_array.shape,
        "dtype": image_array.dtype,
        "min": float(np.min(image_array)),
        "max": float(np.max(image_array)),
    }


def ensure_uint8_image(image_array):
    """
    Convertit un tableau image vers uint8 dans l'intervalle [0, 255].
    """
    if image_array.size == 0:
        raise ValueError("Impossible de convertir une image vide.")

    clipped_array = np.clip(np.rint(image_array), 0, 255)
    return clipped_array.astype(np.uint8)


def prepare_image_for_processing(image_array, max_size=512):
    """
    Cree une copie de travail eventuellement redimensionnee pour les etapes
    scientifiques plus couteuses comme EZW.
    """
    if max_size is None:
        return np.asarray(image_array, dtype=np.float64).copy()

    if max_size <= 0:
        raise ValueError("max_size doit etre un entier strictement positif.")

    rows, cols = image_array.shape[:2]
    longest_side = max(rows, cols)

    if longest_side <= max_size:
        return np.asarray(image_array, dtype=np.float64).copy()

    scale = max_size / float(longest_side)
    new_cols = max(1, int(round(cols * scale)))
    new_rows = max(1, int(round(rows * scale)))

    image_uint8 = ensure_uint8_image(image_array)
    if image_uint8.ndim == 3:
        pil_image = Image.fromarray(image_uint8, mode="RGB")
    else:
        pil_image = Image.fromarray(image_uint8, mode="L")
    resized_image = pil_image.resize((new_cols, new_rows), Image.Resampling.LANCZOS)

    return np.asarray(resized_image, dtype=np.float64)


def save_image(path, image_array):
    """
    Sauvegarde une image grayscale ou RGB selon ses dimensions.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_uint8 = ensure_uint8_image(image_array)
    if image_uint8.ndim == 3:
        output_image = Image.fromarray(image_uint8, mode="RGB")
    else:
        output_image = Image.fromarray(image_uint8, mode="L")

    extension = output_path.suffix.lower()

    save_options = {}
    if extension == ".png":
        save_options = {"format": "PNG", "optimize": True, "compress_level": 9}
    elif extension in {".jpg", ".jpeg"}:
        save_options = {"format": "JPEG", "quality": 95, "optimize": True}
    elif extension in {".tif", ".tiff"}:
        save_options = {"format": "TIFF", "compression": "tiff_lzw"}

    if save_options:
        output_image.save(output_path, **save_options)
    else:
        output_image.save(output_path)

    return output_path


def save_grayscale_image(path, image_array):
    """
    Sauvegarde une image en niveaux de gris.
    """
    return save_image(path, image_array)


def get_file_size_bytes(path):
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    return int(file_path.stat().st_size)


def get_image_format(path):
    image_path = Path(path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Fichier image introuvable : {image_path}")

    try:
        with Image.open(image_path) as image:
            return (image.format or image_path.suffix.replace(".", "")).upper()
    except UnidentifiedImageError as error:
        raise ValueError(f"Format d'image invalide ou non supporte : {image_path}") from error


def encode_lossless_png_payload(image_array):
    """
    Encode l'image en PNG optimise, sans perte, et retourne les octets obtenus.
    """
    image_uint8 = ensure_uint8_image(image_array)
    if image_uint8.ndim == 3:
        image = Image.fromarray(image_uint8, mode="RGB")
    else:
        image = Image.fromarray(image_uint8, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, compress_level=9)
    return buffer.getvalue()
