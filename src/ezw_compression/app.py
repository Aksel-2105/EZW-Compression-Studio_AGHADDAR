from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .ezw import (
    decode_ezw,
    decode_lossless_wavelet,
    encode_ezw,
    encode_lossless_wavelet,
    simple_significance_pass,
)
from .image_utils import get_image_info, load_grayscale_image, prepare_image_for_processing, save_grayscale_image
from .metrics import compression_ratio, mean_squared_error, peak_signal_to_noise_ratio
from .paths import EXAMPLES_DIR, OUTPUTS_DIR
from .wavelet import (
    decompose_image,
    extract_main_subbands,
    normalize_for_display,
    plot_subbands,
    plot_wavelet_mosaic,
    reconstruct_image,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Demonstration complete du projet PFE EZW.")
    parser.add_argument(
        "--image",
        type=Path,
        default=EXAMPLES_DIR / "sample_input.png",
        help="Chemin de l'image a traiter.",
    )
    parser.add_argument("--wavelet", default="haar", help="Nom de l'ondelette PyWavelets.")
    parser.add_argument("--level", type=int, default=3, help="Niveau de decomposition en ondelettes.")
    parser.add_argument("--passes", type=int, default=6, help="Nombre de passes EZW.")
    parser.add_argument(
        "--mode",
        choices=("lossy", "lossless"),
        default="lossy",
        help="Mode de compression : avec perte (lossy) ou sans perte (lossless).",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=512,
        help="Taille maximale de la copie de travail pour garder l'application pedagogique fluide.",
    )
    parser.add_argument("--no-show", action="store_true", help="Sauvegarde les figures sans les afficher.")
    parser.add_argument("--gui", action="store_true", help="Lance l'interface graphique tkinter.")
    return parser.parse_args()


def print_image_info(title, image_array):
    image_info = get_image_info(image_array)
    print(title)
    print(f"  Shape : {image_info['shape']}")
    print(f"  Dtype : {image_info['dtype']}")
    print(f"  Valeur minimale : {image_info['min']}")
    print(f"  Valeur maximale : {image_info['max']}")


def print_subband_comments():
    print("\nInterpretation des sous-bandes :")
    print("  cA : approximation basse frequence, information globale de l'image.")
    print("  cH : details horizontaux, sensibles aux variations selon l'axe horizontal.")
    print("  cV : details verticaux, sensibles aux variations selon l'axe vertical.")
    print("  cD : details diagonaux, textures et contours obliques.")


def print_ezw_summary(simple_pass_result, compression_result):
    print("\nPhase EZW - base simple :")
    print(f"  Seuil initial T0 = {simple_pass_result['threshold']}")
    print(f"  POS : {simple_pass_result['counts']['POS']}")
    print(f"  NEG : {simple_pass_result['counts']['NEG']}")
    print(f"  NS  : {simple_pass_result['counts']['NS']}")

    print("\nPhase EZW - codage progressif :")
    for pass_index, pass_data in enumerate(compression_result.passes, start=1):
        counts = pass_data.dominant_counts
        print(
            f"  Passe {pass_index}: seuil={pass_data.threshold}, "
            f"POS={counts['POS']}, NEG={counts['NEG']}, IZ={counts['IZ']}, "
            f"ZTR={counts['ZTR']}, raffinements={len(pass_data.refinement_bits)}"
        )


def print_lossless_summary(compression_result):
    print("\nPhase compression sans perte :")
    print("  L'image est encodee dans un flux PNG optimise, sans perte.")
    print(f"  Format source : {compression_result.source_format}")
    print(f"  Format de sortie : {compression_result.output_format}")
    print(f"  Taille compressee binaire : {compression_result.compressed_file_bytes:.2f} octets")


def build_comparison_figure(original_image, reconstructed_image, save_path=None, show=True):
    error_image = np.abs(original_image - reconstructed_image)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    figure.suptitle("Comparaison image originale / image reconstruite")

    axes[0].imshow(original_image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Originale")
    axes[0].axis("off")

    axes[1].imshow(reconstructed_image, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Reconstruite")
    axes[1].axis("off")

    axes[2].imshow(normalize_for_display(error_image), cmap="inferno")
    axes[2].set_title("Erreur absolue")
    axes[2].axis("off")

    figure.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        figure.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(figure)


def run_cli_workflow(arguments):
    results_dir = OUTPUTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    original_image = load_grayscale_image(arguments.image)
    working_image = prepare_image_for_processing(original_image, max_size=arguments.max_size)

    print_image_info("Image originale :", original_image)
    if working_image.shape != original_image.shape:
        print_image_info("\nCopie de travail redimensionnee :", working_image)
        print("  Note : la version pedagogique de EZW travaille sur cette copie de travail.")

    coefficients, resolved_level = decompose_image(working_image, arguments.wavelet, arguments.level)
    cA, cH, cV, cD = extract_main_subbands(coefficients)
    print_subband_comments()

    subbands_path = results_dir / "subbands.png"
    mosaic_path = results_dir / "wavelet_mosaic.png"
    plot_subbands(cA, cH, cV, cD, save_path=subbands_path, show=not arguments.no_show)
    plot_wavelet_mosaic(
        coefficients,
        arguments.wavelet,
        resolved_level,
        save_path=mosaic_path,
        show=not arguments.no_show,
    )

    if arguments.mode == "lossless":
        lossless_output_path = results_dir / "lossless_compressed.png"
        compression_result = encode_lossless_wavelet(
            original_image,
            arguments.wavelet,
            source_path=arguments.image,
            output_path=lossless_output_path,
        )
        reconstructed_image = decode_lossless_wavelet(compression_result)
        print_lossless_summary(compression_result)
    else:
        simple_pass_result = simple_significance_pass(coefficients)
        compression_result = encode_ezw(
            coefficients,
            arguments.wavelet,
            image_shape=working_image.shape,
            max_passes=arguments.passes,
        )
        reconstructed_coefficients = decode_ezw(compression_result)
        print_ezw_summary(simple_pass_result, compression_result)
        reconstructed_image = reconstruct_image(
            reconstructed_coefficients,
            arguments.wavelet,
            output_shape=working_image.shape,
        )

    reference_image = original_image if arguments.mode == "lossless" else working_image
    mse_value = mean_squared_error(reference_image, reconstructed_image)
    psnr_value = peak_signal_to_noise_ratio(reference_image, reconstructed_image)
    if arguments.mode == "lossless":
        original_bits_for_display = compression_result.original_file_bytes * 8
        compressed_bits_for_display = compression_result.compressed_file_bytes * 8
    else:
        original_bits_for_display = compression_result.original_bits
        compressed_bits_for_display = compression_result.compressed_bits
    compression_value = compression_ratio(
        original_bits_for_display,
        compressed_bits_for_display,
    )

    print("\nPhase reconstruction et evaluation :")
    print(f"  Mode = {arguments.mode}")
    print(f"  MSE  = {mse_value:.6f}")
    print(f"  PSNR = {psnr_value:.6f} dB")
    print(f"  CR   = {compression_value:.6f}")
    print(f"  Taille originale  = {original_bits_for_display / 8.0:.2f} octets")
    print(f"  Taille compressee = {compressed_bits_for_display / 8.0:.2f} octets")
    if arguments.mode == "lossless":
        print(f"  Format source     = {compression_result.source_format}")
        print(f"  Format sortie     = {compression_result.output_format}")
        print(f"  Note              = {compression_result.user_message}")

    reconstructed_path = results_dir / "reconstructed.png"
    save_grayscale_image(reconstructed_path, reconstructed_image)

    comparison_path = results_dir / "comparison.png"
    build_comparison_figure(
        reference_image,
        reconstructed_image,
        save_path=comparison_path,
        show=not arguments.no_show,
    )

    print("\nFichiers generes dans le dossier outputs :")
    print(f"  {subbands_path}")
    print(f"  {mosaic_path}")
    print(f"  {comparison_path}")
    print(f"  {reconstructed_path}")


def main():
    arguments = parse_arguments()

    if arguments.gui:
        from .gui import launch_gui

        launch_gui()
        return

    run_cli_workflow(arguments)


if __name__ == "__main__":
    main()
