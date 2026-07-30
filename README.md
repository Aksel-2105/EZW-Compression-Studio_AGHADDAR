# EZW Compression Studio

Professional academic desktop application for image compression based on the **Embedded Zerotree Wavelet (EZW)** algorithm.

## Project Overview

**EZW Compression Studio** is the final and more advanced version of an undergraduate image-compression application developed for academic presentation and experimentation. It combines scientific image processing, wavelet decomposition, pedagogical EZW coding, reconstruction, visualization, and evaluation inside a desktop interface built with Tkinter.

The project is designed both as:

- a reusable scientific code base,
- a demonstrator for a final-year academic project,
- a visual application for explaining EZW compression step by step.

## Academic Context

This application was created for an **undergraduate final-year project** in:

- **Mathematics Education and Applied Computing**

Author:

- **Abderrahmane Aghaddar**

## Main Features

- Load grayscale and RGB images.
- Automatically detect grayscale, RGB, or RGBA input.
- Convert RGBA images safely to RGB for processing.
- Perform 2D wavelet decomposition with configurable wavelet family and level.
- Display wavelet sub-bands and coefficient mosaics.
- Apply a pedagogical EZW-inspired lossy compression workflow.
- Support a lossless mode through optimized PNG payload storage.
- Reconstruct grayscale or RGB images after compression.
- Compare original and reconstructed images visually.
- Display an absolute-error map.
- Show EZW bitstream-oriented visualizations.
- Compute quantitative evaluation metrics:
  - MSE
  - PSNR
  - compression ratio
- Save reconstructed outputs.
- Switch between dark and light themes.

## EZW Algorithm Overview

The **Embedded Zerotree Wavelet (EZW)** algorithm is a progressive image-compression method that exploits the hierarchical structure of wavelet coefficients.

Its key principles are:

1. Transform the image into wavelet coefficients.
2. Select an initial threshold based on the maximum coefficient magnitude.
3. Traverse coefficients in a significance-oriented order.
4. Encode them progressively using symbols such as:
   - `POS`
   - `NEG`
   - `IZ`
   - `ZTR`
5. Refine the reconstruction over successive passes.

This project implements a **pedagogical and readable EZW-style workflow** intended for academic understanding rather than industrial entropy coding performance.

## Wavelet Decomposition Overview

The application uses 2D discrete wavelet transforms through **PyWavelets**. For each decomposition level, the image is separated into:

- `cA`: approximation coefficients
- `cH`: horizontal detail coefficients
- `cV`: vertical detail coefficients
- `cD`: diagonal detail coefficients

These bands are visualized both individually and as a global coefficient mosaic.

## Compression Workflow

1. Load an image.
2. Detect whether it is grayscale or RGB.
3. Prepare a working copy for interactive visualization.
4. Perform wavelet decomposition.
5. Run lossy EZW coding or lossless storage mode.
6. Reconstruct the image.
7. Compare original and reconstructed images.
8. Compute metrics and generate visual explanations.

## Supported Compression Modes

- **Lossy**: pedagogical EZW-style progressive significance coding
- **Lossless**: exact reconstruction through optimized PNG payload storage

## Evaluation Metrics

The application reports the most important quantitative measures for compression analysis:

- **MSE**: Mean Squared Error
- **PSNR**: Peak Signal-to-Noise Ratio
- **Compression Ratio**
- file size before and after compression
- additional error statistics in the GUI

## Technologies Used

- Python
- Tkinter
- NumPy
- Pillow
- Matplotlib
- PyWavelets

## Project Structure

```text
EZW-Compression-Studio/
├── main.py
├── src/
│   └── ezw_compression/
│       ├── __init__.py
│       ├── app.py
│       ├── ezw.py
│       ├── gui.py
│       ├── image_utils.py
│       ├── metrics.py
│       ├── paths.py
│       └── wavelet.py
├── assets/
│   ├── icons/
│   │   └── logo.png
│   ├── images/
│   └── screenshots/
├── examples/
│   └── sample_input.png
├── tests/
│   └── test_metrics.py
├── docs/
├── outputs/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── pyproject.toml
```

## Installation Instructions

1. Open a terminal in the project root.
2. Create a virtual environment.
3. Install the required dependencies.

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Usage Instructions

### Launch the graphical application

```powershell
python main.py
```

### Run the command-line scientific workflow

```powershell
python main.py --image examples/sample_input.png --no-show
```

## Example Workflow

1. Click **Load image**.
2. Choose the wavelet family and decomposition level.
3. Click **Decompose** to inspect the transformed representation.
4. Click **Compress** and **Reconstruct**.
5. Click **Compare** to inspect differences and quality metrics.
6. Open **Bitstream** for the pedagogical EZW symbol view.
7. Save the reconstructed result if needed.

## Screenshots

Add screenshots of the current interface inside:

```text
assets/screenshots/
```

You can then reference them here for GitHub presentation.

## Current Limitations

- The EZW workflow is educational and not optimized for maximum compression efficiency.
- Lossless mode relies on PNG payload storage rather than a full lossless wavelet bitstream coder.
- The GUI is desktop-oriented and not packaged as a cross-platform installer in this repository.
- Large images may still require resizing for smooth interactive demonstrations.

## Future Improvements

- Add entropy coding after EZW symbol generation.
- Add batch-image compression experiments.
- Add automated regression tests for GUI-independent workflows.
- Export reports with compression metrics and figures.
- Provide a standalone packaged release for non-technical users.

## Author

- **Abderrahmane Aghaddar**
- **Project type:** Undergraduate final-year project
- **Field:** Mathematics Education and Applied Computing

## Recommended GitHub Description

Academic Python desktop application for EZW-based image compression, wavelet decomposition, reconstruction, and visual bitstream exploration.

## Recommended GitHub Topics

- python
- image-compression
- ezw
- wavelet-transform
- signal-processing
- tkinter
- image-processing
- applied-mathematics
