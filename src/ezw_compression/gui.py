from __future__ import annotations

from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib import cm
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageTk

from .ezw import (
    DISPLAY_SYMBOL_MAP,
    DISPLAY_TO_BINARY_MAP,
    collect_first_pass_symbols,
    decode_ezw,
    decode_lossless_wavelet,
    encode_ezw,
    encode_lossless_wavelet,
    generate_bitstream_visualization,
    simple_significance_pass,
)
from .image_utils import (
    ensure_uint8_image,
    get_file_size_bytes,
    get_image_format,
    get_image_info,
    load_image_with_mode,
    prepare_image_for_processing,
    save_image,
)
from .metrics import compression_ratio, mean_squared_error, peak_signal_to_noise_ratio
from .paths import ICONS_DIR, OUTPUTS_DIR, PROJECT_ROOT
from .wavelet import build_wavelet_mosaic, decompose_image, extract_main_subbands, normalize_for_display, reconstruct_image


class ThemeSegmentedControl(tk.Frame):
    def __init__(self, parent, on_change, initial_mode="dark"):
        super().__init__(parent, bd=0, highlightthickness=1, padx=4, pady=4)
        self.on_change = on_change
        self.current_mode = initial_mode
        self.theme_name = initial_mode
        self.palette = {}
        self.accent = "#2563EB"
        self.accent_hover = "#1D4ED8"
        self.accent_light = "#60A5FA"
        self.text_color = "#E5E7EB"
        self.muted_color = "#94A3B8"
        self.container_bg = "#0F172A"
        self.inactive_bg = "#111827"

        self.dark_button = self._create_segment_button("🌙  Dark", "dark")
        self.light_button = self._create_segment_button("☀  Light", "light")
        self.dark_button.pack(side="left", padx=(0, 4))
        self.light_button.pack(side="left")

    def _create_segment_button(self, text, mode):
        button = tk.Button(
            self,
            text=text,
            command=lambda selected_mode=mode: self.set_mode(selected_mode, invoke=True),
            relief="flat",
            bd=0,
            padx=14,
            pady=7,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
            highlightthickness=1,
        )
        button.bind("<Enter>", lambda _event, selected_mode=mode: self._on_enter(selected_mode))
        button.bind("<Leave>", lambda _event: self.refresh_styles())
        return button

    def apply_palette(
        self,
        theme_name,
        palette,
        accent,
        accent_hover,
        accent_light,
        text_color,
        muted_color,
    ):
        self.theme_name = theme_name
        self.palette = palette
        self.accent = accent
        self.accent_hover = accent_hover
        self.accent_light = accent_light
        self.text_color = text_color
        self.muted_color = muted_color
        self.container_bg = palette["panel"]
        self.inactive_bg = palette["input_bg"]
        self.refresh_styles()

    def set_mode(self, mode, invoke=False):
        self.current_mode = mode
        self.refresh_styles()
        if invoke and callable(self.on_change):
            self.on_change(mode)

    def _on_enter(self, mode):
        button = self.dark_button if mode == "dark" else self.light_button
        if mode == self.current_mode:
            button.configure(bg=self.accent_hover, fg="#FFFFFF", highlightbackground=self.accent_hover)
        else:
            button.configure(bg=self.accent_light, fg="#FFFFFF", highlightbackground=self.accent_light)

    def refresh_styles(self):
        self.configure(bg=self.container_bg, highlightbackground=self.palette.get("border", "#1E293B"))

        for mode, button in (("dark", self.dark_button), ("light", self.light_button)):
            if mode == self.current_mode:
                button.configure(
                    bg=self.accent,
                    fg="#FFFFFF",
                    activebackground=self.accent_hover,
                    activeforeground="#FFFFFF",
                    highlightbackground=self.accent,
                )
            else:
                button.configure(
                    bg=self.inactive_bg,
                    fg=self.muted_color,
                    activebackground=self.accent_light,
                    activeforeground="#FFFFFF",
                    highlightbackground=self.palette.get("border", "#1E293B"),
                )


class EZWApp(tk.Tk):
    ACCENT_BORDER = "#2563EB"
    ACCENT_TITLE = "#3B82F6"
    ACCENT_LABEL = "#60A5FA"
    ACCENT_HOVER = "#1D4ED8"
    COLOR_SUCCESS = "#22C55E"
    COLOR_ERROR = "#DC2626"
    COLOR_DARK_TEXT = "#0D0D0D"
    COLOR_WHITE = "#FFFFFF"
    PALETTES = {
        "dark": {
            "bg": "#050A0F",
            "panel": "#111827",
            "panel_alt": "#161B22",
            "border": "#253041",
            "text": "#E5E7EB",
            "muted": "#94A3B8",
            "input_bg": "#111827",
            "image_bg": "#050A0F",
            "info_bg": "#0B0F14",
        },
        "light": {
            "bg": "#F8FAFC",
            "panel": "#FFFFFF",
            "panel_alt": "#F8FAFC",
            "border": "#D5DEE8",
            "text": "#0F172A",
            "muted": "#475569",
            "input_bg": "#FFFFFF",
            "image_bg": "#F8FAFC",
            "info_bg": "#FFFFFF",
        },
    }
    PANEL_TITLE_ICONS = {
        "Originale": "🖼",
        "Image originale": "🖼",
        "Reconstruite": "🔗",
        "Image reconstruite": "🔗",
        "Erreur absolue": "〽",
        "Sous-bandes / coefficients": "▦",
        "Vue : cA / cH / cV / cD": "▦",
    }

    MODE_DISPLAY_TO_INTERNAL = {
        "avec perte": "lossy",
        "sans perte": "lossless",
    }
    MODE_INTERNAL_TO_DISPLAY = {
        "lossy": "avec perte",
        "lossless": "sans perte",
    }

    def __init__(self):
        super().__init__()
        self.title("Compression d'images par EZW")
        self.geometry("1280x820")

        self.project_root = PROJECT_ROOT
        self.results_dir = OUTPUTS_DIR
        self.header_logo_path = ICONS_DIR / "logo.png"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.image_path = None
        self.original_image = None
        self.working_image = None
        self.analysis_image = None
        self.coefficients = None
        self.channel_coefficients = None
        self.compression_result = None
        self.reconstructed_image = None
        self.compression_mode = None
        self.latest_comparison_metrics = None
        self.original_file_bytes = None
        self.original_file_format = None
        self.loaded_image_mode = None
        self.loaded_image_kind = None
        self.alpha_handling_note = None
        self.resolved_level = None
        self.error_image = None
        self.current_view_state = "empty"
        self.bitstream_data = None
        self.last_compression_time_ms = None
        self.last_reconstruction_time_ms = None

        self.wavelet_var = tk.StringVar(value="haar")
        self.level_var = tk.IntVar(value=3)
        self.passes_var = tk.IntVar(value=6)
        self.max_size_var = tk.IntVar(value=512)
        self.mode_var = tk.StringVar(value="avec perte")
        self.info_status_var = tk.StringVar(value="Clique sur le bouton Informations pour afficher les details.")
        self.status_var = tk.StringVar(value="Pret")
        self.info_history = []
        self.information_visible = False
        self.information_button = None
        self.theme_toggle = None
        self.theme_name = "dark"
        self._themed_widgets = []
        self.header_logo_label = None

        self.original_photo = None
        self.wavelet_photo = None
        self.reconstructed_photo = None
        self.error_photo = None
        self.header_logo_photo = None

        self._build_layout()

    def _setup_visual_theme(self):
        palette = self._palette()
        self.configure(bg=palette["bg"])
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "TEntry",
            fieldbackground=palette["input_bg"],
            background=palette["input_bg"],
            foreground=palette["text"],
            bordercolor=palette["border"],
            lightcolor=palette["border"],
            darkcolor=palette["border"],
            padding=4,
        )
        style.configure(
            "TCombobox",
            fieldbackground=palette["input_bg"],
            background=palette["input_bg"],
            foreground=palette["text"],
            bordercolor=palette["border"],
            arrowcolor=palette["text"],
            padding=4,
        )
        style.map("TCombobox", fieldbackground=[("readonly", palette["input_bg"])])
        style.configure(
            "TSpinbox",
            fieldbackground=palette["input_bg"],
            background=palette["input_bg"],
            foreground=palette["text"],
            bordercolor=palette["border"],
            lightcolor=palette["border"],
            darkcolor=palette["border"],
            arrowsize=13,
            padding=4,
        )
        style.configure("Accent.TLabelframe", bordercolor=self.ACCENT_BORDER, relief="solid")
        style.configure(
            "Accent.TLabelframe.Label",
            foreground=self.ACCENT_TITLE,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure("TFrame", background=palette["panel"])
        style.configure("TLabel", background=palette["panel"], foreground=palette["text"])
        style.configure("TNotebook", background=palette["bg"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=palette["panel_alt"],
            foreground=palette["text"],
            padding=(14, 8),
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", palette["panel"])],
            foreground=[("selected", self.ACCENT_TITLE)],
        )

    def _palette(self):
        return self.PALETTES[self.theme_name]

    def _register_widget(self, widget, role):
        self._themed_widgets.append((widget, role))
        return widget

    def _create_panel(self, parent, padx=10, pady=4):
        palette = self._palette()
        frame = tk.Frame(
            parent,
            bg=palette["panel"],
            highlightbackground=palette["border"],
            highlightthickness=1,
            bd=0,
        )
        frame.pack(fill="x", padx=padx, pady=pady)
        return self._register_widget(frame, "panel")

    def _create_hover_button(self, parent, text, command, accent=False):
        palette = self._palette()
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=palette["input_bg"],
            fg=palette["text"],
            activebackground=self.ACCENT_HOVER,
            activeforeground=self.COLOR_WHITE,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=self.ACCENT_BORDER if accent else palette["border"],
            padx=18,
            pady=9,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )
        button.bind("<Enter>", self._on_orange_enter)
        button.bind("<Leave>", self._on_orange_leave)
        return self._register_widget(button, "button_accent" if accent else "button")

    def create_modern_button(self, parent, icon, text, command=None, accent=False):
        return self._create_hover_button(parent, f"{icon}  {text}", command, accent=accent)

    def _create_toolbar_button(self, parent, icon, text, command, accent=False):
        return self.create_modern_button(parent, icon, text, command, accent=accent)

    def _create_hover_badge(self, parent, text):
        palette = self._palette()
        label = tk.Label(
            parent,
            text=text,
            bg=palette["panel"],
            fg=self.ACCENT_LABEL,
            padx=6,
            pady=2,
            font=("Segoe UI", 9, "bold"),
            relief="solid",
            bd=1,
            highlightbackground=self.ACCENT_BORDER,
            highlightthickness=1,
        )
        label.bind("<Enter>", self._on_orange_enter)
        label.bind("<Leave>", self._on_orange_leave)
        return self._register_widget(label, "badge")

    def _on_orange_enter(self, event):
        widget = event.widget
        if isinstance(widget, tk.Button):
            widget.configure(bg=self.ACCENT_HOVER, fg=self.COLOR_WHITE, highlightbackground=self.ACCENT_HOVER)
        else:
            widget.configure(bg=self.ACCENT_HOVER, fg=self.COLOR_WHITE)

    def _on_orange_leave(self, event):
        widget = event.widget
        palette = self._palette()
        role = next((role for item, role in self._themed_widgets if item is widget), "")
        if isinstance(widget, tk.Button):
            widget.configure(
                bg=palette["input_bg"],
                fg=palette["text"],
                highlightbackground=self.ACCENT_BORDER if role == "button_accent" else palette["border"],
            )
        else:
            widget.configure(bg=palette["panel"], fg=self.ACCENT_LABEL)

    def set_theme(self, theme_name):
        self.theme_name = theme_name
        palette = self._palette()
        self.configure(bg=palette["bg"])
        for widget, role in self._themed_widgets:
            if not widget.winfo_exists():
                continue
            if role == "root_bg":
                widget.configure(bg=palette["bg"])
            elif role == "panel":
                widget.configure(bg=palette["panel"], highlightbackground=palette["border"])
            elif role == "panel_alt":
                widget.configure(bg=palette["panel_alt"], highlightbackground=palette["border"])
            elif role == "title":
                widget.configure(bg=palette["bg"], fg=palette["text"])
            elif role == "subtitle":
                widget.configure(bg=palette["bg"], fg=palette["muted"])
            elif role == "label":
                widget.configure(bg=palette["panel"], fg=palette["text"])
            elif role == "accent_label":
                widget.configure(bg=palette["panel"], fg=self.ACCENT_TITLE)
            elif role == "button":
                widget.configure(bg=palette["input_bg"], fg=palette["text"], highlightbackground=palette["border"])
            elif role == "button_accent":
                widget.configure(bg=palette["input_bg"], fg=palette["text"], highlightbackground=self.ACCENT_BORDER)
            elif role == "badge":
                widget.configure(bg=palette["panel"], fg=self.ACCENT_LABEL, highlightbackground=self.ACCENT_BORDER)
            elif role == "text":
                widget.configure(
                    bg=palette["info_bg"],
                    fg=palette["text"],
                    insertbackground=palette["text"],
                    highlightbackground=palette["border"],
                )
            elif role == "image":
                widget.configure(bg=palette["image_bg"])
            elif role == "card":
                widget.configure(bg=palette["panel_alt"], highlightbackground=palette["border"])
            elif role == "accent_label_alt":
                widget.configure(bg=palette["panel_alt"], fg=self.ACCENT_TITLE)
            elif role == "label_alt":
                widget.configure(bg=palette["panel_alt"], fg=palette["muted"])
            elif role == "logo":
                widget.configure(bg=palette["bg"])
            elif role == "status_success":
                widget.configure(bg=palette["panel"], fg=self.COLOR_SUCCESS)
        self._setup_visual_theme()
        self.update_theme_buttons()
        self._update_header_logo()

    def update_theme_buttons(self):
        if self.theme_toggle is None:
            return

        self.theme_toggle.apply_palette(
            self.theme_name,
            self._palette(),
            self.ACCENT_BORDER,
            self.ACCENT_HOVER,
            self.ACCENT_LABEL,
            self._palette()["text"],
            self._palette()["muted"],
        )
        self.theme_toggle.set_mode(self.theme_name, invoke=False)

    def _create_header_logo_photo(self):
        if not self.header_logo_path.is_file():
            raise FileNotFoundError(f"Logo introuvable : {self.header_logo_path}")

        with Image.open(self.header_logo_path) as logo_image:
            prepared_logo = logo_image.convert("RGBA")
            prepared_logo.thumbnail((64, 64), self._get_resample_filter())
            return ImageTk.PhotoImage(prepared_logo)

    def _update_header_logo(self):
        if self.header_logo_label is None:
            return

        try:
            self.header_logo_photo = self._create_header_logo_photo()
            self.header_logo_label.configure(image=self.header_logo_photo, text="")
        except Exception:
            self.header_logo_photo = None
            self.header_logo_label.configure(image="", text="")

    def _on_scrollable_frame_configure(self, _event=None):
        if hasattr(self, "main_canvas"):
            self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        if hasattr(self, "content_window"):
            self.main_canvas.itemconfigure(self.content_window, width=event.width)

    def _on_mousewheel(self, event):
        if hasattr(self, "main_canvas"):
            self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _build_layout(self):
        self._setup_visual_theme()

        palette = self._palette()
        self.main_canvas = tk.Canvas(self, bg=palette["bg"], bd=0, highlightthickness=0)
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self._register_widget(self.main_canvas, "root_bg")

        self.vertical_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.main_canvas.yview)
        self.vertical_scrollbar.pack(side="right", fill="y")
        self.main_canvas.configure(yscrollcommand=self.vertical_scrollbar.set)

        self.content_frame = tk.Frame(self.main_canvas, bg=palette["bg"])
        self.content_window = self.main_canvas.create_window((0, 0), window=self.content_frame, anchor="nw")
        self._register_widget(self.content_frame, "root_bg")
        self.content_frame.bind("<Configure>", self._on_scrollable_frame_configure)
        self.main_canvas.bind("<Configure>", self._on_canvas_configure)
        self.main_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        header = tk.Frame(self.content_frame, bg=palette["bg"])
        header.pack(fill="x", padx=12, pady=(6, 3))
        self._register_widget(header, "root_bg")

        self.header_logo_label = tk.Label(
            header,
            bg=palette["bg"],
            relief="flat",
            bd=0,
            cursor="arrow",
        )
        self.header_logo_label.pack(side="left", padx=(0, 12))
        self._register_widget(self.header_logo_label, "logo")
        self._update_header_logo()

        title_box = tk.Frame(header, bg=palette["bg"])
        title_box.pack(side="left", fill="x", expand=True)
        self._register_widget(title_box, "root_bg")
        title = tk.Label(title_box, text="Compression d'images par EZW", bg=palette["bg"], fg=palette["text"], font=("Segoe UI", 22, "bold"))
        title.pack(anchor="w")
        self._register_widget(title, "title")
        subtitle = tk.Label(
            title_box,
            text="Compression multi-resolution par ondelettes et algorithme EZW",
            bg=palette["bg"],
            fg=palette["muted"],
            font=("Segoe UI", 10),
        )
        subtitle.pack(anchor="w")
        self._register_widget(subtitle, "subtitle")

        self.theme_toggle = ThemeSegmentedControl(
            header,
            on_change=self.set_theme,
            initial_mode=self.theme_name,
        )
        self.theme_toggle.pack(side="right", padx=(8, 0))
        self.update_theme_buttons()

        toolbar = self._create_panel(self.content_frame, padx=12, pady=2)
        for index in range(8):
            toolbar.columnconfigure(index, weight=1, uniform="toolbar")
        buttons = [
            ("📂", "Charger image", self.load_image, False),
            ("▦", "Decomposer", self.decompose_current_image, False),
            ("⚙", "Compresser", self.compress_current_image, False),
            ("↻", "Reconstruire", self.reconstruct_current_image, False),
            ("◫", "Comparer", self.compare_images, False),
            ("ⓘ", "Informations", self.show_information, True),
            ("1010", "Bitstream", self.show_bitstream_view, False),
            ("⇩", "Sauvegarder resultat", self.save_result, False),
        ]
        for index, (icon_text, text, command, accent) in enumerate(buttons):
            button = self._create_toolbar_button(toolbar, icon_text, text, command, accent=accent)
            if text == "Informations":
                self.information_button = button
            button.grid(
                row=0,
                column=index,
                padx=4,
                pady=5,
                sticky="ew",
            )

        params = self._create_panel(self.content_frame, padx=12, pady=3)
        params_title = tk.Label(params, text="Parametres de compression", bg=palette["panel"], fg=self.ACCENT_TITLE, font=("Segoe UI", 11, "bold"))
        params_title.grid(row=0, column=0, columnspan=8, sticky="w", padx=10, pady=(5, 2))
        self._register_widget(params_title, "accent_label")

        self._create_hover_badge(params, "Ondelette").grid(row=1, column=0, sticky="w", padx=(10, 6), pady=(2, 6))
        ttk.Combobox(
            params,
            width=14,
            state="readonly",
            textvariable=self.wavelet_var,
            values=("haar", "db2", "sym2", "coif1"),
        ).grid(row=1, column=1, sticky="w", padx=(0, 18), pady=(2, 6))
        self._create_hover_badge(params, "Niveau").grid(row=1, column=2, sticky="w", padx=(0, 6), pady=(2, 6))
        ttk.Spinbox(params, from_=1, to=4, width=12, textvariable=self.level_var).grid(row=1, column=3, sticky="w", padx=(0, 18), pady=(2, 6))
        self._create_hover_badge(params, "Passes EZW").grid(row=1, column=4, sticky="w", padx=(0, 6), pady=(2, 6))
        ttk.Spinbox(params, from_=1, to=12, width=12, textvariable=self.passes_var).grid(row=1, column=5, sticky="w", padx=(0, 18), pady=(2, 6))
        self._create_hover_badge(params, "Mode").grid(row=1, column=6, sticky="w", padx=(0, 6), pady=(2, 6))
        ttk.Combobox(params, width=14, state="readonly", textvariable=self.mode_var, values=("avec perte", "sans perte")).grid(row=1, column=7, sticky="w", padx=(0, 10), pady=(2, 6))

        self.images_frame = tk.Frame(self.content_frame, bg=palette["bg"])
        self.images_frame.pack(fill="both", expand=True, padx=12, pady=3)
        self._register_widget(self.images_frame, "root_bg")

        self.original_panel = self._create_image_panel(self.images_frame, "Originale", 0)
        self.wavelet_panel = self._create_image_panel(self.images_frame, "Sous-bandes / coefficients", 1)
        self.reconstructed_panel = self._create_image_panel(self.images_frame, "Reconstruite", 2)
        self.error_panel = self._create_image_panel(self.images_frame, "Erreur absolue", 3)

        self.metrics_frame = self._create_panel(self.content_frame, padx=12, pady=3)
        info_header = tk.Frame(self.metrics_frame, bg=palette["panel"])
        info_header.pack(fill="x", padx=10, pady=(5, 3))
        self._register_widget(info_header, "panel")
        info_title = tk.Label(info_header, text="Informations", bg=palette["panel"], fg=self.ACCENT_TITLE, font=("Segoe UI", 11, "bold"))
        info_title.pack(side="left")
        self._register_widget(info_title, "accent_label")
        self._create_hover_button(info_header, "🗑  Effacer", self.clear_information).pack(side="right")

        self.info_text = tk.Text(
            self.metrics_frame,
            height=7,
            wrap="word",
            bg=palette["info_bg"],
            fg=palette["text"],
            insertbackground=palette["text"],
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=palette["border"],
            font=("Consolas", 10),
        )
        self.info_text.pack(fill="x", expand=True, padx=10, pady=(0, 6))
        self._register_widget(self.info_text, "text")
        self.info_text.insert("1.0", "Aucune information affichee pour le moment.")
        self.info_text.configure(state="disabled")
        self.metrics_frame.pack_forget()

        self.status_bar = self._create_panel(self.content_frame, padx=12, pady=(2, 6))
        status_text = tk.Label(self.status_bar, textvariable=self.status_var, bg=palette["panel"], fg=self.COLOR_SUCCESS, font=("Segoe UI", 10, "bold"))
        status_text.pack(side="left", padx=10, pady=5)
        self._register_widget(status_text, "status_success")
        self.level_badge = self._create_hover_badge(self.status_bar, "Niveau : 3")
        self.level_badge.pack(side="right", padx=8)
        self.passes_badge = self._create_hover_badge(self.status_bar, "Passes : 6")
        self.passes_badge.pack(side="right", padx=8)
        self.mode_badge = self._create_hover_badge(self.status_bar, "Mode : avec perte")
        self.mode_badge.pack(side="right", padx=8)

        self._build_bitstream_view()

    def _build_bitstream_view(self):
        palette = self._palette()
        self.bitstream_view_frame = tk.Frame(self.content_frame, bg=palette["bg"], padx=8, pady=0)
        self._register_widget(self.bitstream_view_frame, "root_bg")

        header_frame = tk.Frame(self.bitstream_view_frame, bg=palette["bg"])
        header_frame.pack(fill="x", anchor="n", pady=(0, 0))
        self._register_widget(header_frame, "root_bg")

        self._create_hover_button(header_frame, "↩  Retour", self.return_to_main_view).pack(side="right", anchor="ne")

        self.bitstream_summary_frame = ttk.LabelFrame(
            self.bitstream_view_frame,
            text="Resume global",
            padding=8,
            style="Accent.TLabelframe",
        )
        self.bitstream_summary_frame.pack(fill="x", pady=(0, 8))
        self.bitstream_summary_text = tk.Text(
            self.bitstream_summary_frame,
            height=3,
            wrap="word",
            font=("Segoe UI", 9),
        )
        self.bitstream_summary_text.pack(fill="x", expand=True)
        self.bitstream_summary_text.configure(state="disabled")

        self.bitstream_notebook = ttk.Notebook(self.bitstream_view_frame)
        self.bitstream_notebook.pack(fill="both", expand=True)

        self.pixel_matrix_text = self._create_bitstream_tab(
            "Matrice image",
            "Matrice des pixels",
            "Cette matrice montre une portion numerique de l'image en niveaux de gris. "
            "Chaque valeur est un niveau d'intensite compris entre 0 et 255.",
        )
        self.symbol_matrix_text = self._create_bitstream_tab(
            "Symboles EZW",
            "Matrice symbolique P, N, Z, T",
            "Cette matrice represente la classification EZW des coefficients d'ondelettes. "
            "Elle illustre quels coefficients sont significatifs et quelles zones forment des zerotrees.",
        )
        self.binary_matrix_text = self._create_bitstream_tab(
            "Codage binaire",
            "Table de codage et matrice binaire",
            "Chaque symbole EZW est remplace par deux bits. La matrice binaire garde la meme organisation "
            "que la matrice symbolique affichee dans l'aperçu.",
        )
        self.bitstream_text = self._create_bitstream_tab(
            "Bitstream final",
            "Sequence binaire finale",
            "Le bitstream final est obtenu par concatenation des codes binaires. Il represente le flux binaire "
            "pedagogique produit par cette etape EZW.",
            add_copy_button=True,
        )

    def _create_bitstream_tab(self, tab_title, panel_title, explanation, add_copy_button=False):
        palette = self._palette()
        tab_frame = ttk.Frame(self.bitstream_notebook, padding=8)
        self.bitstream_notebook.add(tab_frame, text=tab_title)
        tab_frame.columnconfigure(0, weight=1)
        tab_frame.rowconfigure(0, weight=1)

        frame = ttk.LabelFrame(tab_frame, text=panel_title, padding=8, style="Accent.TLabelframe")
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        text = tk.Text(
            frame,
            wrap="none",
            font=("Courier New", 10),
            bg=palette["info_bg"],
            fg=palette["text"],
            insertbackground=palette["text"],
            relief="flat",
        )
        text.grid(row=0, column=0, sticky="nsew")
        self._register_widget(text, "text")

        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")

        text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set, state="disabled")

        bottom_frame = ttk.Frame(tab_frame)
        bottom_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        bottom_frame.columnconfigure(0, weight=1)
        ttk.Label(bottom_frame, text=explanation, justify="left", wraplength=1050).grid(row=0, column=0, sticky="w")

        if add_copy_button:
            self._create_hover_button(bottom_frame, "⧉  Copier le bitstream", self.copy_bitstream_to_clipboard).grid(
                row=0,
                column=1,
                sticky="e",
                padx=(8, 0),
            )

        return text

    def _create_image_panel(self, parent, title, column):
        palette = self._palette()
        panel = tk.Frame(
            parent,
            bg=palette["panel_alt"],
            highlightbackground=palette["border"],
            highlightthickness=1,
            bd=0,
        )
        panel.grid(row=0, column=column, padx=8, pady=6, sticky="nsew")
        parent.columnconfigure(column, weight=1)
        parent.rowconfigure(0, weight=1)

        title_label = tk.Label(
            panel,
            text=self._format_panel_title(title),
            bg=palette["panel_alt"],
            fg=self.ACCENT_TITLE,
            font=("Segoe UI", 12, "bold"),
            anchor="w",
        )
        title_label.pack(fill="x", padx=12, pady=(8, 4))

        image_label = tk.Label(panel, bg=palette["image_bg"], relief="flat")
        image_label.pack(fill="both", expand=True, padx=10, pady=(0, 5))

        footer_label = tk.Label(
            panel,
            text="",
            bg=palette["panel_alt"],
            fg=palette["muted"],
            font=("Segoe UI", 9),
            anchor="w",
        )
        footer_label.pack(fill="x", padx=12, pady=(0, 7))

        self._register_widget(panel, "card")
        self._register_widget(title_label, "accent_label_alt")
        self._register_widget(image_label, "image")
        self._register_widget(footer_label, "label_alt")
        return {"frame": panel, "title_label": title_label, "label": image_label, "footer_label": footer_label}

    def _format_panel_title(self, title):
        icon = self.PANEL_TITLE_ICONS.get(title)
        if not icon:
            return title

        return f"{icon}  {title}"

    def _array_to_photo(self, image_array):
        array = np.asarray(image_array)

        if array.ndim == 3:
            image = Image.fromarray(array.astype(np.uint8), mode="RGB")
        else:
            display_array = ensure_uint8_image(array)
            image = Image.fromarray(display_array.astype(np.uint8), mode="L")

        max_width, max_height = self._get_image_display_size()
        image = self._resize_pil_image_for_panel(image, max_width, max_height)
        return ImageTk.PhotoImage(image)

    def _is_color_image(self, image_array):
        array = np.asarray(image_array)
        return array.ndim == 3 and array.shape[2] >= 3

    def _convert_to_grayscale_array(self, image_array):
        array = ensure_uint8_image(image_array)
        if array.ndim == 3:
            grayscale_image = Image.fromarray(array, mode="RGB").convert("L")
            return np.asarray(grayscale_image, dtype=np.float64)

        return np.asarray(array, dtype=np.float64)

    def _get_analysis_image(self):
        if self.working_image is None:
            return None

        if self.analysis_image is None:
            self.analysis_image = self._convert_to_grayscale_array(self.working_image)

        return self.analysis_image

    def _get_mode_description(self):
        if self.loaded_image_kind is not None:
            return self.loaded_image_kind

        if self.original_image is None:
            return "Inconnu"

        return "Image couleur RGB" if self._is_color_image(self.original_image) else "Image en niveaux de gris"

    def _split_rgb_channels(self, image_array):
        array = np.asarray(image_array, dtype=np.float64)
        if not self._is_color_image(array):
            raise ValueError("Le decoupage RGB demande une image couleur.")

        return [array[:, :, channel_index] for channel_index in range(3)]

    def _merge_rgb_channels(self, channel_arrays):
        merged_array = np.stack(channel_arrays, axis=2)
        return np.asarray(ensure_uint8_image(merged_array), dtype=np.float64)

    def _get_resample_filter(self):
        try:
            return Image.Resampling.LANCZOS
        except AttributeError:
            return Image.LANCZOS

    def _get_image_display_size(self):
        window_width = max(self.winfo_width(), 1280)
        window_height = max(self.winfo_height(), 820)
        max_width = max(260, min(430, (window_width - 170) // 4))
        max_height = max(300, min(460, window_height - 430))
        return max_width, max_height

    def _resize_pil_image_for_panel(self, image, max_width, max_height):
        resized_image = image.copy()
        resized_image.thumbnail((max_width, max_height), self._get_resample_filter())
        return resized_image

    def _set_panel_title(self, panel, title):
        if "title_label" in panel:
            panel["title_label"].configure(text=self._format_panel_title(title))

    def _set_panel_image(self, panel, photo):
        panel["label"].configure(image=photo if photo is not None else "")

    def _set_panel_footer(self, panel, text):
        if "footer_label" in panel:
            panel["footer_label"].configure(text=text)

    def _clear_panel(self, panel, title):
        self._set_panel_title(panel, title)
        self._set_panel_image(panel, None)

    def _render_breakdown_photo(self):
        if self.coefficients is None:
            return None

        cA, cH, cV, cD = extract_main_subbands(self.coefficients)
        subbands = [
            (cA, "cA", "gray", False),
            (cH, "cH", "seismic", True),
            (cV, "cV", "seismic", True),
            (cD, "cD", "seismic", True),
        ]

        grid_size = 720
        outer_padding = 8
        separator = 18
        cell_size = (grid_size - (2 * outer_padding) - separator) // 2
        card_padding = 8
        label_height = 24
        image_width = cell_size - (2 * card_padding)
        image_height = cell_size - (2 * card_padding) - label_height

        canvas = Image.new("RGB", (grid_size, grid_size), (245, 247, 250))
        draw = ImageDraw.Draw(canvas)

        positions = [
            (outer_padding, outer_padding),
            (outer_padding + cell_size + separator, outer_padding),
            (outer_padding, outer_padding + cell_size + separator),
            (outer_padding + cell_size + separator, outer_padding + cell_size + separator),
        ]

        for (data, short_title, cmap_name, center_zero), (x, y) in zip(subbands, positions):
            normalized = normalize_for_display(data, center_zero=center_zero)

            if cmap_name == "gray":
                cell_image = Image.fromarray(normalized.astype(np.uint8), mode="L").convert("RGB")
            else:
                rgba = cm.get_cmap(cmap_name)(normalized / 255.0)
                rgb_array = (rgba[:, :, :3] * 255.0).astype(np.uint8)
                cell_image = Image.fromarray(rgb_array, mode="RGB")

            fitted_image = ImageOps.fit(cell_image, (image_width, image_height), method=self._get_resample_filter())

            draw.rounded_rectangle(
                (x, y, x + cell_size - 1, y + cell_size - 1),
                radius=10,
                fill=(252, 253, 255),
                outline=(211, 216, 223),
                width=1,
            )
            draw.rectangle(
                (
                    x + card_padding,
                    y + card_padding + label_height,
                    x + card_padding + image_width - 1,
                    y + card_padding + label_height + image_height - 1,
                ),
                outline=(222, 226, 232),
                width=1,
            )
            canvas.paste(fitted_image, (x + card_padding, y + card_padding + label_height))

            draw.text((x + card_padding, y + 6), short_title, fill=(20, 28, 38))

        return self._array_to_photo(np.asarray(canvas))

    def _render_error_photo(self, reference_image, reconstructed_image):
        absolute_error = np.abs(
            reference_image.astype(np.float32) - reconstructed_image.astype(np.float32)
        )
        if absolute_error.ndim == 3:
            absolute_error = np.mean(absolute_error, axis=2)

        max_error = float(np.max(absolute_error)) if absolute_error.size else 0.0

        if np.isclose(max_error, 0.0):
            rgb_array = np.zeros((absolute_error.shape[0], absolute_error.shape[1], 3), dtype=np.uint8)
            return self._array_to_photo(rgb_array)

        error_display = (absolute_error / max_error * 255.0).astype(np.uint8)
        rgba = cm.get_cmap("inferno")(error_display / 255.0)
        rgb_array = (rgba[:, :, :3] * 255.0).astype(np.uint8)
        return self._array_to_photo(rgb_array)

    def _compute_absolute_error_stats(self, reference_image, reconstructed_image):
        absolute_error = np.abs(
            reference_image.astype(np.float32) - reconstructed_image.astype(np.float32)
        )

        if absolute_error.size == 0:
            return 0.0, 0.0

        return float(np.max(absolute_error)), float(np.mean(absolute_error))

    def _show_loaded_image_breakdown_view(self):
        if self.original_image is None:
            return

        if self.coefficients is None and not self._compute_decomposition():
            return

        self.current_view_state = "loaded"

        self.original_photo = self._array_to_photo(self.original_image)
        self.wavelet_photo = self._render_breakdown_photo()
        self.reconstructed_photo = None
        self.error_photo = None

        self._set_panel_title(self.original_panel, "Originale")
        self._set_panel_title(self.wavelet_panel, "Vue : cA / cH / cV / cD")
        self._set_panel_title(self.reconstructed_panel, "Reconstruite")
        self._set_panel_title(self.error_panel, "Erreur absolue")
        self._set_panel_footer(
            self.original_panel,
            f"{self.original_image.shape[0]} x {self.original_image.shape[1]} px    {self.original_file_format or ''}    {self._get_mode_description()}",
        )
        wavelet_breakdown_footer = "Aperçu pedagogique des sous-bandes"
        if self._is_color_image(self.original_image):
            wavelet_breakdown_footer += " (projection en luminance)"
        self._set_panel_footer(self.wavelet_panel, wavelet_breakdown_footer)
        self._set_panel_footer(self.reconstructed_panel, "")
        self._set_panel_footer(self.error_panel, "Lance Comparer pour afficher l'erreur")
        self.status_var.set("Image chargee avec succes")
        self._refresh_status_badges()

        self._set_panel_image(self.original_panel, self.original_photo)
        self._set_panel_image(self.wavelet_panel, self.wavelet_photo)
        self._set_panel_image(self.reconstructed_panel, None)
        self._set_panel_image(self.error_panel, None)

    def _show_decomposition_view(self):
        if self.original_image is None:
            return

        if self.coefficients is None and not self._compute_decomposition():
            return

        self.current_view_state = "decomposition"

        mosaic, _ = build_wavelet_mosaic(self.coefficients)
        self.original_photo = self._array_to_photo(self.original_image)
        self.wavelet_photo = self._array_to_photo(mosaic)

        self._set_panel_title(self.original_panel, "Originale")
        self._set_panel_title(self.wavelet_panel, "Sous-bandes / coefficients")
        self._set_panel_title(self.reconstructed_panel, "Reconstruite")
        self._set_panel_title(self.error_panel, "Erreur absolue")
        self._set_panel_footer(
            self.original_panel,
            f"{self.original_image.shape[0]} x {self.original_image.shape[1]} px    {self.original_file_format or ''}    {self._get_mode_description()}",
        )
        wavelet_footer = f"Ondelettes {self.wavelet_var.get()} - niveau {self.resolved_level}"
        if self._is_color_image(self.original_image):
            wavelet_footer += " - luminance pour l'affichage"
        self._set_panel_footer(self.wavelet_panel, wavelet_footer)
        self._set_panel_footer(self.reconstructed_panel, "")
        self._set_panel_footer(self.error_panel, "Lance Comparer pour afficher l'erreur")
        self.status_var.set("Decomposition en ondelettes prete")
        self._refresh_status_badges()

        self._set_panel_image(self.original_panel, self.original_photo)
        self._set_panel_image(self.wavelet_panel, self.wavelet_photo)
        self._set_panel_image(self.reconstructed_panel, self.reconstructed_photo)
        self._set_panel_image(self.error_panel, self.error_photo)

    def _show_reconstruction_view(self):
        self._show_decomposition_view()
        self.current_view_state = "reconstruction"
        self._set_panel_title(self.reconstructed_panel, "Reconstruite")
        self._set_panel_image(self.reconstructed_panel, self.reconstructed_photo)
        if self.reconstructed_image is not None:
            self._set_panel_footer(
                self.reconstructed_panel,
                f"{self.reconstructed_image.shape[0]} x {self.reconstructed_image.shape[1]} px    EZW ({self.mode_var.get()})",
            )
        self.status_var.set("Image reconstruite")
        self._refresh_status_badges()

    def _show_comparison_view(self):
        if self.original_image is None or self.reconstructed_image is None:
            return

        self.current_view_state = "comparison"

        reference_image = self.original_image if self.compression_mode == "lossless" else self.working_image
        self.original_photo = self._array_to_photo(reference_image)
        if self.coefficients is not None:
            mosaic, _ = build_wavelet_mosaic(self.coefficients)
            self.wavelet_photo = self._array_to_photo(mosaic)
        self.reconstructed_photo = self._array_to_photo(self.reconstructed_image)
        self.error_photo = self._render_error_photo(reference_image, self.reconstructed_image)

        self._set_panel_title(self.original_panel, "Originale")
        self._set_panel_title(self.wavelet_panel, "Sous-bandes / coefficients")
        self._set_panel_title(self.reconstructed_panel, "Reconstruite")
        self._set_panel_title(self.error_panel, "Erreur absolue")
        self._set_panel_footer(
            self.original_panel,
            f"{reference_image.shape[0]} x {reference_image.shape[1]} px    {self._get_mode_description()}",
        )
        wavelet_footer = f"Ondelettes {self.wavelet_var.get()} - niveau {self.resolved_level}"
        if self._is_color_image(reference_image):
            wavelet_footer += " - luminance pour l'affichage"
        self._set_panel_footer(self.wavelet_panel, wavelet_footer)
        self._set_panel_footer(
            self.reconstructed_panel,
            f"{self.reconstructed_image.shape[0]} x {self.reconstructed_image.shape[1]} px    EZW ({self.mode_var.get()})",
        )
        self._set_panel_footer(self.error_panel, "Carte d'erreur absolue : noir = 0, inferno = erreur elevee")
        self.status_var.set("Comparaison affichee")
        self._refresh_status_badges()

        self._set_panel_image(self.original_panel, self.original_photo)
        self._set_panel_image(self.wavelet_panel, self.wavelet_photo)
        self._set_panel_image(self.reconstructed_panel, self.reconstructed_photo)
        self._set_panel_image(self.error_panel, self.error_photo)

    def _format_numeric_matrix_text(self, matrix):
        return "\n".join(" ".join(f"{int(value):>4}" for value in row) for row in matrix)

    def _format_string_matrix_text(self, matrix, cell_width=3):
        if not matrix:
            return "Aucune donnee disponible."

        return "\n".join(" ".join(f"{value:>{cell_width}}" for value in row) for row in matrix)

    def _set_text_widget_content(self, widget, content):
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def clear_information(self):
        self.info_history = []
        self._refresh_information_display()
        self.status_var.set("Informations effacees")

    def _refresh_status_badges(self):
        self.level_badge.configure(text=f"Niveau : {self.level_var.get()}")
        self.passes_badge.configure(text=f"Passes : {self.passes_var.get()}")
        self.mode_badge.configure(text=f"Mode : {self.mode_var.get()}")

    def copy_bitstream_to_clipboard(self):
        bitstream = self.generate_bitstream()
        if not bitstream:
            messagebox.showwarning("Information", "Aucun bitstream disponible a copier.")
            return

        self.clipboard_clear()
        self.clipboard_append(bitstream)
        self.update()
        messagebox.showinfo("Succes", "Bitstream copie dans le presse-papiers.")

    def _get_selected_mode_internal(self):
        return self.MODE_DISPLAY_TO_INTERNAL.get(self.mode_var.get(), "lossy")

    def _get_mode_display_label(self, internal_mode):
        return self.MODE_INTERNAL_TO_DISPLAY.get(internal_mode, internal_mode)

    def generate_image_matrix(self, max_rows=10, max_cols=10):
        if self.original_image is None:
            return None

        analysis_image = self._convert_to_grayscale_array(self.original_image)
        image_matrix = np.asarray(analysis_image[:max_rows, :max_cols], dtype=np.int32)
        return image_matrix

    def generate_ezw_symbol_matrix(self, max_rows=10, max_cols=10):
        if self.coefficients is None and not self._compute_decomposition():
            return None

        self.bitstream_data = generate_bitstream_visualization(
            self.coefficients,
            max_rows=max_rows,
            max_cols=max_cols,
        )
        threshold, dominant_symbols = collect_first_pass_symbols(self.coefficients)
        display_symbols = [DISPLAY_SYMBOL_MAP.get(symbol, symbol) for symbol in dominant_symbols]
        self.bitstream_data["threshold"] = threshold
        self.bitstream_data["all_display_symbols"] = display_symbols
        return self.bitstream_data

    def symbols_to_binary_matrix(self):
        if self.bitstream_data is None:
            return None
        return self.bitstream_data["binary_matrix"]

    def generate_bitstream(self):
        if self.bitstream_data is None:
            return ""
        return self.bitstream_data["bitstream"]

    def show_bitstream_view(self):
        if self.original_image is None:
            messagebox.showwarning("Information", "Veuillez d'abord choisir une image.")
            return

        bitstream_data = self.generate_ezw_symbol_matrix(max_rows=10, max_cols=10)
        if bitstream_data is None:
            messagebox.showwarning("Information", "Impossible de generer les donnees EZW.")
            return

        image_matrix = self.generate_image_matrix(max_rows=10, max_cols=10)
        binary_matrix = self.symbols_to_binary_matrix()
        bitstream = self.generate_bitstream()
        all_display_symbols = bitstream_data.get("all_display_symbols", [])

        original_rows, original_cols = self.original_image.shape[:2]
        preview_rows, preview_cols = image_matrix.shape[:2]

        symbol_count_total = len(all_display_symbols)
        symbol_preview_rows = len(bitstream_data["symbol_matrix"])
        symbol_preview_cols = max((len(row) for row in bitstream_data["symbol_matrix"]), default=0)
        symbol_counts = {
            symbol: all_display_symbols.count(symbol)
            for symbol in ("P", "N", "Z", "T")
        }

        def symbol_percent(symbol):
            if symbol_count_total == 0:
                return 0.0
            return 100.0 * symbol_counts[symbol] / symbol_count_total

        bitstream_length = len(bitstream)
        bitstream_size_bytes = (bitstream_length + 7) // 8 if bitstream_length > 0 else 0
        binary_preview_rows = len(binary_matrix) if binary_matrix else 0
        binary_preview_cols = max((len(row) for row in binary_matrix), default=0) if binary_matrix else 0

        summary_text = (
            f"Taille reelle de l'image : {original_rows} x {original_cols} pixels\n"
            f"Taille de l'aperçu affiche : {preview_rows} x {preview_cols}\n"
            f"Seuil initial EZW : {bitstream_data['threshold']} | "
            f"Total symboles : {symbol_count_total} | "
            f"Longueur bitstream : {bitstream_length} bits | "
            f"Taille approx. : {bitstream_size_bytes} octets"
        )

        pixel_text = self._format_numeric_matrix_text(image_matrix)
        pixel_text += (
            f"\n\nTaille reelle de la matrice : {original_rows} x {original_cols}"
            f"\nTaille de l'aperçu affiche : {preview_rows} x {preview_cols}"
            "\nValeurs de pixels en niveaux de gris comprises entre 0 et 255."
        )

        symbol_text = self._format_string_matrix_text(bitstream_data["symbol_matrix"], cell_width=2)
        symbol_text += (
            f"\n\nSeuil initial EZW : {bitstream_data['threshold']}"
            f"\nTaille reelle de la matrice symbolique : {symbol_count_total} symboles"
            f"\nTaille de l'aperçu affiche : {symbol_preview_rows} x {symbol_preview_cols}"
            f"\nSymboles affiches : {bitstream_data['displayed_symbol_count']} / {bitstream_data['total_symbol_count']}"
            f"\nNombre de P : {symbol_counts['P']} ({symbol_percent('P'):.2f} %)"
            f"\nNombre de N : {symbol_counts['N']} ({symbol_percent('N'):.2f} %)"
            f"\nNombre de Z : {symbol_counts['Z']} ({symbol_percent('Z'):.2f} %)"
            f"\nNombre de T : {symbol_counts['T']} ({symbol_percent('T'):.2f} %)"
            f"\nTotal des symboles : {symbol_count_total}"
            "\nP = coefficient significatif positif"
            "\nN = coefficient significatif negatif"
            "\nZ = zero isole"
            "\nT = racine de zerotree"
        )

        binary_text = (
            "Table de codage :\n"
            "P -> 11\n"
            "N -> 10\n"
            "T -> 01\n"
            "Z -> 00\n\n"
            "Matrice binaire :\n"
        )
        binary_text += self._format_string_matrix_text(binary_matrix, cell_width=2)
        binary_text += (
            f"\n\nTaille reelle equivalente : {symbol_count_total} symboles codes"
            f"\nTaille de l'aperçu affiche : {binary_preview_rows} x {binary_preview_cols}"
        )

        bitstream_lines = [
            bitstream if bitstream else "Bitstream : non disponible",
            "",
            f"Longueur totale du bitstream : {bitstream_length} bits",
            f"Taille approximative : {bitstream_size_bytes} octets",
            "Codage utilise : P=11, N=10, T=01, Z=00",
        ]

        self._set_text_widget_content(self.bitstream_summary_text, summary_text)
        self._set_text_widget_content(self.pixel_matrix_text, pixel_text)
        self._set_text_widget_content(self.symbol_matrix_text, symbol_text)
        self._set_text_widget_content(self.binary_matrix_text, binary_text)
        self._set_text_widget_content(self.bitstream_text, "\n".join(bitstream_lines))

        self.images_frame.pack_forget()
        self.metrics_frame.pack_forget()
        self.bitstream_view_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def return_to_main_view(self):
        self.bitstream_view_frame.pack_forget()
        self.images_frame.pack(fill="both", expand=True, padx=12, pady=3)
        if self.information_visible:
            self.metrics_frame.pack(fill="x", padx=10, pady=(0, 6), before=self.status_bar)

        if self.current_view_state == "comparison":
            self._show_comparison_view()
        elif self.current_view_state == "reconstruction":
            self._show_reconstruction_view()
        elif self.current_view_state == "decomposition":
            self._show_decomposition_view()
        elif self.current_view_state == "loaded":
            self._show_loaded_image_breakdown_view()

    def _compute_decomposition(self):
        try:
            analysis_image = self._get_analysis_image()
            if analysis_image is None:
                raise ValueError("Aucune image de travail n'est disponible pour la decomposition.")

            self.coefficients, self.resolved_level = decompose_image(
                analysis_image,
                self.wavelet_var.get(),
                self.level_var.get(),
            )

            if self._is_color_image(self.working_image):
                # Pour une image RGB, chaque canal est traite separement afin de
                # conserver la logique EZW existante, initialement concue pour une
                # matrice 2D en niveaux de gris.
                self.channel_coefficients = {}
                for channel_name, channel_array in zip(("R", "G", "B"), self._split_rgb_channels(self.working_image)):
                    channel_coefficients, _ = decompose_image(
                        channel_array,
                        self.wavelet_var.get(),
                        self.level_var.get(),
                    )
                    self.channel_coefficients[channel_name] = channel_coefficients
            else:
                self.channel_coefficients = None

            return True
        except Exception as error:
            messagebox.showerror("Erreur", str(error))
            return False

    def _is_rgb_lossy_result(self):
        return isinstance(self.compression_result, dict) and self.compression_result.get("kind") == "rgb_lossy"

    def calculate_mse_psnr(self, original_image, reconstructed_image):
        mse_value = mean_squared_error(original_image, reconstructed_image)
        psnr_value = peak_signal_to_noise_ratio(original_image, reconstructed_image)
        return mse_value, psnr_value

    def compress_grayscale_image(self):
        if self.compression_mode == "lossless":
            lossless_output_path = self.results_dir / "lossless_compressed.png"
            result = encode_lossless_wavelet(
                self.original_image,
                self.wavelet_var.get(),
                source_path=self.image_path,
                output_path=lossless_output_path,
            )
            info_lines = [
                "Compression sans perte terminee.",
                "L'image a ete stockee dans un format PNG optimise, sans perte.",
                f"Format source : {result.source_format}",
                f"Format de sortie : {result.output_format}",
                f"Taille avant compression : {self._format_size_bytes_in_mb(result.original_file_bytes)}",
                f"Taille apres compression : {self._format_size_bytes_in_mb(result.compressed_file_bytes)}",
                self._build_size_variation_line(
                    result.original_file_bytes,
                    result.compressed_file_bytes,
                ),
                f"Fichier compresse genere : {result.payload_path}",
                result.user_message,
            ]
            return result, info_lines

        simple_pass = simple_significance_pass(self.coefficients)
        result = encode_ezw(
            self.coefficients,
            self.wavelet_var.get(),
            image_shape=self.working_image.shape[:2],
            max_passes=self.passes_var.get(),
        )
        info_lines = [
            "Compression avec perte EZW terminee.",
            f"Seuil initial T0 : {simple_pass['threshold']}",
            f"POS simple : {simple_pass['counts']['POS']}",
            f"NEG simple : {simple_pass['counts']['NEG']}",
            f"NS simple : {simple_pass['counts']['NS']}",
            f"Nombre de passes EZW : {len(result.passes)}",
            f"Taille compressee estimee : {self._format_size_in_mb(result.compressed_bits)}",
        ]
        return result, info_lines

    def compress_rgb_image(self):
        if self.compression_mode == "lossless":
            lossless_output_path = self.results_dir / "lossless_compressed.png"
            result = encode_lossless_wavelet(
                self.original_image,
                self.wavelet_var.get(),
                source_path=self.image_path,
                output_path=lossless_output_path,
            )
            info_lines = [
                "Compression RGB sans perte terminee.",
                "Les trois canaux RGB ont ete preserves dans un PNG optimise sans perte.",
                f"Format source : {result.source_format}",
                f"Format de sortie : {result.output_format}",
                f"Taille avant compression : {self._format_size_bytes_in_mb(result.original_file_bytes)}",
                f"Taille apres compression : {self._format_size_bytes_in_mb(result.compressed_file_bytes)}",
                self._build_size_variation_line(
                    result.original_file_bytes,
                    result.compressed_file_bytes,
                ),
                f"Fichier compresse genere : {result.payload_path}",
                result.user_message,
            ]
            return result, info_lines

        if not self.channel_coefficients:
            raise ValueError("Les coefficients RGB ne sont pas disponibles pour la compression.")

        channel_results = {}
        channel_simple_passes = {}
        total_compressed_bits = 0
        total_pos = 0
        total_neg = 0
        total_ns = 0

        for channel_name in ("R", "G", "B"):
            channel_coefficients = self.channel_coefficients[channel_name]
            simple_pass = simple_significance_pass(channel_coefficients)
            result = encode_ezw(
                channel_coefficients,
                self.wavelet_var.get(),
                image_shape=self.working_image.shape[:2],
                max_passes=self.passes_var.get(),
            )
            channel_results[channel_name] = result
            channel_simple_passes[channel_name] = simple_pass
            total_compressed_bits += int(result.compressed_bits)
            total_pos += int(simple_pass["counts"]["POS"])
            total_neg += int(simple_pass["counts"]["NEG"])
            total_ns += int(simple_pass["counts"]["NS"])

        bundle = {
            "kind": "rgb_lossy",
            "output_format": "Flux EZW estime RGB",
            "compressed_bits": int(total_compressed_bits),
            "original_bits": int(np.prod(self.working_image.shape) * 8),
            "channel_results": channel_results,
            "channel_simple_passes": channel_simple_passes,
        }

        threshold_summary = ", ".join(
            f"{channel_name}={channel_simple_passes[channel_name]['threshold']}"
            for channel_name in ("R", "G", "B")
        )
        info_lines = [
            "Compression couleur RGB avec perte terminee.",
            "Chaque canal R, G et B a ete compresse separement par la meme logique EZW.",
            f"Seuils initiaux T0 par canal : {threshold_summary}",
            f"POS total : {total_pos}",
            f"NEG total : {total_neg}",
            f"NS total : {total_ns}",
            f"Nombre de passes EZW par canal : {self.passes_var.get()}",
            f"Taille compressee estimee totale : {self._format_size_in_mb(bundle['compressed_bits'])}",
        ]
        return bundle, info_lines

    def reconstruct_grayscale_image(self):
        if self.compression_mode == "lossless":
            return decode_lossless_wavelet(self.compression_result)

        reconstructed_coefficients = decode_ezw(self.compression_result)
        return reconstruct_image(
            reconstructed_coefficients,
            self.wavelet_var.get(),
            output_shape=self.working_image.shape[:2],
        )

    def reconstruct_rgb_image(self):
        if self.compression_mode == "lossless":
            return decode_lossless_wavelet(self.compression_result)

        if not self._is_rgb_lossy_result():
            raise ValueError("Le resultat de compression RGB avec perte est invalide.")

        reconstructed_channels = []
        for channel_name in ("R", "G", "B"):
            channel_result = self.compression_result["channel_results"][channel_name]
            reconstructed_coefficients = decode_ezw(channel_result)
            reconstructed_channel = reconstruct_image(
                reconstructed_coefficients,
                self.wavelet_var.get(),
                output_shape=self.working_image.shape[:2],
            )
            reconstructed_channels.append(reconstructed_channel)

        return self._merge_rgb_channels(reconstructed_channels)

    def _reset_info_history(self):
        self.info_history = []
        self.info_status_var.set("Clique sur le bouton Informations pour afficher les details.")
        self._refresh_information_display()

    def _store_information(self, title, lines, replace=False):
        block = [title] + [f"  {line}" for line in lines]

        if replace:
            self.info_history = [block]
        else:
            self.info_history.append(block)

        self.info_status_var.set("De nouvelles informations sont pretes. Clique sur Informations pour les afficher.")
        self._refresh_information_display()

    def _upsert_information(self, title, lines):
        block = [title] + [f"  {line}" for line in lines]

        for index, existing_block in enumerate(self.info_history):
            if existing_block and existing_block[0] == title:
                self.info_history[index] = block
                self.info_status_var.set(
                    "De nouvelles informations sont pretes. Clique sur Informations pour les afficher."
                )
                self._refresh_information_display()
                return

        self.info_history.append(block)
        self.info_status_var.set("De nouvelles informations sont pretes. Clique sur Informations pour les afficher.")
        self._refresh_information_display()

    def _compute_pixel_count(self, image_shape):
        rows, cols = image_shape[:2]
        return int(rows * cols)

    def _get_original_pixel_count(self):
        if self.original_image is None:
            return None
        return self._compute_pixel_count(self.original_image.shape)

    def _get_compressed_pixel_count(self):
        if self.reconstructed_image is not None:
            return self._compute_pixel_count(self.reconstructed_image.shape)

        if self.compression_mode == "lossless" and self.compression_result is not None:
            payload_path = getattr(self.compression_result, "payload_path", None)
            if payload_path:
                try:
                    with Image.open(payload_path) as compressed_image:
                        width, height = compressed_image.size
                    return int(width * height)
                except Exception:
                    return None

        return None

    def _build_pixel_information_lines(self):
        original_pixels = self._get_original_pixel_count()
        compressed_pixels = self._get_compressed_pixel_count()

        if original_pixels is None:
            return [
                "Nombre de pixels avant compression : non disponible",
                "Nombre de pixels apres compression : non disponible",
                "Taux de reduction : non disponible",
                "Aucune image n'est chargee.",
            ]

        lines = [f"Nombre de pixels avant compression : {original_pixels}"]

        if compressed_pixels is None:
            lines.extend(
                [
                    "Nombre de pixels apres compression : non disponible",
                    "Taux de reduction : non disponible",
                    "Image compressee non encore generee",
                ]
            )
            return lines

        reduction_percent = 100.0 * (1.0 - (compressed_pixels / original_pixels))
        lines.append(f"Nombre de pixels apres compression : {compressed_pixels}")
        lines.append(f"Taux de reduction : {reduction_percent:.2f} %")
        return lines

    def _iter_coefficient_arrays(self):
        if self.coefficients is None:
            return

        yield np.asarray(self.coefficients[0])

        for detail_triplet in self.coefficients[1:]:
            for detail_array in detail_triplet:
                yield np.asarray(detail_array)

    def _count_wavelet_coefficients(self):
        if self.channel_coefficients:
            total_count = 0
            for channel_coefficients in self.channel_coefficients.values():
                total_count += int(np.asarray(channel_coefficients[0]).size)
                for detail_triplet in channel_coefficients[1:]:
                    for detail_array in detail_triplet:
                        total_count += int(np.asarray(detail_array).size)
            return total_count

        if self.coefficients is None:
            return None

        return int(sum(array.size for array in self._iter_coefficient_arrays()))

    def _get_report_symbol_count(self):
        if self.compression_result is None:
            return None

        if self.compression_mode == "lossless":
            return self._count_wavelet_coefficients()

        if self._is_rgb_lossy_result():
            total_symbols = 0
            for channel_result in self.compression_result["channel_results"].values():
                total_symbols += int(sum(len(pass_data.dominant_symbols) for pass_data in channel_result.passes))
            return total_symbols

        return int(sum(len(pass_data.dominant_symbols) for pass_data in self.compression_result.passes))

    def _get_report_initial_threshold(self):
        if self._is_rgb_lossy_result():
            thresholds = [
                float(channel_result.initial_threshold)
                for channel_result in self.compression_result["channel_results"].values()
            ]
            return max(thresholds) if thresholds else None

        if self.compression_result is not None and hasattr(self.compression_result, "initial_threshold"):
            return float(self.compression_result.initial_threshold)

        if self.coefficients is not None:
            return float(simple_significance_pass(self.coefficients)["threshold"])

        return None

    def _get_report_final_threshold(self):
        if self.compression_result is None:
            return None

        if self.compression_mode == "lossless":
            return 0.0

        if self._is_rgb_lossy_result():
            thresholds = []
            for channel_result in self.compression_result["channel_results"].values():
                if channel_result.passes:
                    thresholds.append(float(channel_result.passes[-1].threshold))
            return max(thresholds) if thresholds else None

        if not self.compression_result.passes:
            return None

        return float(self.compression_result.passes[-1].threshold)

    def _get_report_metrics(self):
        if self.latest_comparison_metrics is not None:
            return (
                self.latest_comparison_metrics["mse"],
                self.latest_comparison_metrics["psnr"],
            )

        if self.reconstructed_image is None:
            return None, None

        reference_image = self.original_image if self.compression_mode == "lossless" else self.working_image
        if reference_image is None:
            return None, None

        return (
            mean_squared_error(reference_image, self.reconstructed_image),
            peak_signal_to_noise_ratio(reference_image, self.reconstructed_image),
        )

    def _format_report_number(self, value, decimals=2):
        if value is None:
            return "non disponible"

        return f"{value:.{decimals}f}"

    def _build_streamlit_style_report(self):
        if self.original_image is None or self.working_image is None:
            return None

        rows, cols = self.working_image.shape[:2]
        source_name = self.image_path.name if self.image_path is not None else "image_non_nommee"
        source_format = self.original_file_format or "Niveaux de gris"
        image_type_text = self._get_mode_description()
        mode_internal = self.compression_mode or self._get_selected_mode_internal()
        passes_text = (
            "Illimite (mode sans perte)"
            if mode_internal == "lossless"
            else str(self.passes_var.get())
        )

        raw_original_bytes = int(self.working_image.size)
        compressed_bytes = self._get_display_compressed_file_bytes() if self.compression_result is not None else None
        compression_value = None
        if compressed_bytes:
            compression_value = raw_original_bytes / max(1, compressed_bytes)

        symbol_count = self._get_report_symbol_count()
        initial_threshold = self._get_report_initial_threshold()
        final_threshold = self._get_report_final_threshold()
        mse_value, psnr_value = self._get_report_metrics()

        total_time_ms = 0.0
        has_timing = False
        for timing_value in (self.last_compression_time_ms, self.last_reconstruction_time_ms):
            if timing_value is not None:
                total_time_ms += timing_value
                has_timing = True

        status_text = (
            "Image reconstruite avec succes. Etape finalisee."
            if self.reconstructed_image is not None
            else "Traitement en cours. Lance compression puis reconstruction pour finaliser."
        )

        compressed_text = (
            f"{compressed_bytes} octets"
            if compressed_bytes is not None
            else "non disponible"
        )
        ratio_text = (
            f"{compression_value:.2f} : 1"
            if compression_value is not None
            else "non disponible"
        )
        symbol_text = (
            f"{symbol_count} symboles emis"
            if symbol_count is not None
            else "non disponible"
        )
        time_text = (
            f"{total_time_ms:.2f} ms"
            if has_timing
            else "non disponible"
        )

        return "\n".join(
            [
                "Informations de chargement et compression (EZW System LOG) :",
                "---------------------------------------------------------------------------------",
                f"Nom du fichier charge            : {source_name}",
                f"Format d'image source            : {source_format}",
                f"Type d'image detecte             : {image_type_text}",
                f"Dimensions de calcul             : {cols} x {rows} px",
                f"Fichier source sur disque        : {self._format_size_bytes_in_mb(self.original_file_bytes or 0)}",
                f"Mode de compression              : {self._get_mode_display_label(mode_internal)}",
                f"Ondelette active               : {self.wavelet_var.get().upper()}",
                f"Niveau de decomposition active   : {self.resolved_level or self.level_var.get()}",
                f"Nombre de passes codantes EZW    : {passes_text}",
                "",
                "Resultats d'evaluation :",
                "---------------------------------------------------------------------------------",
                f"Poids d'origine brute            : {raw_original_bytes} octets",
                f"Poids compresse estime           : {compressed_text}",
                f"Taux de compression obtenu       : {ratio_text}",
                f"Nombre total de symboles EZW     : {symbol_text}",
                f"Seuil initial d'encodage (T0)    : {self._format_report_number(initial_threshold, 2)}",
                f"Seuil final d'encodage (T_final) : {self._format_report_number(final_threshold, 4)}",
                "",
                "Qualite de fidelite spatiale :",
                "---------------------------------------------------------------------------------",
                f"Erreur Quadratique Moyenne       : {self._format_report_number(mse_value, 2)} (MSE)",
                f"Facteur de qualite PSNR          : {self._format_report_number(psnr_value, 2)} dB",
                f"Temps total calcul mathematique  : {time_text}",
                "---------------------------------------------------------------------------------",
                f"Status                           : {status_text}",
            ]
        )

    def _render_information_content(self):
        sections = []
        streamlit_style_report = self._build_streamlit_style_report()
        if streamlit_style_report is not None:
            sections.append(streamlit_style_report)

        sections.extend("\n".join(block) for block in self.info_history)

        if self.original_image is None:
            sections.append("État de l'application\n  Veuillez d'abord charger l'image originale.")
        elif self.compression_result is None:
            sections.append("État de l'application\n  Veuillez d'abord compresser l'image.")
        elif self.reconstructed_image is None:
            sections.append("État de l'application\n  Image compressee disponible, reconstruction non encore generee.")

        if not sections:
            return "Aucune information disponible pour le moment."

        return "\n\n".join(sections)

    def _refresh_information_display(self):
        content = self._render_information_content()
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", content)
        self.info_text.configure(state="disabled")

    def _format_size_in_mb(self, bits):
        bytes_value = float(bits) / 8.0
        kilobytes_value = bytes_value / 1024.0
        megabytes_value = bytes_value / (1024.0 ** 2)
        return f"{megabytes_value:.4f} MB ({kilobytes_value:.2f} KB | {bytes_value:.2f} octets)"

    def _format_size_bytes_in_mb(self, file_size_bytes):
        bytes_value = float(file_size_bytes)
        kilobytes_value = bytes_value / 1024.0
        megabytes_value = bytes_value / (1024.0 ** 2)
        return f"{megabytes_value:.4f} MB ({kilobytes_value:.2f} KB | {bytes_value:.2f} octets)"

    def _get_display_original_file_bytes(self):
        if self.compression_mode == "lossless" and self.compression_result is not None:
            return self.compression_result.original_file_bytes

        if self.original_file_bytes is not None:
            return self.original_file_bytes

        if self.image_path is not None and self.image_path.is_file():
            return get_file_size_bytes(self.image_path)

        return 0

    def _get_display_compressed_file_bytes(self):
        if self.compression_result is None:
            return 0

        if self.compression_mode == "lossless":
            return self.compression_result.compressed_file_bytes

        if self._is_rgb_lossy_result():
            return int(self.compression_result["compressed_bits"] / 8.0)

        return int(self.compression_result.compressed_bits / 8.0)

    def _build_size_variation_line(self, original_file_bytes, compressed_file_bytes):
        if original_file_bytes <= 0:
            return "Variation de taille : indisponible."

        variation_percent = 100.0 * ((compressed_file_bytes - original_file_bytes) / original_file_bytes)
        sign = "+" if variation_percent >= 0 else ""
        return f"Variation de taille : {sign}{variation_percent:.2f} %"

    def _build_comparison_information_lines(self, mse_value, psnr_value, compression_value):
        if self.original_image is None:
            return ["Veuillez d'abord charger l'image originale."]

        if self.compression_result is None or self.reconstructed_image is None:
            return ["Veuillez d'abord compresser l'image."]

        original_info = get_image_info(self.original_image)
        working_info = get_image_info(self.working_image)
        reconstructed_info = get_image_info(self.reconstructed_image)
        original_file_bytes = self._get_display_original_file_bytes()
        compressed_file_bytes = self._get_display_compressed_file_bytes()
        compression_shape = original_info["shape"] if self.compression_mode == "lossless" else working_info["shape"]
        source_format = self.original_file_format or "Inconnu"
        output_format = self.compression_result.output_format if self.compression_mode == "lossless" else (
            self.compression_result["output_format"] if self._is_rgb_lossy_result() else "Flux EZW estime"
        )
        reference_image = self.original_image if self.compression_mode == "lossless" else self.working_image
        max_error, mean_error = self._compute_absolute_error_stats(reference_image, self.reconstructed_image)

        lines = [
            "==================== IMAGE AVANT COMPRESSION ====================",
            f"Fichier source : {self.image_path}",
            f"Format source : {source_format}",
            f"Type d'image : {self._get_mode_description()}",
            f"Shape originale : {original_info['shape']}",
            f"Shape utilisee pour la compression : {compression_shape}",
            f"Dtype de travail : {working_info['dtype']}",
            f"Valeurs de travail : min={working_info['min']}, max={working_info['max']}",
            f"Taille avant compression : {self._format_size_bytes_in_mb(original_file_bytes)}",
            "",
            "==================== IMAGE APRES COMPRESSION ====================",
            f"Mode : {self._get_mode_display_label(self.compression_mode)}",
            f"Format de sortie / stockage : {output_format}",
            f"Shape reconstruite : {reconstructed_info['shape']}",
            f"Dtype reconstruit : {reconstructed_info['dtype']}",
            f"Valeurs reconstruites : min={reconstructed_info['min']}, max={reconstructed_info['max']}",
            f"Taille apres compression : {self._format_size_bytes_in_mb(compressed_file_bytes)}",
            self._build_size_variation_line(original_file_bytes, compressed_file_bytes),
            "",
            "==================== MESURES DE COMPARAISON ====================",
            f"MSE : {mse_value:.6f}",
            f"PSNR : {psnr_value:.6f} dB",
            f"Taux de compression : {compression_value:.6f}",
            f"Erreur maximale : {max_error:.4f}",
            f"Erreur moyenne : {mean_error:.4f}",
        ]

        if np.isclose(max_error, 0.0):
            lines.append("Erreur absolue nulle : reconstruction identique a l'image de travail.")

        if self.compression_mode == "lossless" and getattr(self.compression_result, "user_message", ""):
            lines.extend(
                [
                    "",
                    "==================== MESSAGE UTILISATEUR ====================",
                    self.compression_result.user_message,
                ]
            )

        return lines

    def _update_information_button_text(self):
        if self.information_button is None:
            return

        text = "ⓘ  Masquer informations" if self.information_visible else "ⓘ  Informations"
        self.information_button.configure(text=text)

    def show_information(self):
        if self.information_visible:
            self.metrics_frame.pack_forget()
            self.information_visible = False
            self.info_status_var.set("Informations masquees.")
            self._update_information_button_text()
            return

        self._refresh_information_display()
        self.metrics_frame.pack(fill="x", padx=10, pady=(0, 6), before=self.status_bar)
        self.information_visible = True
        self.info_status_var.set("Informations affichees.")
        self._update_information_button_text()

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Tous les fichiers", "*.*")],
        )

        if not file_path:
            return

        try:
            original_image, detected_mode, normalized_mode = load_image_with_mode(file_path)
            working_image = prepare_image_for_processing(original_image, max_size=self.max_size_var.get())
            original_file_bytes = get_file_size_bytes(file_path)
            original_file_format = get_image_format(file_path)
        except Exception as error:
            messagebox.showerror("Erreur", str(error))
            return

        loaded_image_kind = "Image couleur RGB" if normalized_mode == "RGB" else "Image en niveaux de gris"
        alpha_handling_note = None
        if detected_mode == "RGBA" and normalized_mode == "RGB":
            alpha_handling_note = "Le canal alpha a ete ignore pour traiter l'image comme une image RGB."

        self.image_path = Path(file_path)
        self.original_image = original_image
        self.working_image = working_image
        self.analysis_image = self._convert_to_grayscale_array(working_image)
        self.original_file_bytes = original_file_bytes
        self.original_file_format = original_file_format
        self.loaded_image_mode = normalized_mode
        self.loaded_image_kind = loaded_image_kind
        self.alpha_handling_note = alpha_handling_note
        self.coefficients = None
        self.channel_coefficients = None
        self.resolved_level = None
        self.compression_result = None
        self.reconstructed_image = None
        self.compression_mode = None
        self.latest_comparison_metrics = None
        self.error_image = None
        self.error_photo = None
        self.bitstream_data = None
        self.last_compression_time_ms = None
        self.last_reconstruction_time_ms = None
        self._reset_info_history()

        original_info = get_image_info(original_image)
        working_info = get_image_info(working_image)
        self._store_information(
            "Informations de chargement",
            [
                f"Image chargee : {self.image_path}",
                f"Format source : {self.original_file_format}",
                f"Type detecte : {loaded_image_kind}",
                f"Mode source PIL : {detected_mode}",
                f"Shape originale : {original_info['shape']}",
                f"Shape de travail : {working_info['shape']}",
                f"Dtype de travail : {working_info['dtype']}",
                f"Valeurs : min={working_info['min']}, max={working_info['max']}",
                f"Taille du fichier source : {self._format_size_bytes_in_mb(self.original_file_bytes)}",
            ],
            replace=True,
        )
        if alpha_handling_note:
            self._upsert_information("Gestion du canal alpha", [alpha_handling_note])
        self._upsert_information("Informations sur les pixels", self._build_pixel_information_lines())
        self._show_loaded_image_breakdown_view()

    def decompose_current_image(self):
        if self.working_image is None:
            messagebox.showwarning("Information", "Charge d'abord une image.")
            return

        if not self._compute_decomposition():
            return
        resolved_level = self.resolved_level
        self._show_decomposition_view()

        self._store_information(
            "Informations de decomposition",
            [
                "Decomposition en ondelettes realisee.",
                f"Ondelette : {self.wavelet_var.get()}",
                f"Niveau utilise : {resolved_level}",
                "cA : approximation globale",
                "cH : details horizontaux",
                "cV : details verticaux",
                "cD : details diagonaux",
            ],
        )

    def compress_current_image(self):
        if self.working_image is None:
            messagebox.showwarning("Information", "Charge d'abord une image.")
            return

        if self.coefficients is None and not self._compute_decomposition():
            return

        start_time = time.perf_counter()

        try:
            self.compression_mode = self._get_selected_mode_internal()
            if self._is_color_image(self.original_image):
                self.compression_result, info_lines = self.compress_rgb_image()
            else:
                self.compression_result, info_lines = self.compress_grayscale_image()
        except Exception as error:
            messagebox.showerror("Erreur", str(error))
            return

        self.last_compression_time_ms = (time.perf_counter() - start_time) * 1000.0
        self._upsert_information("Informations de compression", info_lines)
        self._upsert_information("Informations sur les pixels", self._build_pixel_information_lines())
        self._show_decomposition_view()

    def reconstruct_current_image(self):
        if self.compression_result is None:
            messagebox.showwarning("Information", "Lance d'abord la compression EZW.")
            return

        start_time = time.perf_counter()

        try:
            if self._is_color_image(self.original_image):
                self.reconstructed_image = self.reconstruct_rgb_image()
            else:
                self.reconstructed_image = self.reconstruct_grayscale_image()
        except Exception as error:
            messagebox.showerror("Erreur", str(error))
            return

        self.last_reconstruction_time_ms = (time.perf_counter() - start_time) * 1000.0
        self.reconstructed_photo = self._array_to_photo(self.reconstructed_image)
        self._show_reconstruction_view()

        self._upsert_information(
            "Informations de reconstruction",
            [
                "Reconstruction terminee.",
                f"Mode : {self._get_mode_display_label(self.compression_mode)}",
                f"Shape reconstruite : {self.reconstructed_image.shape}",
                "Utilise l'action Comparer pour calculer les mesures finales.",
            ],
        )
        self._upsert_information("Informations sur les pixels", self._build_pixel_information_lines())

    def compare_images(self):
        if self.original_image is None:
            messagebox.showwarning("Information", "Veuillez d'abord charger l'image originale.")
            return

        if self.compression_result is None or self.reconstructed_image is None:
            messagebox.showwarning("Information", "Veuillez d'abord compresser l'image.")
            return

        reference_image = self.original_image if self.compression_mode == "lossless" else self.working_image
        mse_value, psnr_value = self.calculate_mse_psnr(reference_image, self.reconstructed_image)
        original_file_bytes = self._get_display_original_file_bytes()
        compressed_file_bytes = self._get_display_compressed_file_bytes()
        compression_value = compression_ratio(
            int(original_file_bytes * 8),
            int(max(1, compressed_file_bytes) * 8),
        )

        self.latest_comparison_metrics = {
            "mse": mse_value,
            "psnr": psnr_value,
            "compression_ratio": compression_value,
        }

        self._upsert_information(
            "Informations de comparaison",
            self._build_comparison_information_lines(mse_value, psnr_value, compression_value),
        )
        self._show_comparison_view()

    def save_result(self):
        if self.reconstructed_image is None:
            messagebox.showwarning("Information", "Aucune image reconstruite a sauvegarder.")
            return

        default_path = self.results_dir / "gui_reconstructed.png"
        file_path = filedialog.asksaveasfilename(
            title="Sauvegarder l'image reconstruite",
            initialfile=default_path.name,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
        )

        if not file_path:
            return

        try:
            saved_path = save_image(file_path, self.reconstructed_image)
        except Exception as error:
            messagebox.showerror("Erreur", str(error))
            return

        messagebox.showinfo("Succes", f"Image sauvegardee : {saved_path}")

def launch_gui():
    application = EZWApp()
    application.mainloop()


if __name__ == "__main__":
    launch_gui()
