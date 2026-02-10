
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from pathlib import Path
from threading import Thread
from typing import Optional
from .config import config
from .pipeline import Pipeline, ProgressInfo
from . import core

class EnhancerGUI:
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image Enhancer")
        self.root.geometry("1200x800")

        self.pipeline: Optional[Pipeline] = None
        self.current_image: Optional[np.ndarray] = None
        self.preview_image: Optional[ImageTk.PhotoImage] = None
        self.is_processing = False

        self.input_dir = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value="")
        self.mode = tk.StringVar(value="auto")
        self.parallel_mode = tk.BooleanVar(value=True)
        self.skip_existing = tk.BooleanVar(value=True)
        self.histogram_matching = tk.BooleanVar(value=True)
        self.progress_var = tk.DoubleVar(value=0)

        self._init_enhancement_vars()

        self._create_ui()
    
    def _init_enhancement_vars(self) -> None:
        self.var_dehaze = tk.BooleanVar(value=False)
        self.var_white_balance = tk.BooleanVar(value=False)
        self.var_clahe = tk.BooleanVar(value=True)
        self.var_gamma = tk.BooleanVar(value=True)
        self.var_sharpen = tk.BooleanVar(value=True)
        self.var_color = tk.BooleanVar(value=False)
        self.var_detail = tk.BooleanVar(value=False)
        self.var_auto_levels = tk.BooleanVar(value=False)
        self.var_denoise = tk.BooleanVar(value=True)
        
        self.slider_dehaze = tk.DoubleVar(value=0.5)
        self.slider_clahe = tk.DoubleVar(value=2.0)
        self.slider_gamma = tk.DoubleVar(value=1.0)
        self.slider_sharpen = tk.DoubleVar(value=1.0)
        self.slider_saturation = tk.DoubleVar(value=1.0)
    
    def _create_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)

        left_panel = ttk.LabelFrame(main_frame, text="Kontroller", padding="10")
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        
        self._create_directory_section(left_panel)
        self._create_mode_section(left_panel)
        self._create_options_section(left_panel)
        self._create_buttons_section(left_panel)
        self._create_progress_section(left_panel)

        right_panel = ttk.LabelFrame(main_frame, text="Önizleme", padding="10")
        right_panel.pack(side="right", fill="both", expand=True)
        
        self._create_preview_section(right_panel)
    
    def _create_directory_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Giriş Klasörü:").grid(row=0, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.input_dir, width=30).grid(row=0, column=1, padx=5)
        ttk.Button(parent, text="...", width=3, command=self._select_input_dir).grid(row=0, column=2)
        
        ttk.Label(parent, text="Çıkış Klasörü:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(parent, textvariable=self.output_dir, width=30).grid(row=1, column=1, padx=5, pady=(5, 0))
        ttk.Button(parent, text="...", width=3, command=self._select_output_dir).grid(row=1, column=2, pady=(5, 0))
    
    def _create_mode_section(self, parent: ttk.Frame) -> None:
        mode_frame = ttk.LabelFrame(parent, text="Mod", padding="5")
        mode_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=10)
        
        ttk.Radiobutton(mode_frame, text="Otomatik", variable=self.mode, value="auto",
                       command=self._on_mode_change).pack(side="left", padx=10)
        ttk.Radiobutton(mode_frame, text="Manuel", variable=self.mode, value="manual",
                       command=self._on_mode_change).pack(side="left", padx=10)
    
    def _create_options_section(self, parent: ttk.Frame) -> None:
        options_frame = ttk.LabelFrame(parent, text="Seçenekler", padding="5")
        options_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)
        
        ttk.Checkbutton(options_frame, text="Paralel İşleme (Hızlı)", 
                       variable=self.parallel_mode).pack(anchor="w")
        ttk.Checkbutton(options_frame, text="Var Olanları Atla (Resume)", 
                       variable=self.skip_existing).pack(anchor="w")
        ttk.Checkbutton(options_frame, text="Histogram Eşitleme (Tutarlı Renkler)", 
                       variable=self.histogram_matching).pack(anchor="w")

        self.manual_frame = ttk.LabelFrame(parent, text="Manuel Ayarlar", padding="5")
        self.manual_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5)
        
        self._create_manual_controls(self.manual_frame)
        self._update_manual_visibility()
    
    def _create_manual_controls(self, parent: ttk.Frame) -> None:
        controls = [
            ("Dehaze", self.var_dehaze, self.slider_dehaze, 0.0, 1.0),
            ("CLAHE", self.var_clahe, self.slider_clahe, 1.0, 4.0),
            ("Gamma", self.var_gamma, self.slider_gamma, 0.5, 2.0),
            ("Sharpen", self.var_sharpen, self.slider_sharpen, 0.0, 3.0),
            ("Saturation", self.var_color, self.slider_saturation, 0.5, 1.5),
        ]
        
        for i, (name, var, slider, min_val, max_val) in enumerate(controls):
            ttk.Checkbutton(parent, text=name, variable=var,
                           command=self._on_param_change).grid(row=i, column=0, sticky="w")
            ttk.Scale(parent, from_=min_val, to=max_val, variable=slider, orient="horizontal",
                     command=lambda _: self._on_param_change()).grid(row=i, column=1, sticky="ew", padx=5)
            ttk.Label(parent, textvariable=slider, width=5).grid(row=i, column=2)

        other_frame = ttk.Frame(parent)
        other_frame.grid(row=len(controls), column=0, columnspan=3, sticky="w", pady=5)
        
        ttk.Checkbutton(other_frame, text="White Balance", variable=self.var_white_balance,
                       command=self._on_param_change).pack(side="left")
        ttk.Checkbutton(other_frame, text="Detail", variable=self.var_detail,
                       command=self._on_param_change).pack(side="left", padx=10)
        ttk.Checkbutton(other_frame, text="Denoise", variable=self.var_denoise,
                       command=self._on_param_change).pack(side="left")
    
    def _create_buttons_section(self, parent: ttk.Frame) -> None:
        ttk.Button(parent, text="Önizleme Yükle", 
                  command=self._load_preview).grid(row=5, column=0, columnspan=3, sticky="ew", pady=5)
        
        self.process_btn = ttk.Button(parent, text="Tümünü İşle", command=self._process_all)
        self.process_btn.grid(row=6, column=0, columnspan=3, sticky="ew", pady=5)
        
        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=7, column=0, columnspan=3, sticky="ew")
        
        self.cancel_btn = ttk.Button(btn_frame, text="Durdur", command=self._cancel, state="disabled")
        self.cancel_btn.pack(side="left", expand=True, fill="x", padx=(0, 2))
        
        self.resume_btn = ttk.Button(btn_frame, text="Devam Et", command=self._resume, state="disabled")
        self.resume_btn.pack(side="left", expand=True, fill="x", padx=(2, 0))
    
    def _create_progress_section(self, parent: ttk.Frame) -> None:
        progress_frame = ttk.Frame(parent)
        progress_frame.grid(row=8, column=0, columnspan=3, sticky="ew", pady=10)
        
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x")
        
        self.progress_label = ttk.Label(progress_frame, text="Hazır")
        self.progress_label.pack(pady=5)
        
        self.percent_label = ttk.Label(progress_frame, text="0%")
        self.percent_label.pack()
    
    def _create_preview_section(self, parent: ttk.Frame) -> None:
        preview_frame = ttk.Frame(parent)
        preview_frame.pack(fill="both", expand=True)

        orig_frame = ttk.LabelFrame(preview_frame, text="Orijinal")
        orig_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self.original_label = ttk.Label(orig_frame)
        self.original_label.pack(fill="both", expand=True)

        enh_frame = ttk.LabelFrame(preview_frame, text="İyileştirilmiş")
        enh_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        self.enhanced_label = ttk.Label(enh_frame)
        self.enhanced_label.pack(fill="both", expand=True)

    
    def _select_input_dir(self) -> None:
        path = filedialog.askdirectory(title="Giriş Klasörü Seç")
        if path:
            self.input_dir.set(path)
            self._load_first_preview(path)
    
    def _select_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Çıkış Klasörü Seç")
        if path:
            self.output_dir.set(path)
    
    def _load_first_preview(self, directory: str) -> None:
        first_png = self._find_first_png(Path(directory))
        if first_png:
            self._load_image(first_png)
    
    def _find_first_png(self, directory: Path) -> Optional[Path]:
        try:
            for entry in sorted(directory.iterdir()):
                if entry.is_file() and entry.name.lower().endswith('.png'):
                    return entry
                elif entry.is_dir():
                    result = self._find_first_png(entry)
                    if result:
                        return result
        except:
            pass
        return None
    
    def _load_preview(self) -> None:
        path = filedialog.askopenfilename(
            title="Görüntü Seç",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if path:
            self._load_image(Path(path))
    
    def _load_image(self, path: Path) -> None:
        img = core.read_image(path)
        if img is None:
            messagebox.showerror("Hata", "Görüntü yüklenemedi")
            return
        
        self.current_image = img
        self._update_preview()
    
    def _update_preview(self) -> None:
        if self.current_image is None:
            return

        orig_display = self._prepare_display(self.current_image)
        self.original_label.configure(image=orig_display)
        self.original_label.image = orig_display

        params = self._get_manual_params()
        analysis = core.analyze(self.current_image)
        
        if self.mode.get() == "auto":
            params = core.calculate_auto_params(analysis)
        
        enhanced = core.enhance(self.current_image, params, analysis)
        enh_display = self._prepare_display(enhanced)
        self.enhanced_label.configure(image=enh_display)
        self.enhanced_label.image = enh_display
    
    def _prepare_display(self, img: np.ndarray, max_size: int = 400) -> ImageTk.PhotoImage:
        if len(img.shape) > 2 and img.shape[2] == 4:
            display = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = display.shape[:2]
        scale = min(max_size / w, max_size / h)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            display = cv2.resize(display, (new_w, new_h))
        
        pil_img = Image.fromarray(display)
        return ImageTk.PhotoImage(pil_img)
    
    def _on_mode_change(self) -> None:
        self._update_manual_visibility()
        self._update_preview()
    
    def _update_manual_visibility(self) -> None:
        if self.mode.get() == "manual":
            self.manual_frame.grid()
        else:
            self.manual_frame.grid_remove()
    
    def _on_param_change(self) -> None:
        self._update_preview()
    
    def _get_manual_params(self) -> dict:
        return {
            'dehaze': {'apply': self.var_dehaze.get(), 'strength': self.slider_dehaze.get()},
            'white_balance': {'apply': self.var_white_balance.get()},
            'clahe': {'apply': self.var_clahe.get(), 'clip_limit': self.slider_clahe.get()},
            'gamma': {'apply': self.var_gamma.get(), 'value': self.slider_gamma.get()},
            'sharpen': {'apply': self.var_sharpen.get(), 'strength': self.slider_sharpen.get()},
            'color': {'apply': self.var_color.get(), 'saturation': self.slider_saturation.get()},
            'detail': {'apply': self.var_detail.get()},
            'auto_levels': {'apply': self.var_auto_levels.get()},
            'denoise': {'apply': self.var_denoise.get()}
        }

    def _process_all(self) -> None:
        input_dir = self.input_dir.get()
        output_dir = self.output_dir.get()
        
        if not input_dir or not output_dir:
            messagebox.showwarning("Uyarı", "Lütfen giriş ve çıkış klasörlerini seçin")
            return
        
        self.is_processing = True
        self.process_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")

        self.pipeline = Pipeline()
        self.pipeline.on_progress = self._on_progress
        self.pipeline.on_complete = self._on_complete
        
        def run():
            count = self.pipeline.scan_files(
                input_dir, 
                self.skip_existing.get(), 
                output_dir
            )
            
            if count == 0:
                self.root.after(0, lambda: messagebox.showinfo("Bilgi", "İşlenecek dosya yok"))
                self.root.after(0, self._reset_ui)
                return

            self.pipeline.prepare_histogram(input_dir, self.histogram_matching.get())

            mode = self.mode.get()
            params = self._get_manual_params() if mode == "manual" else None
            
            if self.parallel_mode.get():
                self.pipeline.run_parallel(input_dir, output_dir, mode, params)
            else:
                self.pipeline.run_sequential(input_dir, output_dir, mode, params)
        
        Thread(target=run, daemon=True).start()
    
    def _cancel(self) -> None:
        if self.pipeline:
            self.pipeline.cancel()
        self.cancel_btn.configure(state="disabled")
        self.resume_btn.configure(state="normal")
    
    def _resume(self) -> None:
        self._process_all()
    
    def _on_progress(self, info: ProgressInfo) -> None:
        def update():
            percent = (info.current / info.total) * 100 if info.total > 0 else 0
            self.progress_var.set(percent)
            self.percent_label.configure(text=f"{percent:.1f}%")
            
            eta_str = f"{info.eta:.0f}sn" if info.eta < 60 else f"{info.eta/60:.1f}dk"
            self.progress_label.configure(
                text=f"{info.current}/{info.total} - {info.filename} | Hız: {info.speed:.1f}/sn | ETA: {eta_str}"
            )
        
        self.root.after(0, update)
    
    def _on_complete(self, processed: int, errors: int, error_list: list) -> None:
        def update():
            self._reset_ui()
            
            if errors > 0:
                msg = f"{processed} dosya işlendi, {errors} hata oluştu."
                messagebox.showwarning("Tamamlandı", msg)
            else:
                messagebox.showinfo("Tamamlandı", f"{processed} dosya başarıyla işlendi")
        
        self.root.after(0, update)
    
    def _reset_ui(self) -> None:
        self.is_processing = False
        self.process_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.resume_btn.configure(state="disabled")
        self.progress_var.set(0)
        self.progress_label.configure(text="Hazır")
        self.percent_label.configure(text="0%")


def run_gui() -> None:
    root = tk.Tk()
    app = EnhancerGUI(root)
    root.mainloop()
