"""
Core Image Processing
Tüm analiz ve enhancement fonksiyonları burada.
Duplicate yok - tek kaynak.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .config import config


def analyze(img: np.ndarray) -> Dict[str, Any]:
    """
    Görüntü analizi
    
    Args:
        img: BGR formatında numpy array (HxWx3)
    
    Returns:
        Analiz sonuçları dictionary'si
    """
    # Alpha varsa sadece BGR al
    if len(img.shape) > 2 and img.shape[2] == 4:
        img = img[:, :, :3]
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    brightness = np.mean(hsv[:, :, 2])
    contrast = np.std(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    saturation = np.mean(hsv[:, :, 1])
    
    # Haziness (dark channel prior)
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(min_channel, kernel)
    haziness = np.mean(dark_channel)
    
    # Color cast
    b, g, r = cv2.split(img)
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    avg_all = (avg_b + avg_g + avg_r) / 3
    color_cast = {
        'b_ratio': avg_b / avg_all if avg_all > 0 else 1,
        'g_ratio': avg_g / avg_all if avg_all > 0 else 1,
        'r_ratio': avg_r / avg_all if avg_all > 0 else 1
    }
    
    return {
        'brightness': brightness, 
        'contrast': contrast, 
        'sharpness': sharpness,
        'saturation': saturation, 
        'haziness': haziness, 
        'color_cast': color_cast
    }


def calculate_auto_params(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analiz sonuçlarına göre otomatik parametreleri hesapla
    
    Args:
        analysis: analyze() fonksiyonundan gelen sonuçlar
    
    Returns:
        Enhancement parametreleri
    """
    th = config.thresholds
    mp = config.multipliers
    
    params = {
        'dehaze': {'apply': False, 'strength': 0.5},
        'white_balance': {'apply': False},
        'clahe': {'apply': False, 'clip_limit': 2.0},
        'gamma': {'apply': False, 'value': 1.0},
        'sharpen': {'apply': True, 'strength': 1.0},
        'color': {'apply': False, 'saturation': 1.0},
        'detail': {'apply': False},
        'auto_levels': {'apply': False},
        'denoise': {'apply': True}
    }

    # Dehaze
    if analysis['haziness'] > th['haziness']:
        params['dehaze']['apply'] = True
        params['dehaze']['strength'] = min(
            mp['dehaze_strength_max'], 
            max(mp['dehaze_strength_min'], analysis['haziness'] / mp['dehaze_divisor'])
        )
    
    # White Balance
    cc = analysis['color_cast']
    color_deviation = max(abs(cc['r_ratio'] - 1), abs(cc['g_ratio'] - 1), abs(cc['b_ratio'] - 1))
    if color_deviation > th['color_deviation']:
        params['white_balance']['apply'] = True

    # CLAHE
    if analysis['contrast'] < th['contrast_min'] or analysis['brightness'] < th['brightness_low']:
        params['clahe']['apply'] = True
        params['clahe']['clip_limit'] = max(
            mp['clahe_clip_min'], 
            min(mp['clahe_clip_max'], mp['clahe_clip_max'] - (analysis['contrast'] / 25))
        )
    
    # Gamma
    brightness_threshold = th['brightness_low'] + 20
    if analysis['brightness'] < brightness_threshold:
        params['gamma']['apply'] = True
        params['gamma']['value'] = max(mp['gamma_min'], min(1.0, analysis['brightness'] / brightness_threshold))
    elif analysis['brightness'] > th['brightness_high']:
        params['gamma']['apply'] = True
        params['gamma']['value'] = min(mp['gamma_max'], 1.0 + (analysis['brightness'] - th['brightness_high']) / 150)

    # Sharpen
    if analysis['sharpness'] < th['sharpness']:
        params['sharpen']['strength'] = max(
            mp['sharpen_strength_min'], 
            min(mp['sharpen_strength_max'], mp['sharpen_strength_max'] - (analysis['sharpness'] / 750))
        )

    # Color/Saturation
    if analysis['saturation'] < th['saturation_min']:
        params['color']['apply'] = True
        current_sat = max(analysis['saturation'], 20)
        params['color']['saturation'] = min(mp['saturation_cap'], max(mp['saturation_min_mult'], 80 / current_sat * 0.70))
    
    # Detail
    if th['detail_sharpness_min'] < analysis['sharpness'] < th['detail_sharpness_max']:
        params['detail']['apply'] = True
    
    # Auto Levels
    if analysis['contrast'] < th['auto_levels_contrast']:
        params['auto_levels']['apply'] = True
    
    return params


def enhance(img: np.ndarray, params: Dict[str, Any], analysis: Dict[str, Any]) -> np.ndarray:
    """
    Görüntüyü enhance et (LAB tabanlı + Adaptif Gamma)
    
    Args:
        img: BGR veya BGRA formatında numpy array
        params: Enhancement parametreleri
        analysis: Görüntü analiz sonuçları
    
    Returns:
        Enhanced görüntü
    """
    ag = config.adaptive_gamma
    
    # Alpha kanalını ayır (varsa)
    has_alpha = len(img.shape) > 2 and img.shape[2] == 4
    if has_alpha:
        bgr = img[:, :, :3].copy()
        alpha = img[:, :, 3]
    else:
        bgr = img.copy()
        alpha = None
    
    # Dehaze (RGB'de)
    if params['dehaze']['apply']:
        bgr = _apply_dehaze(bgr, params['dehaze']['strength'])
    
    # === LAB RENK UZAYINA GEÇİŞ ===
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # White Balance (LAB'da)
    if params['white_balance']['apply']:
        a_mean = np.mean(a_channel)
        b_mean = np.mean(b_channel)
        a_channel = np.clip(a_channel.astype(np.float32) - (a_mean - 128) * 0.3, 0, 255).astype(np.uint8)
        b_channel = np.clip(b_channel.astype(np.float32) - (b_mean - 128) * 0.3, 0, 255).astype(np.uint8)
    
    # Auto Levels (L kanalında)
    if params['auto_levels']['apply']:
        min_val = np.percentile(l_channel, 1)
        max_val = np.percentile(l_channel, 99)
        if max_val > min_val:
            l_channel = np.clip((l_channel.astype(np.float32) - min_val) * 255.0 / (max_val - min_val), 0, 255).astype(np.uint8)
    
    # CLAHE (L kanalında)
    if params['clahe']['apply']:
        clahe = cv2.createCLAHE(clipLimit=params['clahe']['clip_limit'], tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
    
    # Adaptif Gamma (L kanalında)
    if params['gamma']['apply']:
        l_channel = _apply_adaptive_gamma(l_channel, params['gamma']['value'], ag)
    
    # Detail Enhancement (L kanalında)
    if params['detail']['apply']:
        l_smooth = cv2.bilateralFilter(l_channel, 9, 75, 75)
        l_detail = cv2.subtract(l_channel, l_smooth)
        l_channel = np.clip(cv2.add(l_channel, l_detail), 0, 255).astype(np.uint8)
    
    # Sharpen (L kanalında)
    if params['sharpen']['apply']:
        l_blurred = cv2.GaussianBlur(l_channel, (0, 0), 1.0)
        l_channel = np.clip(
            cv2.addWeighted(l_channel, 1.0 + params['sharpen']['strength'], l_blurred, -params['sharpen']['strength'], 0),
            0, 255
        ).astype(np.uint8)
    
    # === LAB'DAN BGR'YE GERİ DÖN ===
    lab = cv2.merge([l_channel, a_channel, b_channel])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Color/Saturation (BGR'de)
    if params['color']['apply']:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params['color']['saturation'], 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Denoise (en sonda)
    if params['denoise']['apply']:
        result = cv2.medianBlur(result, 3)
        result = cv2.bilateralFilter(result, 7, 50, 50)
        result = cv2.fastNlMeansDenoisingColored(result, None, 5, 5, 7, 21)
    
    # Alpha'yı geri ekle
    if has_alpha and alpha is not None:
        result = cv2.merge([result[:, :, 0], result[:, :, 1], result[:, :, 2], alpha])
    
    return result


def _apply_dehaze(img: np.ndarray, strength: float) -> np.ndarray:
    """Dark Channel Prior ile dehaze"""
    img_float = img.astype(np.float64) / 255.0
    min_channel = np.min(img_float, axis=2)
    
    kernel_size = max(3, min(img.shape[0], img.shape[1]) // 50)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    flat_dark = cv2.erode(min_channel, kernel).flatten()
    flat_img = img_float.reshape(-1, 3)
    num_brightest = max(1, int(len(flat_dark) * 0.001))
    brightest_indices = np.argsort(flat_dark)[-num_brightest:]
    atmospheric_light = np.max(flat_img[brightest_indices], axis=0)
    
    normalized = img_float / (atmospheric_light + 1e-6)
    min_normalized = np.min(normalized, axis=2)
    transmission = 1 - strength * cv2.erode(min_normalized, kernel)
    transmission = np.clip(transmission, 0.1, 1.0)
    
    result = np.zeros_like(img_float)
    for i in range(3):
        result[:, :, i] = (img_float[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
    
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def _apply_adaptive_gamma(l_channel: np.ndarray, gamma_value: float, ag: Dict[str, float]) -> np.ndarray:
    """Adaptif lokal gamma - gölgelere farklı, aydınlık yerlere farklı"""
    l_float = l_channel.astype(np.float32) / 255.0
    
    dark_th = ag['dark_threshold']
    bright_th = ag['bright_threshold']
    trans_w = ag['transition_width']
    
    dark_mask = np.clip((dark_th - l_float) / trans_w, 0, 1)
    bright_mask = np.clip((l_float - bright_th) / trans_w, 0, 1)
    mid_mask = np.clip(1 - dark_mask - bright_mask, 0, 1)
    
    if gamma_value < 1.0:
        dark_gamma = min(gamma_value * 1.3, 1.0)
        bright_gamma = gamma_value
    else:
        dark_gamma = gamma_value
        bright_gamma = max(gamma_value * 0.5 + 0.5, 1.0)
    
    mid_gamma = gamma_value
    
    l_dark = np.power(l_float + 1e-6, 1.0 / dark_gamma)
    l_bright = np.power(l_float + 1e-6, 1.0 / bright_gamma)
    l_mid = np.power(l_float + 1e-6, 1.0 / mid_gamma)
    
    l_float = l_dark * dark_mask + l_mid * mid_mask + l_bright * bright_mask
    return np.clip(l_float * 255, 0, 255).astype(np.uint8)


# ============== HISTOGRAM MATCHING ==============

def compute_histogram(img: np.ndarray) -> np.ndarray:
    """LAB L kanalının histogramını hesapla"""
    if len(img.shape) > 2 and img.shape[2] == 4:
        bgr = img[:, :, :3]
    else:
        bgr = img
    
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    hist, _ = np.histogram(l_channel.flatten(), bins=256, range=(0, 256))
    return hist.astype(np.float64)


def apply_histogram_matching(img: np.ndarray, ref_histogram: np.ndarray) -> np.ndarray:
    """Görüntüyü referans histograma eşitle (sadece L kanalı)"""
    has_alpha = len(img.shape) > 2 and img.shape[2] == 4
    if has_alpha:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img
        alpha = None
    
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # CDF'ler
    src_hist, _ = np.histogram(l_channel.flatten(), bins=256, range=(0, 256))
    src_cdf = np.cumsum(src_hist).astype(np.float64)
    src_cdf = src_cdf / src_cdf[-1]
    
    ref_cdf = np.cumsum(ref_histogram).astype(np.float64)
    ref_cdf = ref_cdf / ref_cdf[-1]
    
    # LUT
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        diff = np.abs(ref_cdf - src_cdf[i])
        lut[i] = np.argmin(diff)
    
    l_matched = cv2.LUT(l_channel, lut)
    
    lab_matched = cv2.merge([l_matched, a_channel, b_channel])
    result = cv2.cvtColor(lab_matched, cv2.COLOR_LAB2BGR)
    
    if has_alpha and alpha is not None:
        result = cv2.merge([result[:, :, 0], result[:, :, 1], result[:, :, 2], alpha])
    
    return result


# ============== I/O HELPERS ==============

def read_image(path: Path) -> Optional[np.ndarray]:
    """Unicode destekli görüntü okuma (alpha dahil)"""
    try:
        img_array = np.fromfile(str(path), np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def write_image(path: Path, img: np.ndarray) -> bool:
    """Unicode destekli görüntü yazma"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        is_success, buffer = cv2.imencode(".png", img)
        if is_success:
            buffer.tofile(str(path))
            return True
    except Exception:
        pass
    return False
