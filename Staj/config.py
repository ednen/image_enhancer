import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    'thresholds': {
        'haziness': 40,
        'contrast_min': 45,
        'brightness_low': 100,
        'brightness_high': 160,
        'sharpness': 1500,
        'saturation_min': 100,
        'color_deviation': 0.08,
        'auto_levels_contrast': 35,
        'detail_sharpness_min': 300,
        'detail_sharpness_max': 1500,
    },
    'multipliers': {
        'dehaze_strength_max': 0.75,
        'dehaze_strength_min': 0.20,
        'dehaze_divisor': 200,
        'clahe_clip_max': 2.5,
        'clahe_clip_min': 1.2,
        'gamma_min': 0.75,
        'gamma_max': 1.4,
        'sharpen_strength_max': 2.5,
        'sharpen_strength_min': 0.5,
        'saturation_cap': 1.08,
        'saturation_min_mult': 1.02,
    },
    'processing': {
        'chunk_size_small': 5,
        'chunk_size_medium': 10,
        'chunk_size_large': 20,
        'files_per_subdir': 10000,
        'skip_existing': True,
        'max_workers': None,  
        'write_buffer_limit': 50,  
        'histogram_samples': 50,
    },
    'adaptive_gamma': {
        'dark_threshold': 0.4,
        'bright_threshold': 0.6,
        'transition_width': 0.2,
    }
}
class Config:
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = DEFAULT_CONFIG.copy()
        return cls._instance
    
    def load(self, config_path: Optional[str] = "config.json") -> 'Config':
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    if user_config:
                        self._deep_merge(user_config)
                logger.info(f"Config yüklendi: {config_path}")
            except Exception as e:
                logger.warning(f"Config yüklenemedi, default kullanılıyor: {e}")
        return self
    
    def _deep_merge(self, user_config: Dict[str, Any]) -> None:
        for key, value in user_config.items():
            if key in self._config and isinstance(value, dict):
                self._config[key].update(value)
            else:
                self._config[key] = value
    
    def save(self, config_path: str = "config.json") -> None:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
        print(f"Config kaydedildi: {config_path}")
    
    @property
    def thresholds(self) -> Dict[str, Any]:
        return self._config['thresholds']
    
    @property
    def multipliers(self) -> Dict[str, Any]:
        return self._config['multipliers']
    
    @property
    def processing(self) -> Dict[str, Any]:
        return self._config['processing']
    
    @property
    def adaptive_gamma(self) -> Dict[str, Any]:
        return self._config['adaptive_gamma']
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        return self._config.copy()

config = Config()
