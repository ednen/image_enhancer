
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from . import core

logger = logging.getLogger(__name__)
class WriteTask:
    path: Path
    buffer: bytes
    filename: str
    rel_path: str
@dataclass 
class ProcessResult:
    success: bool
    filename: str
    rel_path: str
    error: Optional[str] = None
    
    @property
    def message(self) -> str:
        return self.error if self.error else "OK"


def compute_reference_histogram(
    input_dir: str, 
    sample_files: List[Tuple],
    max_samples: int = 50
) -> Optional[np.ndarray]:

    input_path = Path(input_dir)
    histograms = []
    
    samples = sample_files[:max_samples]
    
    for rel_path, filename, src_subdir, idx in samples:
        try:
            in_path = input_path / rel_path
            img = core.read_image(in_path)
            
            if img is not None:
                hist = core.compute_histogram(img)
                histograms.append(hist)
        except Exception:
            continue
    
    if not histograms:
        return None
    
    return np.mean(histograms, axis=0)
