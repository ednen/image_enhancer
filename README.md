#  Staj - Aerial Image Enhancer (Internship Project)

A high-performance batch image enhancement tool designed for aerial photography. Optimized for processing building rooftops, terrain, and urban imagery at scale.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

##  Features

- Automatic Analysis - Detects image properties (brightness, contrast, sharpness, haze) automatically
- Smart Enhancement - Calculates optimal parameters per image
- Batch Processing - Process millions of files with parallel execution
- Memory Efficient - Path-based pipeline with minimal RAM footprint
- GUI & CLI - Both graphical and command-line interfaces
- Production Ready - Atomic writes, disk checks, retry logic, telemetry

##  Before & After

| Original | Enhanced |
|----------|----------|
| Low contrast aerial shot | Improved clarity and detail |
| Hazy building rooftops | Dehazed with preserved colors |

##  Installation

```bash
# Clone the repository
git clone https://github.com/username/staj.git
cd staj

# Install dependencies
pip install -r Staj/requirements.txt
```

### Requirements

```
opencv-python>=4.5.0
numpy>=1.20.0
Pillow>=8.0.0
psutil>=5.8.0
```

For headless servers, use `opencv-python-headless` instead.

##  Usage

### GUI Mode

```bash
python -m Staj
```

### CLI Mode

```bash
python -m Staj.cli --input ./raw_images --output ./enhanced --mode auto
```

### As a Library

```python
from Staj import analyze, calculate_auto_params, enhance
import cv2

# Load image
img = cv2.imread("aerial_photo.png", cv2.IMREAD_UNCHANGED)

# Analyze and enhance
analysis = analyze(img)
params = calculate_auto_params(analysis)
result = enhance(img, params, analysis)

# Save
cv2.imwrite("enhanced.png", result)
```

##  Configuration

Edit `config.json` to customize thresholds and processing parameters:

```json
{
  "thresholds": {
    "haziness": 40,
    "contrast_min": 45,
    "sharpness": 1500,
    "saturation_min": 100
  },
  "processing": {
    "batch_size": 15,
    "files_per_subdir": 10000,
    "skip_existing": true
  }
}
```

##  Algorithms

All core algorithms are industry-standard, peer-reviewed techniques:

| Algorithm | Reference | Used For |
|-----------|-----------|----------|
| Dark Channel Prior | He et al., CVPR 2009 (Best Paper) | Dehazing |
| CLAHE | Zuiderveld, 1994 | Adaptive contrast |
| Bilateral Filter | Tomasi & Manduchi, 1998 | Edge-preserving smoothing |
| Non-Local Means | Buades et al., 2005 | Denoising |
| Unsharp Masking | Traditional (1930s+) | Sharpening |
| LAB Color Space | CIE 1976 | Color processing |

### Processing Pipeline

```
Input Image
    │
    ├─► Dehaze (Dark Channel Prior)
    │
    ├─► White Balance (LAB a,b channels)
    │
    ├─► Auto Levels (Percentile stretch)
    │
    ├─► CLAHE (Adaptive histogram)
    │
    ├─► Gamma Correction (Adaptive per-region)
    │
    ├─► Detail Enhancement (Bilateral + add)
    │
    ├─► Sharpening (Unsharp mask)
    │
    ├─► Saturation (HSV adjustment)
    │
    └─► Denoise (Median + Bilateral + NLMeans)
            │
            ▼
      Enhanced Image
```

##  Performance

| Metric | Value |
|--------|-------|
| Throughput | ~4 img/s (1024×768) |
| Memory | <500MB for 10K queue |
| Batch Size | 15 images/process call |
| PNG Compression | Level 4 (balanced) |

### Telemetry

Real-time progress includes:
- RAM usage (MB)
- Queue depth
- Throughput (MB/s)
- ETA with smoothing

##  Testing

```bash
# Run all tests
python -m pytest Staj/tests/ -v

# Run specific test class
python -m pytest Staj/tests/test_core.py::TestPerformance -v
```

### Test Coverage

| Test Class | Tests | Description |
|------------|-------|-------------|
| TestAnalyze | 6 | Image analysis |
| TestCalculateAutoParams | 5 | Parameter calculation |
| TestEnhance | 3 | Enhancement output |
| TestPerformance | 3 | 4K benchmarks |
| TestMemory | 2 | Leak detection |
| TestLoadSimulation | 1 | 1K image simulation |
| TestQualityRegression | 3 | Quality metrics |
| Total | 23 | All passing  |

##  Project Structure

```
Staj/
├── __init__.py        # Package exports
├── __main__.py        # Entry point
├── core.py            # Image processing algorithms
├── pipeline.py        # Batch processing engine
├── workers.py         # Utilities (histogram)
├── gui.py             # Tkinter GUI
├── cli.py             # Command-line interface
├── config.py          # Configuration loader
├── config.json        # Default settings
├── requirements.txt   # Dependencies
└── tests/
    ├── test_core.py      # Unit + performance tests
    └── test_pipeline.py  # Integration tests
```

##  Reliability Features

- Atomic Writes: temp file + rename prevents corruption
- Disk Space Check: 50MB minimum before write
- Retry Logic: Failed images retry up to 2 times
- State Persistence: Resume interrupted batches
- Graceful Shutdown: Signal handling (Ctrl+C)

##  Version History

| Version | Changes |
|---------|---------|
| 2.4.0 | Batch processing, telemetry, atomic writes |
| 2.3.0 | Memory optimization, path-based pipeline |
| 2.2.0 | ProcessPoolExecutor, removed unused code |
| 2.1.0 | Error handling model, comprehensive tests |
| 2.0.0 | Initial production release |

##  License

MIT License - see [LICENSE](LICENSE) for details.

   Author
Ozan Bulen
GitHub: @ednen

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

##  References

1. He, K., Sun, J., & Tang, X. (2009). *Single Image Haze Removal Using Dark Channel Prior*. CVPR.
2. Zuiderveld, K. (1994). *Contrast Limited Adaptive Histogram Equalization*. Graphics Gems IV.
3. Tomasi, C., & Manduchi, R. (1998). *Bilateral Filtering for Gray and Color Images*. ICCV.
4. Buades, A., Coll, B., & Morel, J. M. (2005). *A Non-Local Algorithm for Image Denoising*. CVPR.

