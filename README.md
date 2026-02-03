# Image Enhancer Pipeline

High-performance, headless-ready image processing pipeline with GUI & CLI interfaces, designed for large-scale batch enhancement tasks.

##  Features

* Headless Mode for server / automation usage
* CLI & GUI interfaces from the same core
* True parallel processing (CPU + I/O separation)
* Modular, pluggable processing steps
* Progress tracking & safe interruption
* Config-driven behavior
* Error-tolerant pipeline

##  Architecture

Layered pipeline architecture following separation of concerns:

1. Reader (I/O bound)
2. Processor (CPU bound)
3. Writer (I/O bound)

This design enables:

* Multiprocessing without UI freeze
* Scalable throughput
* Easy extension with new filters
* Testable components

##  Project Structure

```
Staj/
├── cli.py          # Headless CLI interface  
├── gui.py          # Desktop UI  
├── pipeline.py     # Orchestration layer  
├── workers.py      # CPU processing workers  
├── core.py         # Business logic  
├── config.py       # Configuration  
└── __main__.py     # Entry point
```

##  Usage

### GUI Mode

```bash
python run_app.py
```

### Headless / CLI Mode

```bash
python -m Staj --input ./images --output ./result --workers 8
```

### Automation Example

```bash
python -m Staj \
  --input /data/raw \
  --output /data/enhanced \
  --config config.json \
  --headless
```

##  Configuration

All behavior is controlled via `config.json`:

* processing parameters
* worker count
* output format
* logging level
* pipeline steps

##  Requirements

See `requirements.txt`

##  Design Goals

* Process millions of images reliably
* Keep UI responsive
* Production-ready headless usage
* Clean, maintainable architecture
* Extensible processing steps
* Safe crash recovery

##  Technical Highlights

* CPU / I-O separation
* Queue based pipeline
* Graceful shutdown
* Progress reporting
* Cross-platform support

---

Target Use Cases

* Dataset preprocessing
* Bulk image enhancement
* Automation pipelines
* Research workflows
* Offline batch processing
