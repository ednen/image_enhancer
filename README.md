# Image Enhancer – Production Ready Pipeline

High-performance image enhancement pipeline designed for **large scale batch processing** with both **GUI and Headless CLI** interfaces.

##  Core Capabilities

*  Separation of I/O and CPU workloads
*  Multiprocessing with `ProcessPoolExecutor`
*  Producer → Consumer → Writer pipeline
*  Headless automation mode
*  Shutdown safe
*  Logging & error recovery
*  Progress + ETA estimation
*  Config-driven behavior

---

##  Real Architecture (from code)

```
[Reader Thread] → input_queue → [Process Pool]
                               (CPU workers)
                    → result_queue → [Writer Thread]
```

### Components

* Reader (Thread – I/O bound)

  * Scans input directory
  * Streams images without blocking CPU
  * Feeds bounded queue

* Processor (Process Pool – CPU bound)

  * Runs enhancement algorithms
  * Histogram matching
  * Core image operations

* **Writer (Thread – I/O bound)**

  * Saves results
  * Handles disk latency
  * Keeps pipeline flowing

This design prevents:

* UI freeze
* GIL bottlenecks
* RAM explosion on millions of files

---

##  Project Layout

```
Staj/
├── cli.py        → Headless interface
├── gui.py        → Desktop UI
├── pipeline.py   → Orchestration & queues
├── workers.py    → WriteTask + utilities
├── core.py       → Image algorithms
├── config.py     → Config management
└── __main__.py   → Entry point
```

---

##  Usage

### GUI Mode

```bash
python run_app.py
```

### Headless CLI Mode

```bash
python -m Staj \
  --input ./images \
  --output ./result \
  --mode fast
```

### Server / Automation

```bash
python -m Staj \
  --input /data/raw \
  --output /data/enhanced \
  --config config.json \
  --headless
```

---

##  Configuration

All behavior is controlled via `config.json`

* worker count
* enhancement parameters
* logging level
* retry rules
* output format

---

##  Technical Highlights

* Producer–Consumer with **bounded queues**
* CPU/GIL aware design
* Exception isolation per image
* ETA based on real throughput
* Single source of truth (`core.py`)
* Cross-platform logging
* Modular steps

---

##  Target Scenarios

* Millions of image preprocessing
* Dataset normalization
* Research pipelines
* Offline batch enhancement
* Automated quality improvement

---

##  Requirements

See `Staj/requirements.txt`

---

##  Notes

* Designed for long-running jobs
* Safe interruption supported
* All components reusable from other projects
* Same core used by GUI & CLI

* Research workflows
* Offline batch processing
