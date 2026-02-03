# Image Enhancer Pipeline

High-performance image processing pipeline with GUI & CLI support.

##  Features

- Parallel image processing (CPU-bound + I/O-bound separation)
- Modular architecture
- CLI and GUI interfaces
- Configurable pipeline
- Error-tolerant processing
- Progress tracking

##  Architecture

The project follows a layered pipeline architecture:

1. Reader (I/O)
2. Processor (CPU)
3. Writer (I/O)

Separation allows:
- Efficient multiprocessing  
- Non-blocking I/O  
- Scalable design  

##  Structure

Staj/
├── cli.py          # Command line interface  
├── gui.py          # Desktop interface  
├── pipeline.py     # Orchestration  
├── workers.py      # CPU workers  
├── core.py         # Business logic  
├── config.py       # Settings  
└── __main__.py     # Entry point  

##  Usage

```bash
python run_app.py
