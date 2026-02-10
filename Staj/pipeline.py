
import os
import sys
import time
import json
import signal
import logging
import traceback
from pathlib import Path
from queue import Queue, Empty, Full
from threading import Thread, Event
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional, Callable, Dict, Set
from dataclasses import dataclass, field

from .config import config
from . import workers
from .workers import WriteTask

def setup_logging(output_dir: str) -> logging.Logger:
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "enhance.log"
    
    logger = logging.getLogger("image_enhancer")
    logger.setLevel(logging.DEBUG)

    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = logging.getLogger("image_enhancer")

@dataclass
class ProcessTask:
    img_data: bytes
    rel_path: str
    filename: str
    src_subdir: str
    global_idx: int
    retry_count: int = 0


@dataclass
class ProgressInfo:
    current: int
    total: int
    filename: str
    elapsed: float
    speed: float
    eta: float
    errors: int
    pending_writes: int


@dataclass
class ErrorInfo:
    filename: str
    rel_path: str
    error: str
    stage: str
    timestamp: float = field(default_factory=time.time)

class ProcessingState:
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.state_file = self.output_dir / ".enhance_state.json"
        self.processed_files: Set[str] = set()
        self._load()
    
    def _load(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_files = set(data.get('processed', []))
                logger.info(f"State yüklendi: {len(self.processed_files)} dosya işlenmiş")
            except Exception as e:
                logger.warning(f"State yüklenemedi: {e}")
                self.processed_files = set()
    
    def save(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump({'processed': list(self.processed_files)}, f)
        except Exception as e:
            logger.error(f"State kaydedilemedi: {e}")
    
    def mark_processed(self, rel_path: str) -> None:
        self.processed_files.add(rel_path)
    
    def is_processed(self, rel_path: str) -> bool:
        return rel_path in self.processed_files
    
    def clear(self) -> None:
        self.processed_files = set()
        if self.state_file.exists():
            self.state_file.unlink()
    
    @property
    def count(self) -> int:
        return len(self.processed_files)

class Pipeline:

    
    READ_QUEUE_SIZE = 30
    WRITE_QUEUE_SIZE = 50
    MAX_RETRIES = 2
    
    def __init__(self):
        self.files: List[Tuple[str, str, str, int]] = []
        self.total_files = 0
        self.processed = 0
        self.errors: List[ErrorInfo] = []
        self.start_time: Optional[float] = None
        self.state: Optional[ProcessingState] = None

        self.read_queue: Queue = Queue(maxsize=self.READ_QUEUE_SIZE)
        self.write_queue: Queue = Queue(maxsize=self.WRITE_QUEUE_SIZE)

        self.reader_thread: Optional[Thread] = None
        self.writer_thread: Optional[Thread] = None
        self.executor: Optional[ProcessPoolExecutor] = None

        self.shutdown_event = Event()

        self.ref_histogram = None
        self.mode = "auto"
        self.manual_params = None
        self.input_dir = ""
        self.output_dir = ""
        self.current_filename = ""

        self.on_progress: Optional[Callable] = None
        self.on_complete: Optional[Callable] = None
        
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self) -> None:
        def handler(signum, frame):
            logger.info("Signal alındı, kapatılıyor")
            self.shutdown()
        
        try:
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
        except:
            pass
    
    def scan_files(self, input_dir: str, output_dir: str, use_state: bool = True) -> int:
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        input_path = Path(input_dir)
        
        if use_state:
            self.state = ProcessingState(output_dir)
        
        all_files = []
        
        def scan_recursive(directory: Path, base_path: str = "") -> None:
            try:
                for entry in sorted(directory.iterdir()):
                    if entry.is_file() and entry.name.lower().endswith('.png'):
                        rel_path = f"{base_path}/{entry.name}" if base_path else entry.name
                        all_files.append((rel_path, entry.name, base_path))
                    elif entry.is_dir():
                        new_base = f"{base_path}/{entry.name}" if base_path else entry.name
                        scan_recursive(entry, new_base)
            except PermissionError:
                pass
        
        scan_recursive(input_path)

        self.files = []
        skipped = 0
        
        for idx, (rel_path, filename, src_subdir) in enumerate(all_files):
            if self.state and self.state.is_processed(rel_path):
                skipped += 1
                continue
            self.files.append((rel_path, filename, src_subdir, idx))
        
        self.total_files = len(self.files)
        
        if skipped > 0:
            logger.info(f"{skipped} dosya daha önce işlenmiş")
        logger.info(f"İşlenecek: {self.total_files} dosya")
        
        return self.total_files
    
    def prepare_histogram(self, input_dir: str, enable: bool = True) -> None:
        if not enable or not self.files:
            self.ref_histogram = None
            return
        
        max_samples = config.processing.get('histogram_samples', 50)
        self.ref_histogram = workers.compute_reference_histogram(input_dir, self.files, max_samples)
    
    def run(self, mode: str = "auto", manual_params: Optional[Dict] = None, parallel: bool = True) -> None:
        if not self.files:
            logger.warning("İşlenecek dosya yok")
            return
        
        self.mode = mode
        self.manual_params = manual_params
        self.start_time = time.time()
        self.processed = 0
        self.errors = []
        self.shutdown_event.clear()
        
        setup_logging(self.output_dir)
        
        max_workers = config.processing.get('max_workers') or max(1, os.cpu_count() - 1)
        
        logger.info(f"Başlatılıyor: {self.total_files} dosya, {max_workers} worker")
        
        try:
            self.reader_thread = Thread(target=self._reader_loop, daemon=True)
            self.reader_thread.start()

            self.writer_thread = Thread(target=self._writer_loop, daemon=True)
            self.writer_thread.start()

            self.executor = ProcessPoolExecutor(max_workers=max_workers)

            self._main_loop()
            
        except Exception as e:
            logger.error(f"Pipeline hatası: {e}")
        finally:
            self._cleanup()
        
        self._finish()
    
    def _reader_loop(self) -> None:
        input_path = Path(self.input_dir)
        
        for rel_path, filename, src_subdir, global_idx in self.files:
            if self.shutdown_event.is_set():
                break
            
            try:
                in_path = input_path / rel_path
                with open(in_path, 'rb') as f:
                    img_data = f.read()
                
                task = ProcessTask(
                    img_data=img_data,
                    rel_path=rel_path,
                    filename=filename,
                    src_subdir=src_subdir,
                    global_idx=global_idx
                )

                while not self.shutdown_event.is_set():
                    try:
                        self.read_queue.put(task, timeout=0.5)
                        break
                    except Full:
                        continue
                
            except Exception as e:
                self.errors.append(ErrorInfo(filename, rel_path, str(e), "read"))
                logger.error(f"Okuma hatası: {filename}")

        try:
            self.read_queue.put(None, timeout=5.0)
        except:
            pass
    
    def _writer_loop(self) -> None:
        write_count = 0
        last_save = time.time()
        
        while not self.shutdown_event.is_set():
            try:
                task = self.write_queue.get(timeout=0.5)
                
                if task is None:
                    break
                
                try:
                    task.path.parent.mkdir(parents=True, exist_ok=True)

                    tmp_path = task.path.with_suffix('.tmp')
                    task.buffer.tofile(str(tmp_path))
                    tmp_path.replace(task.path)  
                    
                    if self.state:
                        self.state.mark_processed(task.rel_path)
                    
                    self.processed += 1
                    write_count += 1

                    if write_count % 50 == 0 or time.time() - last_save > 30:
                        if self.state:
                            self.state.save()
                        last_save = time.time()
                    
                except Exception as e:
                    tmp_path = task.path.with_suffix('.tmp')
                    if tmp_path.exists():
                        try:
                            tmp_path.unlink()
                        except:
                            pass
                    self.errors.append(ErrorInfo(task.filename, task.rel_path, str(e), "write"))
                
            except Empty:
                continue
        
        if self.state:
            self.state.save()
    
    def _main_loop(self) -> None:
        config_dict = config.to_dict()
        files_per_subdir = config.processing.get('files_per_subdir', 10000)
        output_path = str(Path(self.output_dir))

        max_pending = (config.processing.get('max_workers') or os.cpu_count()) * 2
        active_futures = {}
        
        reader_done = False
        last_progress = time.time()
        
        while not self.shutdown_event.is_set():
            if time.time() - last_progress > 0.5:
                self._report_progress()
                last_progress = time.time()

            while len(active_futures) < max_pending and not reader_done:
                try:
                    task = self.read_queue.get(timeout=0.1)
                    
                    if task is None:
                        reader_done = True
                        break
                    
                    future = self.executor.submit(
                        _process_single,
                        task.img_data, task.rel_path, task.filename,
                        task.src_subdir, task.global_idx, self.mode,
                        self.manual_params, config_dict, self.ref_histogram,
                        files_per_subdir, output_path
                    )
                    active_futures[future] = task
                    
                except Empty:
                    break

            done = [f for f in active_futures if f.done()]
            
            for future in done:
                task = active_futures.pop(future)
                self.current_filename = task.filename
                
                try:
                    result = future.result(timeout=1.0)
                    
                    if result['success']:
                        write_task = WriteTask(
                            path=Path(result['output_path']),
                            buffer=result['buffer'],
                            filename=task.filename,
                            rel_path=task.rel_path
                        )

                        while not self.shutdown_event.is_set():
                            try:
                                self.write_queue.put(write_task, timeout=0.5)
                                logger.debug(f"Write queue'ya eklendi: {task.filename}")
                                break
                            except Full:
                                continue
                    else:
                        logger.error(f"İşleme hatası: {task.filename} - {result.get('error', '?')}")
                        if task.retry_count < self.MAX_RETRIES:
                            task.retry_count += 1
                            try:
                                self.read_queue.put(task, timeout=1.0)
                            except:
                                self.errors.append(ErrorInfo(task.filename, task.rel_path, result.get('error', '?'), "process"))
                        else:
                            self.errors.append(ErrorInfo(task.filename, task.rel_path, result.get('error', '?'), "process"))
                
                except Exception as e:
                    self.errors.append(ErrorInfo(task.filename, task.rel_path, str(e), "process"))

            if reader_done and len(active_futures) == 0:
                break
            
            if not done:
                time.sleep(0.01)

        try:
            self.write_queue.put(None, timeout=5.0)
        except:
            pass
    
    def _report_progress(self) -> None:
        if not self.on_progress:
            return
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        speed = self.processed / elapsed if elapsed > 0 else 0
        remaining = self.total_files - self.processed - len(self.errors)
        eta = remaining / speed if speed > 0 else 0
        
        info = ProgressInfo(
            current=self.processed,
            total=self.total_files,
            filename=self.current_filename,
            elapsed=elapsed,
            speed=speed,
            eta=eta,
            errors=len(self.errors),
            pending_writes=self.write_queue.qsize()
        )
        self.on_progress(info)
    
    def _cleanup(self) -> None:
        logger.info("Cleanup")
        
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None
        
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=3.0)
        
        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=3.0)
        
        if self.state:
            self.state.save()
    
    def _finish(self) -> None:
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        if self.errors:
            logger.warning(f"\nHATALAR ({len(self.errors)}):")
            for err in self.errors[:10]:
                logger.warning(f"  [{err.stage}] {err.rel_path}: {err.error}")
            if len(self.errors) > 10:
                logger.warning(f"  ... +{len(self.errors) - 10} hata")
        
        logger.info(f"Tamamlandı: {self.processed}/{self.total_files}, {len(self.errors)} hata, {elapsed:.1f}sn")
        
        if self.on_complete:
            self.on_complete(self.processed, len(self.errors), self.errors)
    
    def shutdown(self) -> None:
        logger.info("Shutdown")
        self.shutdown_event.set()
    
    def cancel(self) -> None:
        self.shutdown()
    
    def clear_state(self) -> None:
        if self.state:
            self.state.clear()
        elif self.output_dir:
            ProcessingState(self.output_dir).clear()

def _process_single(
    img_data: bytes,
    rel_path: str,
    filename: str,
    src_subdir: str,
    global_idx: int,
    mode: str,
    manual_params,
    config_dict: Dict,
    ref_histogram,
    files_per_subdir: int,
    output_dir: str
) -> Dict:

    import cv2
    import numpy as np
    import traceback
    from pathlib import Path
    
    try:

        import sys
        import os

        current_file = os.path.abspath(__file__)
        package_dir = os.path.dirname(os.path.dirname(current_file))
        if package_dir not in sys.path:
            sys.path.insert(0, package_dir)

        from Staj import core
        from Staj.config import Config

        Config._instance = None  
        cfg = Config()
        cfg._config = config_dict

        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            return {'success': False, 'error': 'Decode hatası', 'filename': filename}

        analysis = core.analyze(img)
        params = core.calculate_auto_params(analysis) if mode == "auto" else (manual_params or {})
        enhanced = core.enhance(img, params, analysis)

        if ref_histogram is not None:
            enhanced = core.apply_histogram_matching(enhanced, ref_histogram)

        success, buffer = cv2.imencode(".png", enhanced)
        if not success:
            return {'success': False, 'error': 'Encode hatası', 'filename': filename}

        output_path = Path(output_dir)
        subdir_index = global_idx // files_per_subdir
        
        if subdir_index > 0:
            target_dir = output_path / f"paket_{subdir_index}" / src_subdir if src_subdir else output_path / f"paket_{subdir_index}"
        else:
            target_dir = output_path / src_subdir if src_subdir else output_path
        
        return {
            'success': True,
            'buffer': buffer,
            'output_path': str(target_dir / filename),
            'rel_path': rel_path,
            'filename': filename
        }
        
    except Exception as e:

        tb = traceback.format_exc()
        return {
            'success': False, 
            'error': f"{str(e)}\n{tb}", 
            'filename': filename
        }
