import unittest
import tempfile
import shutil
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Staj.pipeline import Pipeline
from Staj.config import Config
from Staj import core


class TestPipelineIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="staj_test_")
        cls.input_dir = Path(cls.test_dir) / "input"
        cls.output_dir = Path(cls.test_dir) / "output"
        
        cls.input_dir.mkdir()
        cls.output_dir.mkdir()

        cls._create_test_images(cls.input_dir, count=5)
    
    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(cls.test_dir)
        except Exception:
            pass
    
    @staticmethod
    def _create_test_images(directory: Path, count: int = 5):
        for i in range(count):
            if i % 3 == 0:
                img = np.random.randint(0, 80, (256, 256, 3), dtype=np.uint8)
            elif i % 3 == 1:
                img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            else:
                img = np.random.randint(150, 255, (256, 256, 3), dtype=np.uint8)
            
            path = directory / f"test_image_{i}.png"
            cv2.imwrite(str(path), img)
    
    def setUp(self):
        Config._instance = None

        for f in self.output_dir.iterdir():
            if f.is_file():
                f.unlink()
    
    def test_pipeline_scan_files(self):
        pipeline = Pipeline()
        count = pipeline.scan_files(
            str(self.input_dir),
            str(self.output_dir),
            use_state=False
        )
        
        self.assertEqual(count, 5)
        self.assertEqual(len(pipeline.files), 5)
    
    def test_pipeline_processes_images(self):
        pipeline = Pipeline()

        progress_updates = []
        def on_progress(info):
            progress_updates.append(info.current)
        
        completed = []
        def on_complete(processed, errors, error_list):
            completed.append((processed, errors))
        
        pipeline.on_progress = on_progress
        pipeline.on_complete = on_complete

        pipeline.scan_files(
            str(self.input_dir),
            str(self.output_dir),
            use_state=False
        )

        pipeline.run(mode="auto", parallel=True)

        self.assertEqual(len(completed), 1)
        processed, errors = completed[0]

        self.assertGreater(processed, 0)

        output_files = list(self.output_dir.glob("*.png"))
        self.assertGreater(len(output_files), 0)
    
    def test_pipeline_output_quality(self):
        pipeline = Pipeline()
        pipeline.scan_files(
            str(self.input_dir),
            str(self.output_dir),
            use_state=False
        )
        pipeline.run(mode="auto", parallel=True)

        for output_file in self.output_dir.glob("*.png"):
            img = cv2.imread(str(output_file))

            self.assertIsNotNone(img, f"{output_file} okunamadı")

            self.assertEqual(img.shape, (256, 256, 3))

            self.assertTrue(np.all(img >= 0))
            self.assertTrue(np.all(img <= 255))
    
    def test_pipeline_skip_existing(self):
        pipeline1 = Pipeline()
        pipeline1.scan_files(
            str(self.input_dir),
            str(self.output_dir),
            use_state=True
        )
        pipeline1.run(mode="auto", parallel=True)
        
        first_run_count = pipeline1.processed

        pipeline2 = Pipeline()
        count = pipeline2.scan_files(
            str(self.input_dir),
            str(self.output_dir),
            use_state=True  
        )

        self.assertEqual(count, 0)

        state_file = self.output_dir / ".enhance_state.json"
        if state_file.exists():
            state_file.unlink()


class TestSingleImageProcessing(unittest.TestCase):

    def setUp(self):
        Config._instance = None
    
    def test_full_processing_pipeline(self):

        img = np.random.randint(30, 150, (256, 256, 3), dtype=np.uint8)

        analysis = core.analyze(img)
        self.assertIn('brightness', analysis)

        params = core.calculate_auto_params(analysis)
        self.assertIn('sharpen', params)

        enhanced = core.enhance(img, params, analysis)
        self.assertEqual(enhanced.shape, img.shape)

        success, buffer = cv2.imencode(".png", enhanced)
        self.assertTrue(success)

        decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(decoded)
        np.testing.assert_array_equal(enhanced, decoded)


if __name__ == '__main__':
    unittest.main()
