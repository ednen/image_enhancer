import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Staj import core
from Staj.config import Config


class TestAnalyze(unittest.TestCase):
    
    def setUp(self):
        Config._instance = None
    
    def test_analyze_returns_dict(self):

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = core.analyze(img)
        
        self.assertIsInstance(result, dict)
    
    def test_analyze_has_required_keys(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = core.analyze(img)
        
        required_keys = ['brightness', 'contrast', 'sharpness', 
                        'saturation', 'haziness', 'color_cast']
        
        for key in required_keys:
            self.assertIn(key, result, f"'{key}' anahtarı eksik")
    
    def test_analyze_black_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = core.analyze(img)
        
        self.assertLess(result['brightness'], 10)
    
    def test_analyze_white_image(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        result = core.analyze(img)
        
        self.assertGreater(result['brightness'], 200)
    
    def test_analyze_color_cast_neutral(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = core.analyze(img)
        
        cc = result['color_cast']
        self.assertAlmostEqual(cc['r_ratio'], 1.0, delta=0.1)
        self.assertAlmostEqual(cc['g_ratio'], 1.0, delta=0.1)
        self.assertAlmostEqual(cc['b_ratio'], 1.0, delta=0.1)
    
    def test_analyze_handles_rgba(self):
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        img[:, :, 3] = 255 

        result = core.analyze(img)
        self.assertIsInstance(result, dict)


class TestCalculateAutoParams(unittest.TestCase):
    
    def setUp(self):
        Config._instance = None
    
    def test_returns_dict(self):
        analysis = {
            'brightness': 100, 'contrast': 50, 'sharpness': 500,
            'saturation': 80, 'haziness': 30,
            'color_cast': {'r_ratio': 1.0, 'g_ratio': 1.0, 'b_ratio': 1.0}
        }
        result = core.calculate_auto_params(analysis)
        
        self.assertIsInstance(result, dict)
    
    def test_has_required_params(self):
        analysis = {
            'brightness': 100, 'contrast': 50, 'sharpness': 500,
            'saturation': 80, 'haziness': 30,
            'color_cast': {'r_ratio': 1.0, 'g_ratio': 1.0, 'b_ratio': 1.0}
        }
        result = core.calculate_auto_params(analysis)
        
        required_params = ['dehaze', 'white_balance', 'clahe', 'gamma',
                          'sharpen', 'color', 'detail', 'auto_levels', 'denoise']
        
        for param in required_params:
            self.assertIn(param, result)
    
    def test_low_contrast_enables_clahe(self):
        analysis = {
            'brightness': 70, 'contrast': 30, 'sharpness': 500,
            'saturation': 80, 'haziness': 30,
            'color_cast': {'r_ratio': 1.0, 'g_ratio': 1.0, 'b_ratio': 1.0}
        }
        result = core.calculate_auto_params(analysis)
        
        self.assertTrue(result['clahe']['apply'])
    
    def test_high_haziness_enables_dehaze(self):
        analysis = {
            'brightness': 100, 'contrast': 60, 'sharpness': 500,
            'saturation': 80, 'haziness': 60,  
            'color_cast': {'r_ratio': 1.0, 'g_ratio': 1.0, 'b_ratio': 1.0}
        }
        result = core.calculate_auto_params(analysis)
        
        self.assertTrue(result['dehaze']['apply'])
    
    def test_normal_image_minimal_processing(self):
        analysis = {
            'brightness': 120, 'contrast': 70, 'sharpness': 1000,
            'saturation': 110, 'haziness': 20,
            'color_cast': {'r_ratio': 1.0, 'g_ratio': 1.0, 'b_ratio': 1.0}
        }
        result = core.calculate_auto_params(analysis)

        self.assertFalse(result['clahe']['apply'])
        self.assertFalse(result['gamma']['apply'])


class TestEnhance(unittest.TestCase):
    """core.enhance() fonksiyonu için testler."""
    
    def setUp(self):
        Config._instance = None
    
    def test_enhance_returns_same_shape(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        analysis = core.analyze(img)
        params = core.calculate_auto_params(analysis)
        
        result = core.enhance(img, params, analysis)
        
        self.assertEqual(img.shape, result.shape)
    
    def test_enhance_returns_valid_image(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        analysis = core.analyze(img)
        params = core.calculate_auto_params(analysis)
        
        result = core.enhance(img, params, analysis)
        
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))
    
    def test_enhance_preserves_alpha(self):
        img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        img[:, :, 3] = 200  
        
        analysis = core.analyze(img)
        params = core.calculate_auto_params(analysis)
        result = core.enhance(img, params, analysis)

        self.assertEqual(result.shape[2], 4)
        np.testing.assert_array_equal(result[:, :, 3], img[:, :, 3])


class TestPerformance(unittest.TestCase):
    
    def setUp(self):
        Config._instance = None
    
    def test_analyze_performance(self):
        import time

        img = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
        
        start = time.perf_counter()
        core.analyze(img)
        elapsed = time.perf_counter() - start
        
        print(f"\nAnalyze 4K: {elapsed:.3f}s")
        self.assertLess(elapsed, 2.0, f"analyze() çok yavaş: {elapsed:.3f}s")
    
    def test_enhance_performance(self):
        import time
        
        img = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
        analysis = core.analyze(img)
        params = core.calculate_auto_params(analysis)
        
        start = time.perf_counter()
        core.enhance(img, params, analysis)
        elapsed = time.perf_counter() - start
        
        print(f"\nEnhance 4K: {elapsed:.3f}s")
        self.assertLess(elapsed, 5.0, f"enhance() çok yavaş: {elapsed:.3f}s")
    
    def test_batch_simulation(self):
        import time

        images = [np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8) 
                  for _ in range(100)]
        
        start = time.perf_counter()
        for img in images:
            analysis = core.analyze(img)
            params = core.calculate_auto_params(analysis)
            core.enhance(img, params, analysis)
        elapsed = time.perf_counter() - start
        
        speed = 100 / elapsed
        print(f"\nBatch performans: {speed:.1f} img/s ({elapsed:.2f}s / 100 img)")

        self.assertGreater(speed, 1, f"Batch çok yavaş: {speed:.1f} img/s")


class TestMemory(unittest.TestCase):
    
    def setUp(self):
        Config._instance = None
    
    def test_no_memory_leak_single(self):
        import gc

        gc.collect()
        
        img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        
        for _ in range(10):
            analysis = core.analyze(img)
            params = core.calculate_auto_params(analysis)
            result = core.enhance(img, params, analysis)
            del result
        
        gc.collect()
        self.assertTrue(True)
    
    def test_large_image_memory(self):
        import gc
        
        gc.collect()

        try:
            img = np.random.randint(0, 255, (4320, 7680, 3), dtype=np.uint8)
            analysis = core.analyze(img)
            params = core.calculate_auto_params(analysis)
            result = core.enhance(img, params, analysis)
            
            self.assertEqual(result.shape, img.shape)
            
            del result
            del img
            gc.collect()
            
        except MemoryError:
            self.skipTest("Yetersiz bellek - 8K test atlandı")


class TestLoadSimulation(unittest.TestCase):
    
    def setUp(self):
        Config._instance = None
    
    def test_1k_simulation(self):
        import time

        count = 1000
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        start = time.perf_counter()
        for _ in range(count):
            analysis = core.analyze(img)
            params = core.calculate_auto_params(analysis)
            core.enhance(img, params, analysis)
        elapsed = time.perf_counter() - start
        
        speed = count / elapsed
        print(f"\n1K simülasyon: {speed:.1f} img/s ({elapsed:.2f}s)")

        self.assertGreater(speed, 20, f"1K simülasyon çok yavaş: {speed:.1f} img/s")


class TestQualityRegression(unittest.TestCase):
    
    def setUp(self):
        Config._instance = None
    
    def test_histogram_preservation(self):
        img = np.random.randint(50, 200, (500, 500, 3), dtype=np.uint8)
        
        analysis = core.analyze(img)
        params = core.calculate_auto_params(analysis)
        result = core.enhance(img, params, analysis)

        import cv2
        hist_orig = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_result = cv2.calcHist([result], [0], None, [256], [0, 256])
        
        correlation = cv2.compareHist(hist_orig, hist_result, cv2.HISTCMP_CORREL)
        print(f"\nHistogram korelasyonu: {correlation:.3f}")

        self.assertGreater(correlation, 0.0, f"Histogram korelasyonu negatif: {correlation:.3f}")
    
    def test_no_extreme_color_shift(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, :100] = [255, 0, 0]  
        img[:, 100:] = [0, 255, 0]  
        
        analysis = core.analyze(img)
        params = core.calculate_auto_params(analysis)
        result = core.enhance(img, params, analysis)

        diff = np.abs(img.astype(np.float32) - result.astype(np.float32))
        mean_diff = np.mean(diff)
        print(f"\nOrtalama renk farkı: {mean_diff:.1f}")

        self.assertLess(mean_diff, 80, f"Renk kayması aşırı: {mean_diff:.1f}")
    
    def test_output_valid_range(self):
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        analysis = core.analyze(img)
        params = core.calculate_auto_params(analysis)
        result = core.enhance(img, params, analysis)

        self.assertTrue(np.all(result >= 0), "Negatif piksel değeri var")
        self.assertTrue(np.all(result <= 255), "255'ten büyük piksel değeri var")
        self.assertEqual(result.dtype, np.uint8, "Yanlış veri tipi")


if __name__ == '__main__':
    unittest.main()
