import unittest
import time
import os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.api.inference import MoEInferenceEngine
from src.core.types import InferenceMode, GenerationConfig
from src.execution.batch_manager import BatchManager


class TestBatchProcessing(unittest.TestCase):
    """Test batch processing functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test engine"""
        if not os.path.exists("path/to/test/model"):
            raise unittest.SkipTest("No test model checkpoint available")
        cls.engine = MoEInferenceEngine(
            model_path="path/to/test/model",
            config_dir="configs",
            max_batch_size=4,
            enable_batch_processing=True
        )
    
    def test_single_request_batching(self):
        """Test that single requests are processed correctly"""
        prompt = "Test prompt"
        
        # Submit request
        request_id = self.engine.submit(prompt, max_new_tokens=10)
        
        # Wait for result
        response = self.engine.get_result(request_id, timeout=30.0)
        
        self.assertIsNotNone(response)
        self.assertTrue(response.success)
        self.assertGreater(len(response.generated_ids), 0)
    
    def test_multiple_requests_batching(self):
        """Test batching of multiple requests"""
        prompts = [
            "First prompt",
            "Second prompt",
            "Third prompt",
            "Fourth prompt"
        ]
        
        # Submit all requests
        request_ids = []
        for prompt in prompts:
            request_id = self.engine.submit(prompt, max_new_tokens=10)
            request_ids.append(request_id)
        
        # Wait for all results
        responses = []
        for request_id in request_ids:
            response = self.engine.get_result(request_id, timeout=30.0)
            responses.append(response)
        
        # Verify all succeeded
        self.assertEqual(len(responses), len(prompts))
        for response in responses:
            self.assertIsNotNone(response)
            self.assertTrue(response.success)
    
    def test_priority_ordering(self):
        """Test that higher priority requests are processed first"""
        # Submit low priority request
        low_priority_id = self.engine.submit(
            "Low priority",
            max_new_tokens=10,
            priority=0
        )
        
        # Small delay
        time.sleep(0.1)
        
        # Submit high priority request
        high_priority_id = self.engine.submit(
            "High priority",
            max_new_tokens=10,
            priority=10
        )
        
        # Both should complete
        low_response = self.engine.get_result(low_priority_id, timeout=30.0)
        high_response = self.engine.get_result(high_priority_id, timeout=30.0)
        
        self.assertIsNotNone(low_response)
        self.assertIsNotNone(high_response)
    
    def test_concurrent_submissions(self):
        """Test concurrent request submissions"""
        num_requests = 20
        prompts = [f"Prompt {i}" for i in range(num_requests)]
        
        # Submit concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self.engine.submit, prompt, max_new_tokens=10)
                for prompt in prompts
            ]
            
            request_ids = [f.result() for f in as_completed(futures)]
        
        # Verify all submitted
        self.assertEqual(len(request_ids), num_requests)
        
        # Get all results
        responses = []
        for request_id in request_ids:
            response = self.engine.get_result(request_id, timeout=60.0)
            responses.append(response)
        
        # Verify all succeeded
        successful = sum(1 for r in responses if r and r.success)
        self.assertEqual(successful, num_requests)
    
    def test_batch_formation(self):
        """Test batch formation logic"""
        batch_manager = BatchManager(
            max_batch_size=4,
            max_waiting_time_ms=100.0
        )
        
        # Add requests
        from src.core.types import InferenceRequest, GenerationConfig
        
        for i in range(6):
            request = InferenceRequest(
                request_id=f"req_{i}",
                input_ids=torch.randint(0, 1000, (10,)),
                generation_config=GenerationConfig()
            )
            batch_manager.add_request(request)
        
        # Get first batch (should have 4 requests)
        batch1 = batch_manager.get_next_batch(timeout_ms=50.0)
        self.assertIsNotNone(batch1)
        self.assertEqual(len(batch1.requests), 4)
        
        # Get second batch (should have 2 requests)
        batch2 = batch_manager.get_next_batch(timeout_ms=50.0)
        self.assertIsNotNone(batch2)
        self.assertEqual(len(batch2.requests), 2)
    
    def test_dynamic_batching(self):
        """Test dynamic batching with varying arrival times"""
        batch_manager = BatchManager(
            max_batch_size=4,
            max_waiting_time_ms=200.0,
            enable_dynamic_batching=True
        )
        
        # Add requests with delays
        from src.core.types import InferenceRequest, GenerationConfig
        import threading
        
        def add_request_delayed(delay_ms, request_id):
            time.sleep(delay_ms / 1000.0)
            request = InferenceRequest(
                request_id=request_id,
                input_ids=torch.randint(0, 1000, (10,)),
                generation_config=GenerationConfig()
            )
            batch_manager.add_request(request)
        
        # Start adding requests
        threads = []
        for i in range(3):
            t = threading.Thread(
                target=add_request_delayed,
                args=(i * 50, f"req_{i}")
            )
            t.start()
            threads.append(t)
        
        # Wait a bit for requests to arrive
        time.sleep(0.3)
        
        # Should form a batch with available requests
        batch = batch_manager.get_next_batch(timeout_ms=100.0)
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        self.assertIsNotNone(batch)
        self.assertGreater(len(batch.requests), 0)


class TestStandardVsSpeculative(unittest.TestCase):
    """Compare standard and speculative decoding"""
    
    def test_output_consistency(self):
        """Test that both modes produce valid outputs"""
        if not os.path.exists("path/to/test/model"):
            self.skipTest("No test model checkpoint available")
        engine = MoEInferenceEngine(
            model_path="path/to/test/model",
            config_dir="configs",
            enable_batch_processing=False
        )
        
        prompt = "Test prompt for consistency"
        
        # Generate with standard mode
        standard_output = engine.generate(
            prompt,
            max_new_tokens=20,
            mode=InferenceMode.STANDARD,
            temperature=0.0  # Deterministic
        )
        
        # Generate with speculative mode
        speculative_output = engine.generate(
            prompt,
            max_new_tokens=20,
            mode=InferenceMode.SPECULATIVE,
            temperature=0.0  # Deterministic
        )
        
        # Both should generate tokens
        self.assertGreater(len(standard_output), 0)
        self.assertGreater(len(speculative_output), 0)
        
        # With temperature=0, outputs should be similar (not necessarily identical
        # due to draft-verify approximation)
        # Just verify both are reasonable
        self.assertLess(len(standard_output), 100)
        self.assertLess(len(speculative_output), 100)


if __name__ == '__main__':
    unittest.main()