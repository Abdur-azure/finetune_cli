from datasets import Dataset
import unittest

class MockFineTuner:
    def _get_sample(self, dataset, num_samples: int):
        """Safely get a sample from dataset (handles both HF Dataset and list)"""
        if hasattr(dataset, "select"):
            # HuggingFace Dataset
            return dataset.select(range(min(num_samples, len(dataset))))
        else:
            # List or other iterable
            return dataset[:num_samples]

class TestGetSample(unittest.TestCase):
    def setUp(self):
        self.tuner = MockFineTuner()
        
    def test_hf_dataset(self):
        data = {"input": [f"in{i}" for i in range(20)], "output": [f"out{i}" for i in range(20)]}
        ds = Dataset.from_dict(data)
        
        sample = self.tuner._get_sample(ds, 10)
        self.assertEqual(len(sample), 10)
        self.assertTrue(hasattr(sample, "select")) # Should still be a dataset
        self.assertEqual(sample[0]["input"], "in0")
        
    def test_list_dataset(self):
        data = [{"input": f"in{i}", "output": f"out{i}"} for i in range(20)]
        
        sample = self.tuner._get_sample(data, 10)
        self.assertEqual(len(sample), 10)
        self.assertIsInstance(sample, list)
        self.assertEqual(sample[0]["input"], "in0")

if __name__ == "__main__":
    unittest.main()
