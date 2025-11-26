from dataclasses import dataclass

@dataclass
class BenchmarkConfig:
    model_name: str
    dataset_path: str
    max_samples: int = 100
    metric: str = "rouge"
