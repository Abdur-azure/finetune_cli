from dataclasses import dataclass

@dataclass
class BaseConfig:
    model_name: str
    output_dir: str = "outputs"
    dataset_path: str = None
