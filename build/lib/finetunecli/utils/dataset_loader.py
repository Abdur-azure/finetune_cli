from datasets import load_dataset

def load_json_dataset(path):
    ds = load_dataset("json", data_files=path)
    return ds["train"]
