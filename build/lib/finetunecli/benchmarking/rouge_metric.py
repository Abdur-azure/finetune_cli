from evaluate import load
from finetunecli.benchmarking.base_metric import BaseMetric

class RougeMetric(BaseMetric):
    def __init__(self):
        self.metric = load("rouge")

    def compute(self, preds, refs):
        return self.metric.compute(predictions=preds, references=refs)
