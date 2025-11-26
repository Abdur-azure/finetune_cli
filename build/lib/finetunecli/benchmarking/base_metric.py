class BaseMetric:
    def compute(self, preds, refs):
        raise NotImplementedError
