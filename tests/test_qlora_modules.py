import unittest
from unittest.mock import MagicMock
from finetunecli.quantization.qlora.qlora_model import get_target_modules

class MockModule:
    def __init__(self, children=None):
        self._children = children or []
    def children(self):
        return self._children

class MockModel:
    def __init__(self, modules):
        self.modules_dict = modules
    def named_modules(self):
        return self.modules_dict.items()

class TestTargetModules(unittest.TestCase):
    def test_gpt2_style(self):
        # GPT-2 has c_attn and c_proj
        modules = {
            "transformer.h.0.attn.c_attn": MockModule(),
            "transformer.h.0.attn.c_proj": MockModule(),
            "transformer.h.0.mlp.c_fc": MockModule(),
            "transformer.h.0.mlp.c_proj": MockModule(),
        }
        model = MockModel(modules)
        targets = get_target_modules(model)
        self.assertIn("c_attn", targets)
        self.assertIn("c_proj", targets)
        
    def test_llama_style(self):
        # Llama has q_proj, v_proj, etc.
        modules = {
            "model.layers.0.self_attn.q_proj": MockModule(),
            "model.layers.0.self_attn.v_proj": MockModule(),
            "model.layers.0.self_attn.k_proj": MockModule(),
            "model.layers.0.self_attn.o_proj": MockModule(),
        }
        model = MockModel(modules)
        targets = get_target_modules(model)
        self.assertIn("q_proj", targets)
        self.assertIn("v_proj", targets)

if __name__ == "__main__":
    unittest.main()
