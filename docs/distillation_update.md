# Knowledge Distillation Update

**Version:** 2.1.0  
**Date:** 2025-01-29  
**Feature:** Knowledge Distillation Support

---

## 🎯 Overview

This update adds **Knowledge Distillation** as a new training approach alongside the existing fine-tuning methods (LoRA, QLoRA, AdaLoRA). Knowledge Distillation enables model compression by transferring knowledge from a large "teacher" model to a smaller "student" model.

---

## ✨ What's New

### 1. **Dual Training Paradigm**

Users now choose between two approaches:

**Fine-Tuning** (Existing)
- Adapts pre-trained model to specific tasks
- Methods: LoRA, QLoRA, AdaLoRA
- Output: Task-specialized model (same size)

**Knowledge Distillation** (New)
- Compresses large model into smaller one
- Method: Temperature-scaled KL divergence
- Output: Smaller, faster model with retained performance

### 2. **New Step 0: Approach Selection**

Interactive prompt added before method selection:
```
STEP 0: Select Training Approach
  1. Fine-Tuning    - Adapt model to specific task
  2. Distillation   - Compress model for efficiency
```

### 3. **Distillation Workflow**

Complete end-to-end workflow:
1. **Model Selection**: Teacher (large) and Student (small)
2. **Dataset Loading**: Same flexible options as fine-tuning
3. **Distillation Config**: Temperature and alpha parameters
4. **Training**: Custom DistillationTrainer with KL loss
5. **Benchmarking**: ROUGE score evaluation
6. **Upload**: HuggingFace Hub integration

---

## 🧠 Knowledge Distillation Details

### **Algorithm**

**Objective:**  
Minimize combined loss:
```
Loss = α × CE(student, labels) + (1-α) × KD(student, teacher)
```

Where:
- **CE**: Cross-entropy loss (hard labels)
- **KD**: KL divergence with temperature scaling
- **α**: Weight balancing hard vs soft targets (0.3-0.5 typical)

**Temperature Scaling:**
```
Soft_logits = softmax(logits / T)
KD_loss = KL(student_soft || teacher_soft) × T²
```

### **Key Parameters**

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Temperature (T)** | 1.0-5.0 | 2.0 | Controls softness of probability distribution |
| **Alpha (α)** | 0.0-1.0 | 0.5 | Weight for hard label loss |
| **Learning Rate** | 1e-5 to 5e-4 | 2e-4 | Student model learning rate |

**Parameter Guidelines:**
- **Higher T (3-5)**: More knowledge transfer, better for very different sizes
- **Lower T (1-2)**: Less smoothing, better for similar-sized models
- **Higher α (0.6-0.8)**: More weight on hard labels
- **Lower α (0.2-0.4)**: More weight on teacher knowledge

---

## 🔧 Technical Implementation

### **New Components**

#### 1. DistillationTrainer Class
```python
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, ...):
        # Custom trainer with KL divergence loss
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # Combined CE + KD loss
```

**Features:**
- Inherits from HuggingFace `Trainer`
- Automatic teacher model management
- Temperature-scaled softmax
- KL divergence computation
- Mixed precision support (FP16)

#### 2. Enhanced LLMFineTuner

**New Methods:**
```python
load_teacher_model(teacher_name)        # Load teacher
train_distillation(...)                 # Distillation training
```

**Workflow Integration:**
- Seamless approach selection
- Unified dataset pipeline
- Consistent benchmarking
- Same upload mechanism

---

## 📊 Use Cases

### **When to Use Distillation**

✅ **Recommended:**
- Deploying to resource-constrained environments (mobile, edge)
- Need faster inference with acceptable accuracy trade-off
- Have access to large pre-trained teacher model
- Want smaller model size for storage/bandwidth

❌ **Not Recommended:**
- Need maximum accuracy (use fine-tuning instead)
- Teacher and student are similar sizes
- Limited training data (<1000 samples)
- Task requires specialized adaptations

### **Example Scenarios**

**Scenario 1: Mobile Deployment**
```
Teacher: GPT-2 Large (774M params)
Student: GPT-2 Small (124M params)
Result: 6x smaller, 3x faster, ~85% retained performance
```

**Scenario 2: Edge Inference**
```
Teacher: OPT-1.3B
Student: OPT-125M
Result: 10x compression, real-time inference on CPU
```

**Scenario 3: Domain Adaptation + Compression**
```
Step 1: Fine-tune GPT-2 Medium on domain data (teacher)
Step 2: Distill to GPT-2 Small (student)
Result: Specialized + compressed model
```

---

## 🚀 Usage Examples

### **Basic Distillation**

```bash
python finetune_cli.py

# Step 0: Select Approach
> 2  # Distillation

# Step 1: Models
Teacher: gpt2-medium
Student: gpt2
Output: ./distilled_model

# Step 2: Dataset
Dataset: wikitext
Samples: 5000

# Step 3: Distillation Config
Temperature: 2.0
Alpha: 0.5

# Step 4: Training
Epochs: 3
Batch size: 4
Learning rate: 2e-4
```

### **Advanced Configuration**

```python
# High knowledge transfer (soft targets prioritized)
Temperature: 4.0
Alpha: 0.3

# Balanced approach (default)
Temperature: 2.0
Alpha: 0.5

# Hard labels prioritized
Temperature: 1.5
Alpha: 0.7
```

---

## 📈 Performance Characteristics

### **Memory Usage**

| Phase | Memory Required |
|-------|----------------|
| Training | Teacher + Student (both in VRAM) |
| Inference | Student only (~6x reduction for gpt2-large→gpt2) |

**Optimization:**
- Teacher loaded in FP16 automatically
- Gradient checkpointing supported
- Batch size can be smaller than fine-tuning

### **Training Time**

Compared to fine-tuning:
- **~1.5-2x longer** (teacher forward pass overhead)
- Offset by smaller student model
- Total time depends on teacher size

### **Model Size Reduction**

| Teacher → Student | Size Reduction | Typical Performance Retention |
|-------------------|----------------|-------------------------------|
| Large → Medium | ~2.2x | 92-95% |
| Medium → Small | ~3x | 88-92% |
| Large → Small | ~6x | 82-88% |

---

## 🔄 Backward Compatibility

**Fully Compatible:**
- All existing fine-tuning methods work unchanged
- No breaking changes to API
- Step numbering adjusted (new Step 0, rest shifted)
- Default behavior: Fine-tuning (if user presses Enter on Step 0)

**Migration:**
- No changes needed for existing users
- Distillation is opt-in via Step 0 selection

---

## 🐛 Known Limitations

1. **Memory**: Requires both teacher and student in VRAM during training
2. **Teacher Size**: Very large teachers (7B+) need 24GB+ VRAM
3. **Tokenizer**: Teacher and student must use same tokenizer
4. **Task Mismatch**: Not effective if teacher wasn't trained on similar data

**Workarounds:**
- Use quantized teacher (load_in_8bit) for large models
- Train on smaller batches with gradient accumulation
- Use smaller teacher if memory constrained

---

## 📚 References

**Papers:**
1. Hinton et al. (2015) - "Distilling the Knowledge in a Neural Network"
2. Sanh et al. (2019) - "DistilBERT" (practical application)

**Implementation:**
- Custom `DistillationTrainer` based on HuggingFace Trainer
- KL divergence loss from PyTorch (`nn.KLDivLoss`)
- Temperature scaling from original paper

---

## 🔮 Future Enhancements

**Planned:**
- [ ] Multi-teacher distillation
- [ ] Progressive distillation (iterative compression)
- [ ] Task-specific distillation strategies
- [ ] Quantization-aware distillation
- [ ] Attention transfer mechanisms

**Experimental:**
- [ ] Self-distillation (same model, different training stages)
- [ ] Feature-level distillation (intermediate layers)
- [ ] Data-free distillation (synthetic data generation)

---

## 📝 Changelog

### Version 2.1.0 (2025-01-29)

**Added:**
- Knowledge Distillation training approach
- `DistillationTrainer` class with KL divergence loss
- Step 0: Approach selection (Fine-Tuning vs Distillation)
- `load_teacher_model()` method
- `train_distillation()` method
- Temperature and alpha configuration prompts
- Comprehensive distillation workflow

**Modified:**
- Step numbering (added Step 0, rest incremented)
- Main workflow to support dual paradigm
- Help text and info screens

**Documentation:**
- Added DISTILLATION_UPDATE.md
- Updated README.md with distillation section
- Added usage examples for distillation

---

## 💡 Getting Started

**Quick Start:**
```bash
# 1. Run CLI
python finetune_cli.py

# 2. Choose approach
> 2  # Distillation

# 3. Follow prompts
# - Select teacher (larger model)
# - Select student (smaller model)
# - Configure parameters
# - Train and evaluate
```

**Recommended First Try:**
```
Teacher: gpt2-medium (355M)
Student: gpt2 (124M)
Dataset: wikitext (5000 samples)
Temperature: 2.0
Alpha: 0.5
```

This configuration provides good balance of compression and performance.

---

**Questions or Issues?**  
Open an issue on GitHub: https://github.com/Abdur-azure/finetune_cli/issues