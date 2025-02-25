# LoRA Implementation and Modifications for our model Training

## Introduction

LoRA (Low-Rank Adaptation) is a technique used to fine-tune large language models efficiently by introducing trainable low-rank matrices instead of updating all parameters. This reduces computational cost and memory usage while maintaining performance.

In this document, we describe how to integrate LoRA into an existing GPT-2 training pipeline and discuss whether it is beneficial in this context.

## Original Training Pipeline

- Uses a custom Transformer implementation with GPT-2 as the starting model for the initial weights.
- Trains using a dataset of question-answer pairs.
- Optimizes using Adam with learning rate scheduling.
- Implements early stopping and checkpoint saving.

### Key Components:
- `train_model` function for training loop.
- Transformer model initialized from GPT-2.
- Tokenization using `GPT2Tokenizer`.
- Training on CUDA.

## Integrating LoRA into our model

### 1. Install Required Dependencies

Before integrating LoRA, install the `peft` library (Parameter Efficient Fine-Tuning) which supports LoRA.

```sh
pip install transformers peft datasets evaluate
```

### 2. Load Pretrained GPT-2 with LoRA Configuration

Modify `main()` in `train.py` to wrap GPT-2 with LoRA:

```python
from peft import LoraConfig, get_peft_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model
base_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define LoRA configuration
lora_config = LoraConfig(
    task_type="CAUSAL_LM",  # GPT-2 is a causal language model
    r=8,  # Rank of LoRA adaptation
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1
)

# Apply LoRA
model = get_peft_model(base_model, lora_config)
model.to(device)
```

### 3. Update Parameter Counting and Logging

To verify LoRA efficiency, add the following function to count trainable parameters:

```python
def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = 100 * trainable_params / all_params
    return trainable_params, all_params, trainable_percentage

trainable_params, all_params, trainable_percentage = count_parameters(model)
log(f"Trainable parameters: {trainable_params} / {all_params} ({trainable_percentage:.2f}%)")
```

### 4. Modify Training Loop to Support LoRA

Since LoRA reduces the number of trainable parameters, you may adjust optimizer settings:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
```

Reduce epochs and batch size slightly to optimize performance:

```python
epochs = 25  # Reduce from 40 to 25
batch_size = 8  # Reduce from 16 to 8 due to lower memory usage
```

## Is LoRA Worth It?

### Pros:
- Reduces the number of trainable parameters significantly.
- Allows efficient fine-tuning on limited hardware (e.g., single GPU machines).
- Speeds up training while maintaining performance.

### Cons:
- If fine-tuning the full GPT-2 model is feasible, LoRA may not provide substantial gains.
- Not ideal for cases where full model adaptation is required (e.g., learning new complex tasks).

### Conclusion:
If you are constrained by GPU memory or training time, LoRA is a great option. However, if you can afford to fine-tune all weights, you might not need it.

## Final Steps

1. Save your modified `train.py` with the above changes.
2. Run training with LoRA:

```sh
python src/train.py
```

3. Monitor the logs to ensure efficiency improvements.

## Future Optimizations

- Experiment with different `r` values (e.g., `r=4`, `r=16`).
- Try larger base models (e.g., `gpt2-medium`, `gpt2-large`).
- Use mixed-precision training (`fp16`) to further reduce memory usage.