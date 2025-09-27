# BitNet b1.58 2B4T Testing

This project provides a testing environment for Microsoft's BitNet b1.58 2B4T model - the first open-source, native 1-bit Large Language Model at the 2-billion parameter scale.

## References

- **[Official BitNet Repository](https://github.com/microsoft/BitNet)** - Official inference framework for 1-bit LLMs
- **[Hugging Face Model](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)** - BitNet b1.58 2B4T model on Hugging Face

## Model Information

- **Model**: BitNet b1.58 2B4T
- **Parameters**: ~2 Billion
- **Training Tokens**: 4 Trillion
- **Context Length**: 4096 tokens
- **Quantization**: Native 1.58-bit weights, 8-bit activations
- **Architecture**: Transformer with BitLinear layers
- **License**: MIT

## Quick Start (First Time Setup)

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Clone and navigate to the project:**
   ```bash
   git clone https://github.com/amithegde/bitnet-test
   cd bitnet-test
   ```

2. **Install dependencies using uv:**
   ```bash
   # Create virtual environment and install all dependencies
   uv sync
   ```

3. **Run your first test:**
   ```bash
   # Basic functionality test
   uv run python test_basic.py
   
   # Comprehensive test with multiple scenarios
   uv run python test_comprehensive.py
   ```

### Windows Users

For Windows users, you can also use the provided batch script:

```bash
# Run the installation script
install.bat

# Then run tests
uv run python test_basic.py
```

## Usage

### Basic Test

Run the basic functionality test:

```bash
uv run python test_basic.py
```

### Comprehensive Test

Run the full test suite with memory monitoring and multiple scenarios:

```bash
uv run python test_comprehensive.py
```

### Manual Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_id = "microsoft/bitnet-b1.58-2B-4T"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=False
)

# Create conversation
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Hello! How are you?"},
]

# Generate response
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

print(response)
```

## Important Notes

### Performance Considerations

⚠️ **Important**: This setup uses the standard `transformers` library for testing purposes only. For production use and optimal performance, you should use the dedicated `bitnet.cpp` implementation.

The current transformers implementation:
- Does NOT provide the efficiency benefits of BitNet
- May have similar or worse performance than full-precision models
- Is intended for testing and experimentation only

### For Production Use

For optimal performance, use the official C++ implementation:
- Repository: [microsoft/BitNet](https://github.com/microsoft/BitNet)
- Provides the actual efficiency benefits (memory, energy, latency)
- Optimized for BitNet's 1.58-bit quantization

## Model Variants

Several model variants are available:

- **microsoft/bitnet-b1.58-2B-4T**: Packed 1.58-bit weights (use for deployment)
- **microsoft/bitnet-b1.58-2B-4T-bf16**: Master weights in BF16 (use for training/fine-tuning)
- **microsoft/bitnet-b1.58-2B-4T-gguf**: GGUF format (compatible with bitnet.cpp)

## Performance Benchmarks

According to the official evaluation, BitNet b1.58 2B4T shows:

- **Memory Usage**: 0.4GB (vs 2-4.8GB for comparable models)
- **CPU Latency**: 29ms (vs 41-124ms for comparable models)
- **Energy Usage**: 0.028J (vs 0.186-0.649J for comparable models)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use `device_map="auto"` or run on CPU
2. **Slow inference**: Expected with transformers - use bitnet.cpp for speed
3. **Installation issues**: Ensure you're using the correct transformers version

### System Requirements

- **RAM**: At least 8GB recommended (model uses ~5GB)
- **GPU**: Optional, but recommended for faster inference
- **Storage**: ~2GB for model weights
- **Python**: 3.9+ required

## Project Structure

```
bitnet-test/
├── test_basic.py           # Basic functionality test
├── test_comprehensive.py   # Comprehensive testing suite
├── pyproject.toml          # Project configuration & dependencies
├── install.bat            # Windows installation script (optional)
├── README.md              # This documentation
├── agents.md             # Testing agents documentation
└── uv.lock               # Dependency lock file
```

## References

- [Hugging Face Model Page](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- [BitNet Technical Report](https://arxiv.org/abs/2504.12285)
- [Official BitNet Repository](https://github.com/microsoft/BitNet)

## License

This project is licensed under the MIT License, same as the BitNet model.
