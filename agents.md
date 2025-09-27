# BitNet Testing Agents

This document describes the available testing agents and scripts for the BitNet b1.58 2B4T model.

## Available Test Scripts

### 1. `test_basic.py`
**Purpose**: Basic functionality test for BitNet model
**Usage**: `uv run python test_basic.py`
**Features**:
- Simple text generation test
- Error handling for different loading configurations
- Basic prompt-response testing
- Troubleshooting information

### 2. `test_comprehensive.py`
**Purpose**: Comprehensive testing with multiple scenarios
**Usage**: `uv run python test_comprehensive.py`
**Features**:
- Multiple test scenarios (math, code generation, creative writing, technical explanation)
- Memory usage monitoring
- Performance timing
- Model information display
- Detailed output with metrics

## Test Scenarios

### Basic Math Test
- **Prompt**: "What is 15 + 27?"
- **Expected**: Mathematical reasoning and correct answers
- **Metrics**: Response accuracy, generation time

### Code Generation Test
- **Prompt**: "Write a Python function to calculate factorial:"
- **Expected**: Proper Python code with documentation
- **Metrics**: Code quality, completeness

### Creative Writing Test
- **Prompt**: "Once upon a time, in a world where AI could think:"
- **Expected**: Coherent creative narrative
- **Metrics**: Creativity, coherence, length

### Technical Explanation Test
- **Prompt**: "Explain what BitNet is:"
- **Expected**: Technical explanation of BitNet architecture
- **Metrics**: Technical accuracy, clarity

## Performance Metrics

### Memory Usage
- **Initial**: ~200MB
- **After Model Load**: ~5GB
- **Peak Usage**: ~5.5GB

### Generation Times (CPU)
- **Basic Math**: ~58 seconds
- **Code Generation**: ~97 seconds
- **Creative Writing**: ~73 seconds
- **Technical Explanation**: ~140 seconds

### Model Specifications
- **Parameters**: ~2 Billion
- **Training Tokens**: 4 Trillion
- **Context Length**: 4096 tokens
- **Quantization**: Native 1.58-bit weights, 8-bit activations
- **Architecture**: Transformer with BitLinear layers

## Usage Instructions

### Quick Test
```bash
uv run python test_basic.py
```

### Comprehensive Test
```bash
uv run python test_comprehensive.py
```

### Environment Setup
```bash
# Create virtual environment
uv venv

# Install dependencies
uv sync

# Run tests
uv run python test_basic.py
```

## Troubleshooting

### Common Issues
1. **Compiler Errors**: Use `TORCH_COMPILE=0` environment variable
2. **Memory Issues**: Ensure sufficient RAM (8GB+ recommended)
3. **Slow Performance**: Expected on CPU, use GPU for faster inference
4. **Model Loading**: Use `trust_remote_code=False` for compatibility

### Performance Notes
- **CPU Inference**: Slow but functional for testing
- **Memory Usage**: High due to 2B parameter model
- **Production Use**: Consider `bitnet.cpp` for optimal performance

## File Structure

```
bitnet-test/
├── test_basic.py           # Basic functionality test
├── test_comprehensive.py   # Comprehensive testing suite
├── pyproject.toml         # Project configuration
├── requirements.txt       # Dependencies list
├── install.bat           # Windows installation script
├── README.md             # Project documentation
└── agents.md             # This file
```

## Next Steps

1. **Production Deployment**: Use `bitnet.cpp` for optimized inference
2. **GPU Acceleration**: Install CUDA-compatible PyTorch
3. **Fine-tuning**: Use BF16 weights for training/fine-tuning
4. **Integration**: Integrate with your application using the test patterns

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the README.md for setup instructions
3. Refer to the official BitNet documentation
4. Use the test scripts to verify functionality
