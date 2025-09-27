#!/usr/bin/env python3
"""
Comprehensive BitNet Test - Full testing with multiple scenarios
"""

import os
import torch
import time
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

# Disable compilation to avoid Windows issues
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_bitnet_comprehensive():
    """Comprehensive test of BitNet model"""
    print("🚀 BitNet b1.58 2B4T Comprehensive Test")
    print("=" * 50)
    
    model_id = "microsoft/bitnet-b1.58-2B-4T"
    print(f"📦 Model: {model_id}")
    print(f"💾 Initial memory: {get_memory_usage():.2f} MB")
    
    try:
        # Load tokenizer
        print("🔤 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("✅ Tokenizer loaded")
        
        # Load model
        print("🧠 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=False
        )
        print(f"✅ Model loaded")
        print(f"💾 Memory after load: {get_memory_usage():.2f} MB")
        print(f"🔧 Device: {next(model.parameters()).device}")
        print(f"📊 Dtype: {next(model.parameters()).dtype}")
        
        # Test scenarios
        test_cases = [
            {
                "name": "Basic Math",
                "prompt": "What is 15 + 27?",
                "max_tokens": 50
            },
            {
                "name": "Code Generation", 
                "prompt": "Write a Python function to calculate factorial:",
                "max_tokens": 100
            },
            {
                "name": "Creative Writing",
                "prompt": "Once upon a time, in a world where AI could think:",
                "max_tokens": 80
            },
            {
                "name": "Technical Explanation",
                "prompt": "Explain what BitNet is:",
                "max_tokens": 120
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}: {test_case['name']}")
            print("-" * 40)
            print(f"📝 Prompt: {test_case['prompt']}")
            
            # Tokenize input
            inputs = tokenizer(test_case['prompt'], return_tensors="pt")
            
            # Generate with timing
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=test_case['max_tokens'],
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            end_time = time.time()
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from response
            response = response[len(test_case['prompt']):].strip()
            
            print(f"🤖 Response: {response}")
            print(f"⏱️  Time: {end_time - start_time:.2f}s")
            print(f"📏 Length: {len(response)} chars")
        
        print(f"\n💾 Final memory: {get_memory_usage():.2f} MB")
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

def show_model_info():
    """Display model information"""
    print("\n📋 BitNet b1.58 2B4T Information")
    print("=" * 40)
    print("🏗️  Architecture: Transformer with BitLinear layers")
    print("🔢 Parameters: ~2 Billion")
    print("📚 Training Tokens: 4 Trillion")
    print("📏 Context Length: 4096 tokens")
    print("⚡ Quantization: Native 1.58-bit weights, 8-bit activations")
    print("🔤 Tokenizer: LLaMA 3 Tokenizer")
    print("📄 License: MIT")
    print("\n⚠️  Note: Using transformers for testing only.")
    print("   For production efficiency, use bitnet.cpp")

if __name__ == "__main__":
    show_model_info()
    test_bitnet_comprehensive()
