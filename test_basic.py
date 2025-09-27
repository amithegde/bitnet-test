#!/usr/bin/env python3
"""
Basic BitNet Test - Using standard transformers approach
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Disable compilation to avoid Windows issues
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

def main():
    print("🚀 Basic BitNet Test")
    print("=" * 30)
    
    try:
        model_id = "microsoft/bitnet-b1.58-2B-4T"
        print(f"📦 Loading model: {model_id}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("✅ Tokenizer loaded")
        
        # Try to load model with different configurations
        print("🧠 Loading model...")
        
        # First try with trust_remote_code
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            print("✅ Model loaded with trust_remote_code=True")
        except Exception as e1:
            print(f"❌ Failed with trust_remote_code=True: {str(e1)}")
            
            # Try without trust_remote_code
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    trust_remote_code=False
                )
                print("✅ Model loaded with trust_remote_code=False")
            except Exception as e2:
                print(f"❌ Failed with trust_remote_code=False: {str(e2)}")
                raise e2
        
        # Test basic text generation
        print("\n💬 Testing text generation...")
        
        # Simple prompt
        prompt = "The future of AI is"
        print(f"Prompt: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        print("🤖 Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. The model might require a specific transformers version")
        print("2. Try installing: uv add git+https://github.com/huggingface/transformers.git@main")
        print("3. Or use the official bitnet.cpp implementation")

if __name__ == "__main__":
    main()
