#!/usr/bin/env python3
"""
BitNet-based Flashcard Generator for RAG System
Uses BitNet model to generate educational flashcards from Wikipedia content
"""

import os
import torch
import json
import time
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil

# Disable compilation to avoid Windows issues
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

class BitNetFlashcardGenerator:
    """Generates flashcards using BitNet model"""
    
    def __init__(self, model_id: str = "microsoft/bitnet-b1.58-2B-4T"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
    def load_model(self):
        """Load BitNet model and tokenizer"""
        if self.is_loaded:
            return
            
        print("🧠 Loading BitNet model for flashcard generation...")
        print(f"💾 Initial memory: {self._get_memory_usage():.2f} MB")
        
        try:
            # Load tokenizer
            print("🔤 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            print("✅ Tokenizer loaded")
            
            # Load model
            print("🧠 Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=False
            )
            
            print(f"✅ Model loaded")
            print(f"💾 Memory after load: {self._get_memory_usage():.2f} MB")
            self.is_loaded = True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def generate_flashcards(self, content: Dict[str, Any], num_cards: int = 10) -> List[Dict[str, str]]:
        """Generate flashcards from Wikipedia content"""
        if not self.is_loaded:
            self.load_model()
        
        print(f"🎴 Generating {num_cards} flashcards from: {content['title']}")
        
        flashcards = []
        
        # Process each chunk to generate flashcards
        for i, chunk in enumerate(content['chunks'][:3]):  # Limit to first 3 chunks for efficiency
            print(f"📝 Processing chunk {i+1}/{min(3, len(content['chunks']))}")
            
            # Create prompt for flashcard generation
            prompt = self._create_flashcard_prompt(chunk, num_cards // 3 + 1)
            
            # Generate flashcards for this chunk
            chunk_cards = self._generate_from_chunk(prompt)
            flashcards.extend(chunk_cards)
            
            if len(flashcards) >= num_cards:
                break
        
        # Limit to requested number
        flashcards = flashcards[:num_cards]
        
        print(f"✅ Generated {len(flashcards)} flashcards")
        return flashcards
    
    def _create_flashcard_prompt(self, text: str, num_cards: int) -> str:
        """Create a prompt for generating flashcards"""
        return f"""Based on the following text, create {num_cards} educational flashcards. Each flashcard should have a clear question and a concise answer.

Text: {text[:800]}

Format each flashcard as:
Q: [Question]
A: [Answer]

Generate flashcards that test understanding of key concepts, facts, and relationships in the text."""

    def _generate_from_chunk(self, prompt: str) -> List[Dict[str, str]]:
        """Generate flashcards from a text chunk"""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract flashcards from response
            flashcards = self._parse_flashcards(response)
            return flashcards
            
        except Exception as e:
            print(f"❌ Error generating flashcards: {e}")
            return []
    
    def _parse_flashcards(self, response: str) -> List[Dict[str, str]]:
        """Parse flashcards from model response"""
        flashcards = []
        
        # Split response into lines
        lines = response.split('\n')
        
        current_question = None
        current_answer = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Q:') or line.startswith('Question:'):
                # Save previous card if exists
                if current_question and current_answer:
                    flashcards.append({
                        'question': current_question.strip(),
                        'answer': current_answer.strip()
                    })
                
                # Start new question
                current_question = line[2:].strip() if line.startswith('Q:') else line[9:].strip()
                current_answer = None
                
            elif line.startswith('A:') or line.startswith('Answer:'):
                # Set answer
                current_answer = line[2:].strip() if line.startswith('A:') else line[7:].strip()
                
            elif current_question and not current_answer:
                # Continue question
                current_question += " " + line
                
            elif current_answer:
                # Continue answer
                current_answer += " " + line
        
        # Add last card if exists
        if current_question and current_answer:
            flashcards.append({
                'question': current_question.strip(),
                'answer': current_answer.strip()
            })
        
        return flashcards
    
    def generate_summary_flashcards(self, content: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate summary flashcards covering the entire article"""
        if not self.is_loaded:
            self.load_model()
        
        print(f"📋 Generating summary flashcards for: {content['title']}")
        
        # Create a comprehensive prompt
        prompt = f"""Create 5 comprehensive flashcards that summarize the key points of this Wikipedia article about {content['title']}.

Article content: {content['full_text'][:1500]}

Create flashcards that cover:
1. Main definition/concept
2. Key characteristics or features  
3. Important examples or applications
4. Historical context or significance
5. Related concepts or connections

Format each as:
Q: [Question]
A: [Answer]"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=600,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            flashcards = self._parse_flashcards(response)
            
            print(f"✅ Generated {len(flashcards)} summary flashcards")
            return flashcards
            
        except Exception as e:
            print(f"❌ Error generating summary flashcards: {e}")
            return []
    
    def save_flashcards(self, flashcards: List[Dict[str, str]], filename: str = None):
        """Save flashcards to JSON file"""
        if not filename:
            timestamp = int(time.time())
            filename = f"flashcards_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(flashcards, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Flashcards saved to: {filename}")
        return filename

def test_flashcard_generator():
    """Test the flashcard generator with sample content"""
    generator = BitNetFlashcardGenerator()
    
    # Sample content for testing
    sample_content = {
        'title': 'Artificial Intelligence',
        'full_text': 'Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. AI research is defined as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.',
        'chunks': [
            'Artificial Intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.',
            'AI research is defined as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.'
        ],
        'key_concepts': ['Artificial Intelligence', 'machines', 'intelligence', 'agents']
    }
    
    print("🧪 Testing Flashcard Generator")
    print("=" * 40)
    
    try:
        # Generate flashcards
        flashcards = generator.generate_flashcards(sample_content, num_cards=5)
        
        print(f"\n🎴 Generated {len(flashcards)} flashcards:")
        for i, card in enumerate(flashcards, 1):
            print(f"\n{i}. Q: {card['question']}")
            print(f"   A: {card['answer']}")
        
        # Save to file
        generator.save_flashcards(flashcards, "test_flashcards.json")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_flashcard_generator()
