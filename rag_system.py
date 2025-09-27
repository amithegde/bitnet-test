#!/usr/bin/env python3
"""
BitNet RAG System - Wikipedia to Flashcards
Main application that combines Wikipedia processing with BitNet flashcard generation
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional
from wikipedia_processor import WikipediaProcessor
from flashcard_generator import BitNetFlashcardGenerator

class BitNetRAGSystem:
    """Main RAG system that processes Wikipedia articles and generates flashcards"""
    
    def __init__(self):
        self.wikipedia_processor = WikipediaProcessor()
        self.flashcard_generator = BitNetFlashcardGenerator()
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the RAG system"""
        if self.is_initialized:
            return
            
        print("🚀 Initializing BitNet RAG System")
        print("=" * 50)
        
        try:
            # Load BitNet model
            print("🧠 Loading BitNet model...")
            self.flashcard_generator.load_model()
            
            self.is_initialized = True
            print("✅ RAG System initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize RAG system: {e}")
            raise
    
    def process_wikipedia_to_flashcards(self, 
                                       wikipedia_url: str, 
                                       num_cards: int = 10,
                                       include_summary: bool = True) -> Dict[str, Any]:
        """Main method: Process Wikipedia URL and generate flashcards"""
        
        if not self.is_initialized:
            self.initialize()
        
        print(f"\n🔄 Processing Wikipedia URL: {wikipedia_url}")
        print(f"🎯 Target: {num_cards} flashcards")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Step 1: Process Wikipedia article
            print("📖 Step 1: Fetching and processing Wikipedia content...")
            content = self.wikipedia_processor.process_wikipedia_article(wikipedia_url)
            
            if not content:
                raise ValueError("Failed to process Wikipedia article")
            
            # Step 2: Generate flashcards
            print(f"\n🎴 Step 2: Generating {num_cards} flashcards...")
            flashcards = self.flashcard_generator.generate_flashcards(content, num_cards)
            
            # Step 3: Generate summary flashcards if requested
            summary_flashcards = []
            if include_summary:
                print(f"\n📋 Step 3: Generating summary flashcards...")
                summary_flashcards = self.flashcard_generator.generate_summary_flashcards(content)
            
            # Step 4: Compile results
            end_time = time.time()
            processing_time = end_time - start_time
            
            result = {
                'source': {
                    'url': wikipedia_url,
                    'title': content['title'],
                    'description': content['description'],
                    'word_count': content['word_count'],
                    'chunk_count': content['chunk_count']
                },
                'flashcards': {
                    'content_cards': flashcards,
                    'summary_cards': summary_flashcards,
                    'total_cards': len(flashcards) + len(summary_flashcards)
                },
                'processing': {
                    'time_seconds': round(processing_time, 2),
                    'cards_per_minute': round((len(flashcards) + len(summary_flashcards)) / (processing_time / 60), 2),
                    'timestamp': time.time()
                },
                'metadata': {
                    'key_concepts': content['key_concepts'],
                    'chunks_processed': min(3, content['chunk_count']),
                    'model_used': self.flashcard_generator.model_id
                }
            }
            
            print(f"\n✅ Processing completed in {processing_time:.2f} seconds")
            print(f"📊 Generated {result['flashcards']['total_cards']} flashcards")
            print(f"⚡ Rate: {result['processing']['cards_per_minute']} cards/minute")
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing Wikipedia article: {e}")
            return None
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save results to JSON file"""
        if not filename:
            timestamp = int(time.time())
            safe_title = "".join(c for c in results['source']['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"flashcards_{safe_title}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filename}")
        return filename
    
    def display_results(self, results: Dict[str, Any]):
        """Display results in a formatted way"""
        if not results:
            print("❌ No results to display")
            return
        
        print(f"\n📋 Flashcard Results for: {results['source']['title']}")
        print("=" * 60)
        
        # Display source information
        print(f"🌐 Source: {results['source']['url']}")
        print(f"📖 Title: {results['source']['title']}")
        print(f"📝 Description: {results['source']['description']}")
        print(f"📊 Word Count: {results['source']['word_count']}")
        print(f"⏱️  Processing Time: {results['processing']['time_seconds']}s")
        print(f"🎯 Total Cards: {results['flashcards']['total_cards']}")
        
        # Display content flashcards
        if results['flashcards']['content_cards']:
            print(f"\n🎴 Content Flashcards ({len(results['flashcards']['content_cards'])}):")
            print("-" * 40)
            for i, card in enumerate(results['flashcards']['content_cards'], 1):
                print(f"\n{i}. Q: {card['question']}")
                print(f"   A: {card['answer']}")
        
        # Display summary flashcards
        if results['flashcards']['summary_cards']:
            print(f"\n📋 Summary Flashcards ({len(results['flashcards']['summary_cards'])}):")
            print("-" * 40)
            for i, card in enumerate(results['flashcards']['summary_cards'], 1):
                print(f"\n{i}. Q: {card['question']}")
                print(f"   A: {card['answer']}")
        
        # Display key concepts
        if results['metadata']['key_concepts']:
            print(f"\n🔑 Key Concepts: {', '.join(results['metadata']['key_concepts'][:10])}")
    
    def interactive_mode(self):
        """Run the RAG system in interactive mode"""
        print("🎓 BitNet RAG System - Wikipedia to Flashcards")
        print("=" * 50)
        print("Enter Wikipedia URLs to generate flashcards!")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                # Get user input
                wikipedia_url = input("🌐 Enter Wikipedia URL: ").strip()
                
                if wikipedia_url.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not wikipedia_url:
                    print("❌ Please enter a valid Wikipedia URL")
                    continue
                
                if 'wikipedia.org' not in wikipedia_url:
                    print("❌ Please enter a valid Wikipedia URL (wikipedia.org)")
                    continue
                
                # Get number of cards
                try:
                    num_cards = int(input("🎯 Number of flashcards (default 10): ") or "10")
                    if num_cards < 1 or num_cards > 50:
                        print("❌ Please enter a number between 1 and 50")
                        continue
                except ValueError:
                    num_cards = 10
                
                # Process the URL
                print(f"\n🔄 Processing: {wikipedia_url}")
                results = self.process_wikipedia_to_flashcards(wikipedia_url, num_cards)
                
                if results:
                    # Display results
                    self.display_results(results)
                    
                    # Ask to save
                    save = input("\n💾 Save results to file? (y/n): ").lower()
                    if save in ['y', 'yes']:
                        filename = self.save_results(results)
                        print(f"✅ Saved to: {filename}")
                else:
                    print("❌ Failed to process Wikipedia article")
                
                print("\n" + "="*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again.\n")

def main():
    """Main function to run the RAG system"""
    if len(sys.argv) > 1:
        # Command line mode
        wikipedia_url = sys.argv[1]
        num_cards = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        
        print("🚀 BitNet RAG System - Command Line Mode")
        print("=" * 50)
        
        rag_system = BitNetRAGSystem()
        results = rag_system.process_wikipedia_to_flashcards(wikipedia_url, num_cards)
        
        if results:
            rag_system.display_results(results)
            filename = rag_system.save_results(results)
            print(f"\n💾 Results saved to: {filename}")
        else:
            print("❌ Failed to process Wikipedia article")
    else:
        # Interactive mode
        rag_system = BitNetRAGSystem()
        rag_system.interactive_mode()

if __name__ == "__main__":
    main()
