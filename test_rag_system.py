#!/usr/bin/env python3
"""
Test script for BitNet RAG System
Tests the Wikipedia to flashcards functionality
"""

import sys
import os
from rag_system import BitNetRAGSystem

def test_rag_system():
    """Test the RAG system with a sample Wikipedia URL"""
    print("🧪 Testing BitNet RAG System")
    print("=" * 40)
    
    # Initialize RAG system
    rag_system = BitNetRAGSystem()
    
    # Test with a sample Wikipedia URL
    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    print(f"🌐 Testing with URL: {test_url}")
    print("⏳ This may take a few minutes for model loading and generation...")
    
    try:
        # Process the Wikipedia article
        results = rag_system.process_wikipedia_to_flashcards(
            wikipedia_url=test_url,
            num_cards=5,
            include_summary=True
        )
        
        if results:
            print("\n✅ Test completed successfully!")
            
            # Display results
            rag_system.display_results(results)
            
            # Save results
            filename = rag_system.save_results(results, "test_rag_results.json")
            print(f"\n💾 Test results saved to: {filename}")
            
            return True
        else:
            print("❌ Test failed - no results generated")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_wikipedia_processor_only():
    """Test just the Wikipedia processor"""
    print("🧪 Testing Wikipedia Processor Only")
    print("=" * 40)
    
    from wikipedia_processor import WikipediaProcessor
    
    processor = WikipediaProcessor()
    test_url = "https://en.wikipedia.org/wiki/Machine_learning"
    
    print(f"🌐 Testing Wikipedia processing: {test_url}")
    
    try:
        result = processor.process_wikipedia_article(test_url)
        
        if result:
            print("✅ Wikipedia processing successful!")
            print(f"📖 Title: {result['title']}")
            print(f"📊 Word count: {result['word_count']}")
            print(f"🔑 Key concepts: {', '.join(result['key_concepts'][:5])}")
            print(f"📝 First chunk: {result['chunks'][0][:200]}...")
            return True
        else:
            print("❌ Wikipedia processing failed")
            return False
            
    except Exception as e:
        print(f"❌ Wikipedia processing error: {e}")
        return False

def test_flashcard_generator_only():
    """Test just the flashcard generator"""
    print("🧪 Testing Flashcard Generator Only")
    print("=" * 40)
    
    from flashcard_generator import BitNetFlashcardGenerator
    
    generator = BitNetFlashcardGenerator()
    
    # Sample content
    sample_content = {
        'title': 'Machine Learning',
        'full_text': 'Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed.',
        'chunks': [
            'Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed.'
        ],
        'key_concepts': ['Machine Learning', 'artificial intelligence', 'algorithms', 'data']
    }
    
    print("🎴 Testing flashcard generation...")
    
    try:
        flashcards = generator.generate_flashcards(sample_content, num_cards=3)
        
        if flashcards:
            print("✅ Flashcard generation successful!")
            for i, card in enumerate(flashcards, 1):
                print(f"\n{i}. Q: {card['question']}")
                print(f"   A: {card['answer']}")
            return True
        else:
            print("❌ Flashcard generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Flashcard generation error: {e}")
        return False

def main():
    """Main test function"""
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "wikipedia":
            test_wikipedia_processor_only()
        elif test_type == "flashcards":
            test_flashcard_generator_only()
        elif test_type == "full":
            test_rag_system()
        else:
            print("❌ Unknown test type. Use: wikipedia, flashcards, or full")
    else:
        print("🧪 BitNet RAG System Test Suite")
        print("=" * 40)
        print("Available tests:")
        print("1. Wikipedia processor only")
        print("2. Flashcard generator only") 
        print("3. Full RAG system")
        print("\nUsage:")
        print("  python test_rag_system.py wikipedia")
        print("  python test_rag_system.py flashcards")
        print("  python test_rag_system.py full")
        
        choice = input("\nEnter test number (1-3): ").strip()
        
        if choice == "1":
            test_wikipedia_processor_only()
        elif choice == "2":
            test_flashcard_generator_only()
        elif choice == "3":
            test_rag_system()
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
