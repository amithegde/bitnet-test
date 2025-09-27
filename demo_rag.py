#!/usr/bin/env python3
"""
BitNet RAG System Demo
Demonstrates the Wikipedia to flashcards functionality with sample URLs
"""

import time
from rag_system import BitNetRAGSystem

def demo_rag_system():
    """Run a demonstration of the RAG system"""
    print("🎓 BitNet RAG System Demo")
    print("=" * 50)
    print("This demo will process Wikipedia articles and generate flashcards")
    print("using the BitNet b1.58 2B4T model.\n")
    
    # Initialize RAG system
    print("🚀 Initializing RAG system...")
    rag_system = BitNetRAGSystem()
    
    # Demo URLs
    demo_urls = [
        {
            "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "description": "AI fundamentals and applications",
            "cards": 8
        },
        {
            "url": "https://en.wikipedia.org/wiki/Machine_learning", 
            "description": "ML algorithms and techniques",
            "cards": 6
        }
    ]
    
    print(f"📋 Demo will process {len(demo_urls)} Wikipedia articles")
    print("⏳ Note: This may take several minutes due to model loading and generation\n")
    
    all_results = []
    
    for i, demo in enumerate(demo_urls, 1):
        print(f"🔄 Demo {i}/{len(demo_urls)}: {demo['description']}")
        print(f"🌐 URL: {demo['url']}")
        print(f"🎯 Target: {demo['cards']} flashcards")
        print("-" * 50)
        
        try:
            # Process the Wikipedia article
            start_time = time.time()
            results = rag_system.process_wikipedia_to_flashcards(
                wikipedia_url=demo['url'],
                num_cards=demo['cards'],
                include_summary=True
            )
            end_time = time.time()
            
            if results:
                print(f"✅ Completed in {end_time - start_time:.2f} seconds")
                
                # Display sample flashcards
                print(f"\n🎴 Sample Flashcards from '{results['source']['title']}':")
                print("-" * 40)
                
                # Show first 3 content cards
                content_cards = results['flashcards']['content_cards'][:3]
                for j, card in enumerate(content_cards, 1):
                    print(f"\n{j}. Q: {card['question']}")
                    print(f"   A: {card['answer']}")
                
                # Show first 2 summary cards if available
                if results['flashcards']['summary_cards']:
                    print(f"\n📋 Summary Cards:")
                    summary_cards = results['flashcards']['summary_cards'][:2]
                    for j, card in enumerate(summary_cards, 1):
                        print(f"\n{j}. Q: {card['question']}")
                        print(f"   A: {card['answer']}")
                
                # Save results
                filename = f"demo_results_{i}_{int(time.time())}.json"
                rag_system.save_results(results, filename)
                all_results.append(results)
                
                print(f"\n💾 Results saved to: {filename}")
                
            else:
                print("❌ Failed to process this article")
            
            print("\n" + "="*60 + "\n")
            
        except Exception as e:
            print(f"❌ Error processing {demo['url']}: {e}")
            print("Continuing with next article...\n")
    
    # Summary
    print("📊 Demo Summary")
    print("=" * 30)
    print(f"✅ Successfully processed: {len(all_results)}/{len(demo_urls)} articles")
    
    if all_results:
        total_cards = sum(r['flashcards']['total_cards'] for r in all_results)
        total_time = sum(r['processing']['time_seconds'] for r in all_results)
        avg_rate = total_cards / (total_time / 60) if total_time > 0 else 0
        
        print(f"🎴 Total flashcards generated: {total_cards}")
        print(f"⏱️  Total processing time: {total_time:.2f} seconds")
        print(f"⚡ Average generation rate: {avg_rate:.2f} cards/minute")
        
        print(f"\n📁 Generated files:")
        for i, result in enumerate(all_results, 1):
            print(f"  {i}. {result['source']['title']} - {result['flashcards']['total_cards']} cards")
    
    print(f"\n🎉 Demo completed! Check the generated JSON files for full results.")

def quick_demo():
    """Quick demo with a single article"""
    print("⚡ Quick Demo - Single Article")
    print("=" * 40)
    
    rag_system = BitNetRAGSystem()
    
    # Use a shorter, more focused article
    test_url = "https://en.wikipedia.org/wiki/Neural_network"
    
    print(f"🌐 Processing: {test_url}")
    print("🎯 Generating 5 flashcards...")
    print("⏳ This will take about 2-3 minutes...\n")
    
    try:
        results = rag_system.process_wikipedia_to_flashcards(
            wikipedia_url=test_url,
            num_cards=5,
            include_summary=False  # Skip summary for speed
        )
        
        if results:
            print("✅ Quick demo completed!")
            rag_system.display_results(results)
            
            # Save results
            filename = rag_system.save_results(results, "quick_demo_results.json")
            print(f"\n💾 Results saved to: {filename}")
        else:
            print("❌ Quick demo failed")
            
    except Exception as e:
        print(f"❌ Quick demo error: {e}")

def main():
    """Main demo function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_demo()
    else:
        print("Choose demo mode:")
        print("1. Full demo (2 articles, ~5-10 minutes)")
        print("2. Quick demo (1 article, ~2-3 minutes)")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            demo_rag_system()
        elif choice == "2":
            quick_demo()
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
