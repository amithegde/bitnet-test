# BitNet RAG System - Wikipedia to Flashcards

A Retrieval-Augmented Generation (RAG) system that uses Microsoft's BitNet b1.58 2B4T model to automatically generate educational flashcards from Wikipedia articles.

## 🎯 Features

- **Wikipedia Processing**: Automatically fetches and processes Wikipedia articles
- **Intelligent Chunking**: Breaks down long articles into manageable chunks
- **BitNet Integration**: Uses the efficient BitNet model for flashcard generation
- **Multiple Card Types**: Generates both content-specific and summary flashcards
- **Interactive Interface**: Command-line and interactive modes
- **Export Options**: Save flashcards to JSON format

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager
- At least 8GB RAM (model uses ~5GB)

### Installation

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Test the system:**
   ```bash
   # Test Wikipedia processing only
   uv run python test_rag_system.py wikipedia
   
   # Test flashcard generation only
   uv run python test_rag_system.py flashcards
   
   # Test full RAG system
   uv run python test_rag_system.py full
   ```

### Usage

#### Interactive Mode
```bash
uv run python rag_system.py
```

#### Command Line Mode
```bash
uv run python rag_system.py "https://en.wikipedia.org/wiki/Artificial_intelligence" 10
```

## 📋 System Architecture

```
Wikipedia URL → Wikipedia Processor → Text Chunks → BitNet Generator → Flashcards
                     ↓
              Content Analysis → Key Concepts → Summary Cards
```

### Components

1. **WikipediaProcessor** (`wikipedia_processor.py`)
   - Fetches Wikipedia articles via API and HTML scraping
   - Cleans and normalizes text content
   - Extracts key concepts and terms
   - Chunks text for optimal processing

2. **BitNetFlashcardGenerator** (`flashcard_generator.py`)
   - Loads and manages BitNet model
   - Generates flashcards from text chunks
   - Creates summary flashcards
   - Parses and formats output

3. **BitNetRAGSystem** (`rag_system.py`)
   - Main application orchestrating the workflow
   - Interactive and command-line interfaces
   - Results management and export

## 🎴 Flashcard Types

### Content Flashcards
Generated from specific text chunks, focusing on:
- Key facts and definitions
- Important concepts and relationships
- Specific examples and applications

### Summary Flashcards
Generated from the entire article, covering:
- Main definition/concept
- Key characteristics or features
- Important examples or applications
- Historical context or significance
- Related concepts or connections

## 📊 Performance

### Processing Times (CPU)
- **Model Loading**: ~30-60 seconds
- **Wikipedia Fetching**: ~2-5 seconds
- **Flashcard Generation**: ~30-120 seconds (depending on content)
- **Total Processing**: ~2-5 minutes per article

### Memory Usage
- **Initial**: ~200MB
- **After Model Load**: ~5GB
- **Peak Usage**: ~5.5GB

### Generation Rates
- **Content Cards**: ~2-5 cards per minute
- **Summary Cards**: ~1-2 cards per minute

## 🔧 Configuration

### Model Settings
```python
# In flashcard_generator.py
model_id = "microsoft/bitnet-b1.58-2B-4T"
max_chunk_size = 1000  # characters per chunk
overlap = 200          # character overlap between chunks
```

### Generation Parameters
```python
# Flashcard generation settings
max_new_tokens = 500
temperature = 0.7
top_p = 0.9
```

## 📁 File Structure

```
bitnet-test/
├── rag_system.py              # Main RAG application
├── wikipedia_processor.py     # Wikipedia content processing
├── flashcard_generator.py     # BitNet flashcard generation
├── test_rag_system.py         # Test suite
├── RAG_README.md             # This documentation
├── test_basic.py             # Original BitNet tests
├── test_comprehensive.py     # Original comprehensive tests
└── pyproject.toml           # Dependencies
```

## 🧪 Testing

### Individual Component Tests
```bash
# Test Wikipedia processing
uv run python test_rag_system.py wikipedia

# Test flashcard generation
uv run python test_rag_system.py flashcards
```

### Full System Test
```bash
# Test complete RAG pipeline
uv run python test_rag_system.py full
```

### Sample URLs for Testing
- https://en.wikipedia.org/wiki/Artificial_intelligence
- https://en.wikipedia.org/wiki/Machine_learning
- https://en.wikipedia.org/wiki/Quantum_computing
- https://en.wikipedia.org/wiki/Neural_network

## 📤 Output Format

### JSON Structure
```json
{
  "source": {
    "url": "https://en.wikipedia.org/wiki/...",
    "title": "Article Title",
    "description": "Article description",
    "word_count": 1500,
    "chunk_count": 3
  },
  "flashcards": {
    "content_cards": [
      {
        "question": "What is...?",
        "answer": "The answer is..."
      }
    ],
    "summary_cards": [...],
    "total_cards": 10
  },
  "processing": {
    "time_seconds": 120.5,
    "cards_per_minute": 5.0,
    "timestamp": 1234567890
  },
  "metadata": {
    "key_concepts": ["concept1", "concept2"],
    "chunks_processed": 3,
    "model_used": "microsoft/bitnet-b1.58-2B-4T"
  }
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Set environment variables
   export TORCH_COMPILE=0
   export TORCHDYNAMO_DISABLE=1
   ```

2. **Memory Issues**
   - Ensure at least 8GB RAM available
   - Close other applications
   - Use CPU-only mode (default)

3. **Wikipedia Access Issues**
   - Check internet connection
   - Verify URL format (must be wikipedia.org)
   - Some articles may be restricted

4. **Slow Performance**
   - Expected on CPU (use GPU for faster inference)
   - Consider using bitnet.cpp for production

### Performance Optimization

1. **For Production Use**
   - Use `bitnet.cpp` instead of transformers
   - Implement GPU acceleration
   - Add caching for repeated requests

2. **Memory Optimization**
   - Process smaller chunks
   - Limit number of flashcards
   - Use model quantization

## 🔮 Future Enhancements

- **Multiple Languages**: Support for non-English Wikipedia
- **Custom Prompts**: User-defined flashcard templates
- **Batch Processing**: Process multiple URLs at once
- **Advanced Filtering**: Filter flashcards by difficulty or topic
- **Export Formats**: CSV, Anki, Quizlet integration
- **Web Interface**: Browser-based UI
- **Caching**: Store processed articles for faster retrieval

## 📚 Educational Use Cases

- **Study Aid**: Generate flashcards for exam preparation
- **Research**: Create study materials from research topics
- **Language Learning**: Generate vocabulary cards
- **Professional Development**: Create learning materials from technical articles
- **Content Creation**: Generate educational content for courses

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License, same as the BitNet model.

## 🔗 References

- [BitNet Official Repository](https://github.com/microsoft/BitNet)
- [BitNet Model on Hugging Face](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- [Wikipedia API Documentation](https://en.wikipedia.org/api/rest_v1/)
- [BitNet Technical Report](https://arxiv.org/abs/2504.12285)
