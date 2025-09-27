#!/usr/bin/env python3
"""
Wikipedia Content Processor for RAG System
Fetches and processes Wikipedia articles for flashcard generation
"""

import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote
import nltk
from typing import List, Dict, Tuple
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

class WikipediaProcessor:
    """Processes Wikipedia articles for RAG system"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BitNet-RAG-System/1.0 (Educational Purpose)'
        })
        
    def extract_wikipedia_title_from_url(self, url: str) -> str:
        """Extract Wikipedia article title from URL"""
        try:
            # Handle different Wikipedia URL formats
            if 'wikipedia.org' in url:
                # Extract title from URL path
                parsed = urlparse(url)
                path_parts = parsed.path.split('/')
                
                # Find the title part (usually after /wiki/)
                if 'wiki' in path_parts:
                    wiki_index = path_parts.index('wiki')
                    if wiki_index + 1 < len(path_parts):
                        title = path_parts[wiki_index + 1]
                        # Decode URL encoding
                        title = unquote(title)
                        return title
                        
            return None
        except Exception as e:
            print(f"Error extracting title from URL: {e}")
            return None
    
    def fetch_wikipedia_content(self, url: str) -> Dict[str, str]:
        """Fetch Wikipedia article content"""
        try:
            print(f"🌐 Fetching Wikipedia content from: {url}")
            
            # Get the article title
            title = self.extract_wikipedia_title_from_url(url)
            if not title:
                raise ValueError("Could not extract title from Wikipedia URL")
            
            print(f"📖 Article title: {title}")
            
            # Use Wikipedia API for better content extraction
            api_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + title
            response = self.session.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'title': data.get('title', title),
                    'extract': data.get('extract', ''),
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', url),
                    'description': data.get('description', ''),
                    'type': data.get('type', 'standard')
                }
            else:
                # Fallback to direct HTML scraping
                print("📄 Using HTML scraping fallback...")
                return self._scrape_wikipedia_html(url)
                
        except Exception as e:
            print(f"❌ Error fetching Wikipedia content: {e}")
            return None
    
    def _scrape_wikipedia_html(self, url: str) -> Dict[str, str]:
        """Fallback method to scrape Wikipedia HTML directly"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1', class_='firstHeading')
            title = title_elem.get_text() if title_elem else "Unknown Title"
            
            # Extract main content
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                raise ValueError("Could not find main content")
            
            # Remove unwanted elements
            for elem in content_div.find_all(['table', 'div', 'span'], class_=['navbox', 'infobox', 'reference']):
                elem.decompose()
            
            # Extract text content
            paragraphs = content_div.find_all('p')
            content_text = '\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            return {
                'title': title,
                'extract': content_text[:2000],  # Limit to first 2000 chars
                'url': url,
                'description': '',
                'type': 'scraped'
            }
            
        except Exception as e:
            print(f"❌ Error scraping HTML: {e}")
            return None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove Wikipedia-specific formatting
        text = re.sub(r'\[\d+\]', '', text)  # Remove citation numbers
        text = re.sub(r'\[edit\]', '', text)  # Remove edit links
        
        # Clean up quotes and special characters
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def chunk_text(self, text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for processing"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(0, end - 100)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
                
        return chunks
    
    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts and terms from text"""
        # Simple keyword extraction using NLTK
        sentences = nltk.sent_tokenize(text)
        
        # Find sentences with important terms (capitalized words, technical terms)
        key_concepts = []
        for sentence in sentences:
            # Look for capitalized words and technical terms
            words = sentence.split()
            for word in words:
                if (len(word) > 3 and 
                    word[0].isupper() and 
                    word.isalpha() and 
                    word not in ['The', 'This', 'That', 'There', 'These', 'Those']):
                    key_concepts.append(word)
        
        # Remove duplicates and return top concepts
        unique_concepts = list(set(key_concepts))
        return unique_concepts[:20]  # Return top 20 concepts
    
    def process_wikipedia_article(self, url: str) -> Dict:
        """Main method to process a Wikipedia article"""
        print(f"🔄 Processing Wikipedia article...")
        
        # Fetch content
        content = self.fetch_wikipedia_content(url)
        if not content:
            return None
        
        # Clean the text
        clean_extract = self.clean_text(content['extract'])
        
        # Extract key concepts
        key_concepts = self.extract_key_concepts(clean_extract)
        
        # Chunk the text
        chunks = self.chunk_text(clean_extract)
        
        result = {
            'title': content['title'],
            'url': content['url'],
            'description': content['description'],
            'full_text': clean_extract,
            'chunks': chunks,
            'key_concepts': key_concepts,
            'chunk_count': len(chunks),
            'word_count': len(clean_extract.split()),
            'processed_at': time.time()
        }
        
        print(f"✅ Processed article: {result['title']}")
        print(f"📊 {result['word_count']} words, {result['chunk_count']} chunks")
        print(f"🔑 Found {len(result['key_concepts'])} key concepts")
        
        return result

def test_wikipedia_processor():
    """Test the Wikipedia processor with a sample URL"""
    processor = WikipediaProcessor()
    
    # Test with a sample Wikipedia URL
    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    print("🧪 Testing Wikipedia Processor")
    print("=" * 40)
    
    result = processor.process_wikipedia_article(test_url)
    
    if result:
        print(f"\n📋 Results:")
        print(f"Title: {result['title']}")
        print(f"Word Count: {result['word_count']}")
        print(f"Chunks: {result['chunk_count']}")
        print(f"Key Concepts: {', '.join(result['key_concepts'][:10])}")
        print(f"\nFirst Chunk Preview:")
        print(result['chunks'][0][:200] + "...")
    else:
        print("❌ Failed to process Wikipedia article")

if __name__ == "__main__":
    test_wikipedia_processor()
