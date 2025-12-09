# CRAG: Corrective Retrieval Augmented Generation

A sophisticated RAG (Retrieval-Augmented Generation) system that implements Corrective RAG with hybrid retrieval, reranking, and web search fallback capabilities.

## 🌟 Features

- **Hybrid Retrieval**: Combines BM25 (keyword-based) and semantic vector search using ensemble retrieval
- **Cross-Encoder Reranking**: Uses `cross-encoder/ms-marco-TinyBERT-L-2` to rerank retrieved documents for improved relevance
- **Corrective RAG (CRAG)**: Implements document grading and automatically falls back to web search when local knowledge is insufficient
- **Dual-Model Architecture**: Uses a fast model for grading and a powerful model for generation
  - **Grading**: `llama3.2:3b` for efficient relevance assessment
  - **Generation**: `gemma3:27b` for high-quality answer synthesis
- **Web Search Integration**: Uses Tavily API for real-time web search when knowledge base lacks relevant information
- **LangGraph Workflow**: Implements a stateful graph-based pipeline for complex RAG operations
- **Multilingual Support**: Handles both English and Turkish queries with multilingual embeddings
- **ChromaDB Integration**: Persistent vector storage with HTTP client support
- **Source Citation**: Automatically cites sources in the generated answers

## 📊 Architecture
```mermaid
graph TB
    Start([User Query]) --> Retrieve[Retrieve Documents<br/>Hybrid: BM25 + Vector<br/>Top 20 Candidates]
    Retrieve --> Rerank[Cross-Encoder Reranking<br/>cross-encoder/ms-marco-TinyBERT<br/>Top 5 from Top 20]
    Rerank --> Grade[Grade Documents<br/>LLM: llama3.2:3b<br/>Relevance Check]
    
    Grade --> Decision{All Docs<br/>Irrelevant?}
    
    Decision -->|Yes| WebSearch[Web Search<br/>Tavily API<br/>Max 3 Results]
    Decision -->|No| Generate[Generate Answer<br/>LLM: gemma3:27b<br/>with Citations]
    
    WebSearch --> Generate
    Generate --> End([Final Answer<br/>with Source Citations])
    
    %% Stiller
    style Start fill:#4A90E2,stroke:#333,stroke-width:2px,color:#fff
    style Retrieve fill:#E3F2FD,stroke:#4A90E2,stroke-width:2px
    style Rerank fill:#BBDEFB,stroke:#4A90E2,stroke-width:2px
    style Grade fill:#FFD54F,stroke:#FBC02D,stroke-width:2px
    style Decision fill:#FFFFFF,stroke:#333,stroke-width:4px
    style WebSearch fill:#FF7043,stroke:#E64A19,stroke-width:2px,color:#fff
    style Generate fill:#66BB6A,stroke:#2E7D32,stroke-width:2px,color:#fff
    style End fill:#2E7D32,stroke:#1B5E20,stroke-width:2px,color:#fff
```

## 🔧 System Components

### 1. Database Management (`croma_db_update.py`)
- Loads and chunks documents from JSON files (SQuAD format)
- Creates unique document IDs using SHA-256 hashing
- Manages ChromaDB collections with incremental updates
- Implements ensemble retriever combining BM25 and vector search

### 2. CRAG Pipeline (`CRAG.py`)
- **Retrieve Node**: Fetches top-20 documents using hybrid retrieval
- **Reranking**: Cross-encoder reduces to top-5 most relevant documents
- **Grade Node**: Fast LLM (llama3.2:3b) performs relevance grading for each document
- **Web Search Node**: Tavily integration for external knowledge retrieval
- **Generate Node**: Powerful LLM (gemma3:27b) produces final answer with source citations

## 🚀 Getting Started

### Prerequisites

```bash
# Required packages
pip install langchain-ollama
pip install langchain-chroma
pip install chromadb
pip install sentence-transformers
pip install langchain-huggingface
pip install langchain-community
pip install tavily-python
pip install langgraph
pip install pydantic
```

### Ollama Model Setup

```bash
# Pull required models
ollama pull gemma3:27b    # Main generation model
ollama pull llama3.2:3b   # Fast grading model
```

### Configuration

```python
# Model settings
OLLAMA_MAIN_MODEL = "gemma3:27b"      # For answer generation
OLLAMA_FAST_MODEL = "llama3.2:3b"    # For document grading
CLOUDFLARE_TUNNEL_URL = "your-tunnel-url"  # Your Ollama endpoint

# ChromaDB settings
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "rag_test_data"

# Retrieval settings
TOP_K_RETRIEVAL = 20  # Initial retrieval
TOP_K_RERANK = 5      # After reranking

# Tavily API
TAVILY_API_KEY = "your-api-key"
```

### Running the System

1. **Start ChromaDB server**:
```bash
docker run -p 8000:8000 chromadb/chroma
```

2. **Start Ollama** (if not already running):
```bash
ollama serve
# Or use Cloudflare tunnel for remote access
```

3. **Initialize database**:
```bash
python croma_db_update.py
```

4. **Run queries**:
```bash
python CRAG.py
```

## 📈 Performance Evaluation

### Test Queries and Results

| Query Type | Question | Result | Model Used | Performance |
|------------|----------|--------|------------|-------------|
| **Current Events** | Although the Denver Broncos won Super Bowl 50, who is their current head coach as of the 2024 NFL season? | ✅ **Correct** - Sean Payton (via web search) | Grading: llama3.2:3b<br/>Answer: gemma3:27b | 20.93s |
| **Current Officials** | Kathmandu established its first international relationship with Eugene, Oregon in 1975; however, who is the current mayor of Eugene, Oregon today? | ✅ **Correct** - Kaarin Knudson (via web search) | Grading: llama3.2:3b<br/>Answer: gemma3:27b | 25.09s |
| **Technical/Historical** | The kilopond is described as a non-SI unit of force, but exactly how was its definition impacted by the 2019 redefinition of the SI base units? | ⚠️ **Partial** - Admitted insufficient information | Grading: llama3.2:3b<br/>Answer: gemma3:27b | 19.34s |
| **Historical (Turkish)** | Normanların eski İskandinav dinini ve dilini bırakıp, yerel halkın dinini ve dilini benimsemesindeki temel kültürel adaptasyon süreci nasıldı? | ✅ **Excellent** - Comprehensive answer from local DB | Grading: llama3.2:3b<br/>Answer: gemma3:27b | 23.26s |
| **Physics (Turkish)** | Sürtünme gibi muhafazakar olmayan kuvvetler, neden aslında mikroskobik potansiyellerin sonuçları olarak kabul edilir? | ✅ **Good** - Detailed scientific explanation | Grading: llama3.2:3b<br/>Answer: gemma3:27b | 23.87s |
| **Sports History** | Why were the traditional Roman numerals (L) not used for Super Bowl 50? | ✅ **Excellent** - Clear answer from local DB | Grading: llama3.2:3b<br/>Answer: gemma3:27b | 17.70s |

### Key Insights

**✅ Strengths:**
- **Dual-Model Efficiency**: Fast grading model (3B params) handles filtering quickly, while powerful generation model (27B params) ensures quality answers
- **Effective Web Search Fallback**: Successfully retrieved current information for questions requiring real-time data (coaches, mayors)
- **Multilingual Capability**: Handled Turkish queries effectively using multilingual embeddings
- **Source Attribution**: Consistently cited sources in answers
- **Honest Uncertainty**: Admitted when information was insufficient rather than hallucinating

**⚠️ Areas for Improvement:**
- **Response Time**: Average 21.7 seconds per query (could be optimized with parallel processing)
- **Niche Technical Topics**: Struggled with very specific technical questions (2019 SI redefinition impact)
- **Source Quality**: Some web search results had generic source names (e.g., "story.php")

**🔍 Grading Performance:**
- **Perfect Precision**: All 5 documents correctly rejected for queries requiring current information
- **Good Recall**: 2-3 documents approved for historical/knowledge-based queries
- **Zero False Positives**: No irrelevant documents passed grading
- **Model Efficiency**: llama3.2:3b provides fast, accurate relevance assessments

**🎯 Model Selection Benefits:**
- **Cost-Effective**: Small model for grading saves compute resources
- **Quality Output**: Large model for generation ensures comprehensive, accurate answers
- **Balanced Performance**: Optimal trade-off between speed and quality

## 🛠️ Technical Details

### Dual-Model Architecture
The system uses two different LLMs for optimal performance:

1. **Grading Model** (`llama3.2:3b`):
   - Purpose: Fast relevance assessment of retrieved documents
   - Temperature: 0 (deterministic)
   - Output: Binary yes/no relevance scores
   - Benefits: Quick filtering without sacrificing accuracy

2. **Generation Model** (`gemma3:27b`):
   - Purpose: High-quality answer synthesis with citations
   - Temperature: 0 (deterministic)
   - Output: Comprehensive answers with source attribution
   - Benefits: Superior reasoning and multilingual capabilities

### Retrieval Pipeline
1. **Initial Retrieval**: Ensemble of BM25 (0.5 weight) + Vector Search (0.5 weight) → Top 20 documents
2. **Reranking**: Cross-encoder scores all 20 documents → Top 5 selected
3. **Grading**: Fast LLM (llama3.2:3b) evaluates each of top 5 for relevance → Filters irrelevant documents
4. **Fallback**: If no documents pass grading → Tavily web search (max 3 results)
5. **Generation**: Powerful LLM (gemma3:27b) synthesizes final answer with citations

### Embedding Model
- **Model**: `paraphrase-multilingual-mpnet-base-v2`
- **Dimensions**: 768
- **Languages**: 50+ languages including English and Turkish

### Document Processing
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters
- **Splitter**: RecursiveCharacterTextSplitter
- **ID Generation**: SHA-256 hash of content + index

## 📁 Project Structure

```
.
├── CRAG.py                 # Main CRAG pipeline with dual-model architecture
├── croma_db_update.py      # Database management and ensemble retriever
├── database/               # JSON data files (SQuAD format)
│   ├── dev-v1.1.json
│   ├── train-v1.1.json
│   └── squad-tr-dev-v1.0.0-excluded.json
└── README.md
```

## 🔮 Future Improvements

- [ ] Add streaming responses for better UX
- [ ] Implement caching for frequently asked questions
- [ ] Add query rewriting/expansion for better retrieval
- [ ] Implement multi-query retrieval
- [ ] Add confidence scoring for answers
- [ ] Optimize reranking batch size
- [ ] Add support for PDF and other document formats
- [ ] Implement conversation memory for follow-up questions
- [ ] Add model switching based on query complexity
- [ ] Implement async processing for parallel operations
- [ ] Add monitoring and logging dashboard

## ⚙️ Performance Optimization Tips

1. **Model Selection**: Adjust model sizes based on your hardware
   - Grading: Can use even smaller models (1B-3B params)
   - Generation: Can use larger models if available (70B+ params)

2. **Retrieval Tuning**:
   - Adjust `TOP_K_RETRIEVAL` and `TOP_K_RERANK` based on your use case
   - Higher values = better recall but slower performance

3. **Batch Processing**:
   - Process multiple queries in parallel if needed
   - Consider implementing query queue system


---

**Note**: This system requires:
- Ollama with gemma3:27b and llama3.2:3b models
- ChromaDB server (Docker or local)
- Tavily API key for web search functionality
- Sufficient RAM for running dual models (recommend 22GB+ for gemma3:27b)
