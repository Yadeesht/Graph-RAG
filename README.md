# Graph-RAG

A sophisticated Retrieval-Augmented Generation (RAG) system that combines episodic memory processing with knowledge graph capabilities for intelligent conversation management and context retention.

## 🎯 Overview

Graph-RAG is a Python-based system that implements advanced memory management through two key components:

1. **Episodic RAG**: Processes and chunks conversation logs into meaningful episodes with intelligent text splitting
2. **Knowledge Graph**: Manages entities, relationships, and semantic search capabilities for contextual information retrieval

## 🚀 Features

### Episodic Memory Processing
- **Smart Text Chunking**: Automatically splits conversation logs into semantically meaningful chunks
- **Episode Detection**: Groups messages into coherent episodes based on time gaps and actors
- **Token-Aware Processing**: Respects token limits (25-1000 tokens) for optimal chunk sizes
- **Actor Tracking**: Maintains context of conversation participants (Human, Assistant, Supervisor)
- **Temporal Linking**: Links chunks chronologically with prev/next references

### Knowledge Graph Integration
- **Entity Management**: Add, modify, and track entities with type classification
- **Relationship Mapping**: Create and query relationships between entities
- **Semantic Search**: Find similar nodes using vector embeddings
- **KuzuDB Backend**: Utilizes KuzuDB for efficient graph storage and querying
- **Visualization Support**: Built-in graph visualization capabilities

## 📋 Requirements

### Core Dependencies
- Python 3.8+
- `aiosqlite` - Async SQLite operations
- `kuzu` - Graph database (implied from test file)
- Custom modules:
  - `config.settings` - Configuration management
  - `utils.helper` - Logging and token counting utilities
  - `rag.knowledge_graph` - Knowledge graph implementation

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Yadeesht/Graph-RAG.git
cd Graph-RAG

# Install dependencies
pip install aiosqlite
# Additional dependencies as per your project structure
```

## 💻 Usage

### Episodic RAG

```python
from episodic_rag import EpisodicRAG
from datetime import datetime

# Initialize with a cutoff date for processing
rag = EpisodicRAG(past_summery_date="2026-01-01T00:00:00")

# Process conversation logs into chunks
chunks = await rag.custom_text_splitters()

# Each chunk contains:
# - id: Unique identifier
# - content: Cleaned message content
# - metadata: timestamp, task_id, part, actors, prev_id, next_id
```

### Knowledge Graph

```python
from rag.knowledge_graph import KnowledgeGraph

# Initialize the knowledge graph
kg = KnowledgeGraph()

# Add entities
kg.add_entity(
    node_id="VIT Chennai",
    n_type="Organization",
    keywords="technical university research AI",
    full_desc="Leading technical university in Tamil Nadu."
)

# Add relationships
kg.add_relationship("Yadeesh", "VIT Chennai", "STUDIES_AT")

# Search for similar nodes
similar = kg.search_similar_node(["university", "AI research"])

# Visualize the graph
kg.visualize()
```

## 🔧 Configuration

Key configurable parameters in `episodic_rag.py`:

```python
MIN_MESSAGE_TOKENS = 25      # Minimum tokens per message
MIN_CHUNK_TOKENS = 200        # Minimum tokens per chunk
MAX_CHUNK_TOKENS = 1000       # Maximum tokens per chunk
MAX_TIME_GAP_SECONDS = 3600   # Maximum time gap between episodes (1 hour)
```

## 📊 How It Works

### Episodic Processing Pipeline

1. **Log Retrieval**: Fetches conversation logs from SQLite database after a specified timestamp
2. **Episode Segmentation**: Groups messages by user interactions (Human_node triggers)
3. **Message Cleaning**: Removes formatting artifacts, headers, and decorators
4. **Token Counting**: Calculates token counts for intelligent chunking
5. **Smart Chunking**: 
   - Merges small chunks within time window
   - Splits large chunks while preserving context
   - Maintains actor information throughout
6. **Chunk Linking**: Creates prev/next relationships for temporal navigation

### Knowledge Graph Features

- **Entity Extraction**: Automatically identifies entities from conversation text
- **Relation Detection**: Discovers relationships between entities
- **Similarity Search**: Uses vector embeddings to find related nodes
- **Modification Support**: Update entity properties while maintaining graph integrity
- **Persistence**: All changes are persisted to KuzuDB

## 🧪 Testing

Run the test suite using the provided test file:

```bash
python test-kg.py
```

The test file includes examples for:
- Creating and populating knowledge graphs
- Entity modification
- Similarity search
- Knowledge graph updates
- Visualization

## 🏗️ Architecture

```
Graph-RAG/
├── episodic_rag.py          # Episodic memory processing
├── test-kg.py                # Knowledge graph testing utilities
├── config/
│   └── settings.py           # Configuration (MEMORY_DB, etc.)
├── utils/
│   └── helper.py             # Logging and token counting
├── rag/
│   └── knowledge_graph.py    # KG implementation
├── app_mcp/
│   └── tools/
│       └── knowledgegraph_tools.py
└── core/
    ├── agent.py              # Agent logic
    └── state.py              # State management
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

[Add your license information here]

## 👤 Author

**Yadeesh**
- Second-year CSE student
- Lead Developer

## 🙏 Acknowledgments

- Built with KuzuDB for efficient graph storage
- Inspired by modern RAG architectures
- Designed for conversational AI applications

---

**Note**: This project is actively being developed. Some features mentioned in test files may be under construction.
