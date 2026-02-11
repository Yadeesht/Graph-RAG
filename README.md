# Graph-RAG

A sophisticated Knowledge Graph-based Retrieval-Augmented Generation (RAG) system for intelligent conversation management and contextual information retrieval.

## 🎯 Overview

Graph-RAG is a Python-based system that implements knowledge graph capabilities for managing entities, relationships, and semantic search to enable context-aware information retrieval.

## 🚀 Features

### Knowledge Graph Integration
- **Entity Management**: Add, modify, and track entities with type classification
- **Relationship Mapping**: Create and query relationships between entities
- **Semantic Search**: Find similar nodes using vector embeddings
- **KuzuDB Backend**: Utilizes KuzuDB for efficient graph storage and querying
- **Visualization Support**: Built-in graph visualization capabilities

## 📋 Requirements

### Core Dependencies
- Python 3.8+
- `kuzu` - Graph database
- Custom modules:
  - `config.settings` - Configuration management
  - `rag.knowledge_graph` - Knowledge graph implementation

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Yadeesht/Graph-RAG.git
cd Graph-RAG

# Install dependencies
pip install kuzu
# Additional dependencies as per your project structure
```

## 💻 Usage

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

## 📊 How It Works

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
├── test-kg.py                # Knowledge graph testing utilities
├── config/
│   └── settings.py           # Configuration
├── rag/
│   └── knowledge_graph.py    # KG implementation
├── app_mcp/
│   └── tools/
│       └── knowledgegraph_tools.py
└── core/
    ├── agent.py              # Agent logic
    └── state.py              # State management
```


## 👤 Author

**Yadeesh**
- Third-year CSE student
- Lead Developer

## 🙏 Acknowledgments

- Built with KuzuDB for efficient graph storage
- Inspired by modern RAG architectures
- Designed for conversational AI applications
