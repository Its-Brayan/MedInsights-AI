# MedInsights-AI

A Retrieval-Augmented Generation (RAG) assistant for medical document analysis and Q&A using Google Gemini AI and ChromaDB vector database.

## Overview

MedInsights-AI is an intelligent system that processes medical documents (PDFs) and enables natural language question-answering based on the content. It combines:
- **LLM**: Google Gemini 3 Flash Preview for intelligent responses
- **Vector Database**: ChromaDB for efficient document retrieval
- **Embeddings**: HuggingFace Transformers for semantic understanding
- **Framework**: LangChain for orchestration

## Features

- ✅ PDF document loading and processing
- ✅ Semantic document chunking and embedding
- ✅ Vector-based similarity search
- ✅ Context-aware question answering
- ✅ Medical-focused prompt engineering
- ✅ Persistent vector database storage
- ✅ GPU acceleration support (CUDA/MPS)

## Project Structure

```
MedInsights-AI/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env                      # Environment configuration (create this)
├── config/
│   ├── config.yaml          # LLM and application settings
│   └── prompt_config.yaml   # Prompt templates and instructions
├── data/                     # Medical documents (PDFs)
├── src/
│   ├── app.py               # Main RAG assistant application
│   ├── config_loader.py     # Configuration file loader
│   ├── vectordb.py          # Vector database management
│   └── chroma_db/           # Persistent vector database
└── requirements.txt          # Project dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda
- Google Gemini API key

### Setup Steps

1. **Clone/Setup the project**
   ```bash
   cd MedInsights-AI
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   COLLECTION_NAME=medical_documents
   ```

   - `GOOGLE_API_KEY`: Your Google Gemini API key (get from [Google AI Studio](https://aistudio.google.com/))
   - `EMBEDDING_MODEL`: HuggingFace model for document embeddings (default: all-MiniLM-L6-v2)
   - `COLLECTION_NAME`: Name of ChromaDB collection for storing documents

## Configuration

### `config/config.yaml`
Specifies the LLM model to use:
```yaml
llm: gemini-3-flash-preview
```

### `config/prompt_config.yaml`
Contains prompt templates and instructions for the medical Q&A system:
- System role and purpose
- Output constraints
- Format specifications

## Usage

### Running the RAG Assistant

```bash
cd src
python app.py
```

The application will:
1. Initialize the RAG assistant
2. Load documents from the `data/` folder
3. Process and embed documents using HuggingFace embeddings
4. Launch an interactive Q&A interface

### Example Interaction

```
Enter a question or 'quit' to exit: What are the symptoms of diabetes?
[Assistant responds based on medical documents in the knowledge base]

Enter a question or 'quit' to exit: quit
```

## How It Works

### Document Processing Pipeline

1. **Loading**: PDF documents from `data/` folder using PyMuPDF
2. **Chunking**: Documents split into semantic chunks (10,000 tokens with 400 token overlap)
3. **Embedding**: Each chunk embedded using HuggingFace model
4. **Storage**: Embeddings stored in ChromaDB for fast retrieval

### Query Processing

1. **Embedding**: User question embedded using the same model
2. **Retrieval**: Top-k relevant chunks retrieved from vector database
3. **Context Assembly**: Retrieved chunks combined into context
4. **Generation**: Gemini LLM generates response using prompt template and context

## Key Components

### `app.py` - RAGAssistant Class
- **`__init__`**: Initializes LLM, vector database, and prompt templates
- **`add_documents`**: Adds documents to knowledge base
- **`invoke`**: Processes user queries and returns answers

### `vectordb.py` - VectorDB Class
- **`__init__`**: Initializes ChromaDB client and embedding model
- **`chunk_text`**: Splits documents into semantic chunks
- **`insert_documents`**: Embeds and stores documents
- **`search`**: Retrieves relevant documents for a query

### `config_loader.py`
- **`load_config`**: Loads LLM settings from config.yaml
- **`load_prompt`**: Loads prompt templates from prompt_config.yaml

## Data Preparation

1. Place medical documents (PDF format) in the `data/` folder
2. Run the application to automatically load and process them
3. Documents are embedded and stored in `src/chroma_db/`

**Note**: The application looks for PDFs in `/home/brayan/AiAgents/MediInsights_Ai/data` by default. Modify the path in `app.py` if needed.

## Dependencies

Key libraries:
- **langchain**: LLM orchestration and chain management
- **langchain-google-genai**: Google Gemini integration
- **chromadb**: Vector database
- **sentence-transformers**: Embedding models
- **PyMuPDF**: PDF loading
- **python-dotenv**: Environment variable management
- **PyYAML**: Configuration file parsing

See `requirements.txt` for complete dependency list.

## Troubleshooting

### Error: "NoneType object cannot be converted to PyString"
**Solution**: Ensure `.env` file contains:
- `GOOGLE_API_KEY`
- `EMBEDDING_MODEL`
- `COLLECTION_NAME`

### Error: No documents loaded
**Solution**: 
- Check that PDFs exist in the data folder
- Verify the data folder path in `app.py`
- Ensure PyMuPDF can read the PDF files

### Slow performance
**Solution**:
- Use a smaller embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
- Enable GPU acceleration (CUDA/MPS) - automatically detected
- Reduce chunk size or top-k results

## Performance Optimization

- **GPU Support**: Automatically uses CUDA if available, falls back to CPU
- **Embedding Model**: all-MiniLM-L6-v2 is lightweight yet effective
- **Chunk Size**: Tunable in `vectordb.py` (default: 10,000 tokens)
- **Top-k Results**: Default retrieves 3 documents per query

## API Keys & Credentials

**IMPORTANT**: Never commit `.env` file with real API keys to version control.

1. Get Google Gemini API key:
   - Visit [Google AI Studio](https://aistudio.google.com/)
   - Create a new API key
   - Copy and paste into `.env` file

## Future Enhancements

- Multi-language support
- Document metadata filtering
- Hybrid search (keyword + semantic)
- Response citation tracking
- Performance metrics and logging
- Web UI interface
- Batch document processing

## License

[Specify your license here]

## Support

For issues or questions, please check:
1. `.env` configuration is complete
2. All dependencies installed via `requirements.txt`
3. Data folder contains valid PDF files
4. Google API key is valid and has quota available

---

**Version**: 1.0.0  
**Last Updated**: 2026  
**Maintained by**:Brayan Mwangi
