# Vectoria

A powerful personal note management system built in Rust that combines vector embeddings with intelligent search capabilities.

![Vectoria Screenshot](Screenshot%202025-07-25%20at%209.08.40%20PM.png)

Vectoria transforms how you manage and retrieve your personal notes by leveraging advanced AI techniques. Store your thoughts, documents, and ideas, then find them instantly using semantic search powered by BERT embeddings.

## Features

- **Smart Note Storage**: Store and organize your personal notes with timestamps
- **Semantic Search**: Find notes using natural language queries, not just keywords
- **Document Processing**: Upload and process text files with intelligent chunking
- **BERT-Powered**: Uses advanced BERT tokenization for superior text understanding
- **Vector Embeddings**: TF-IDF based embeddings for accurate semantic matching
- **RAG Capabilities**: Retrieve and generate answers from your stored knowledge
- **SQLite Database**: Fast, reliable local storage for your personal data
- **Memory System**: Never lose important thoughts with persistent note storage

## Installation

```bash
cargo build
```

## Usage

### Note Management

```bash
# Store a new note
cargo run -- remember "Meeting with Yash tomorrow at 2 PM for helping in his job search"

# List all your notes
cargo run -- list-notes

# Remove a note by ID
cargo run -- forget <id>
```

### Document Processing

```bash
# Upload and process a text file
cargo run -- upload <filename>

# List all processed documents
cargo run -- list
```

### Search & Retrieval

```bash
cargo run -- search <query>
cargo run -- vector-search <query>
cargo run -- rag <query>
```

## Database Schema

- `chunk_embeddings`: Stores text chunks and their vector embeddings
- `user_notes`: Stores user notes with timestamps

## Dependencies

- `clap`: Command-line argument parsing
- `regex`: Text processing
- `tokenizers`: BERT tokenization
- `rusqlite`: SQLite database operations
- `serde`: JSON serialization
- `chrono`: Timestamp handling
- `anyhow`: Error handling
