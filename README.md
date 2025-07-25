# Vectoria

A Rust-based vector embedding tool that processes text files and stores embeddings in a SQLite database.

## Features

- **Text Processing**: Chunks text files into manageable pieces
- **BERT Tokenization**: Uses BERT tokenizer for advanced text processing
- **TF-IDF Embeddings**: Creates TF-IDF based embeddings
- **Database Storage**: Stores embeddings in SQLite database with metadata
- **Query Interface**: List and view stored embeddings

## Installation

1. Clone the repository
2. Install Rust dependencies:
   ```bash
   cargo build
   ```

## Usage

### Upload a file and create embeddings

```bash
cargo run -- upload <filename>
```

This will:

- Process the text file into chunks
- Create TF-IDF embeddings for each chunk
- Store embeddings in the SQLite database (`embeddings.db`)

### List stored embeddings

```bash
cargo run -- list
```

This will display all stored embeddings with:

- Database ID
- Source file name
- Chunk index
- Vocabulary size
- Embedding dimensions
- Creation timestamp
- Text preview

### Search embeddings

```bash
cargo run -- search <query>
```

This will search for embeddings containing the specified text in their content:

- Performs SQL LIKE queries on text content
- Returns matching embeddings with full metadata
- Useful for finding specific topics or content

## Database Schema

The SQLite database (`embeddings.db`) contains a table `chunk_embeddings` with:

- `id`: Primary key (auto-increment)
- `file_name`: Source file name
- `chunk_index`: Index of the chunk in the file
- `text_content`: The actual text content of the chunk
- `embedding`: JSON-serialized vector embedding
- `vocabulary_size`: Size of the vocabulary used
- `created_at`: Timestamp when the embedding was created

## Dependencies

- `clap`: Command-line argument parsing
- `regex`: Text processing
- `tokenizers`: BERT tokenization
- `rusqlite`: SQLite database operations
- `serde`: JSON serialization
- `chrono`: Timestamp handling
- `anyhow`: Error handling

## Example Output

```
File 'Ozzy_Osbourne.txt' loaded successfully
created 39 chunks with 10 lines
vocab size: 2427
Done 39 embeddings!
Successfully stored 39 embeddings in database
```

## Database Features

- **Persistent Storage**: Embeddings are stored in a local SQLite database
- **Metadata Tracking**: Each embedding includes file source, chunk index, and timestamp
- **JSON Serialization**: Vector embeddings are stored as JSON strings for flexibility
- **Query Interface**: Easy listing and inspection of stored embeddings
