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

## Quick Installation

### Option 1: Using the Installation Script (Recommended)

```bash
git clone https://github.com/yswa-var/Vectoria.git
cd vecto
./install.sh
```

### Option 2: Using Cargo (Requires Rust)

```bash
cargo install vecto
git clone https://github.com/yourusername/vecto.git
cd vecto
cargo install --path .
```

## Usage

Once installed, you can use `vecto` from anywhere in your terminal:

```bash
# Get help
vecto --help

# See all available commands
vecto --help
```

### Note Management

```bash
# Store a new note
vecto remember "Meeting with Yash tomorrow at 2 PM for helping in his job search"

# List all your notes
vecto list-notes

# Remove a note by ID
vecto forget <id>
```

### Document Processing

```bash
# Upload and process a text file
vecto upload <filename>

# List all documents
vecto list
```

### Search & Retrieval

```bash
# Search for similar content
vecto search <query>

# Perform vector similarity search
vecto vector-search <query>

# Use RAG (Retrieval-Augmented Generation)
vecto rag <query>
```

## Examples

```bash
# Store some notes
vecto remember "Important: Review quarterly reports by Friday"
vecto remember "Meeting notes: Discuss new product features with team"
vecto remember "Todo: Call client about project timeline"

# Search for notes
vecto search "quarterly reports"
vecto vector-search "product features"
vecto rag "What meetings do I have scheduled?"

# Process a document
vecto upload my_document.txt
vecto list

# View all notes
vecto list-notes
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Set this for RAG functionality (optional, uses demo mode if not set)

### Database

The application uses SQLite for local storage. The database file (`embeddings.db`) is created automatically in the current directory.

## Development

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/vecto.git
cd vecto

# Build the project
cargo build --release

# Run tests
cargo test

# Install locally
cargo install --path .
```

### Project Structure

```
vecto/
├── src/
│   └── main.rs          # Main CLI application
├── models/               # BERT model files
├── embeddings.db         # SQLite database
├── Cargo.toml           # Rust dependencies
├── install.sh           # Installation script
└── README.md           # This file
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
- `tokio`: Async runtime
- `reqwest`: HTTP client for OpenAI API

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 [Documentation](https://github.com/yourusername/vecto/wiki)
- 🐛 [Report Issues](https://github.com/yourusername/vecto/issues)
- 💬 [Discussions](https://github.com/yourusername/vecto/discussions)

## Roadmap

- [ ] Web interface
- [ ] Mobile app
- [ ] Cloud sync
- [ ] Advanced search filters
- [ ] Export/import functionality
- [ ] Plugin system
