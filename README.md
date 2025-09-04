# Vectoria

A powerful personal note management system built in Rust that combines vector embeddings with intelligent search capabilities.

![FS Screenshot](https://raw.githubusercontent.com/yswa-var/Vectoria/main/Screenshot%202025-07-25%20at%209.08.40%E2%80%AFPM.png)

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

### Using the Installation Script 
```bash
brew install vectoria
vecto --help
```

```bash
git clone https://github.com/yswa-var/Vectoria.git
cd vecto
./install.sh
```

## Usage

Once installed, you can use `vecto` from anywhere in your terminal:

```bash
vecto --help

vecto --help
```

### Note Management

```bash
vecto remember "Meeting with Yash tomorrow at 2 PM for helping in his job search"

vecto list-notes

vecto forget <id>
```

### Document Processing

```bash
vecto upload <filename>

vecto list
```

### Search & Retrieval

```bash
vecto search <query>

vecto vector-search <query>

vecto rag <query>
```

## Examples

```bash
vecto remember "Important: Review quarterly reports by Friday"
vecto remember "Meeting notes: Discuss new product features with team"
vecto remember "Todo: Call client about project timeline"

vecto search "quarterly reports"
vecto vector-search "product features"
vecto rag "What meetings do I have scheduled?"

vecto upload my_document.txt
vecto list

vecto list-notes
```
