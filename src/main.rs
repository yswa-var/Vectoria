use anyhow::Result;
use clap::{Parser, Subcommand};
use regex::Regex;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(name = "vecto")]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Upload { file: String },
    List,
    Search { query: String },
    VectorSearch { query: String },
}

#[derive(Debug, Serialize, Deserialize)]
struct ChunkEmbedding {
    id: Option<i64>,
    file_name: String,
    chunk_index: usize,
    text_content: String,
    embedding: String, // JSON string of Vec<f32>
    vocabulary_size: usize,
    created_at: String,
}

fn process_file(file: &str, no_lines: usize) -> Vec<Vec<String>> {
    let text = file.to_lowercase();
    let re = Regex::new(r"[^\w\s\.\,\!\?]").unwrap();
    let text = re.replace_all(&text, "");
    let lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
    let line_chunks: Vec<Vec<String>> =
        lines.chunks(no_lines).map(|slice| slice.to_vec()).collect();

    line_chunks
}

struct SimpleEmbedder {
    vocabulary: HashMap<String, usize>,
    idf_scores: HashMap<String, f32>,
    tokenizer: Tokenizer,
}

impl SimpleEmbedder {
    fn new() -> Self {
        let tokenizer = match Tokenizer::from_file("models/bert-base-uncased-tokenizer.json") {
            Ok(t) => {
                println!("Successfully loaded BERT tokenizer");
                t
            }
            Err(e) => {
                eprintln!(
                    "Warning: Could not load tokenizer from file: {}. Using simple word splitting as fallback",
                    e
                );
                Tokenizer::from_file("models/bert-base-uncased-tokenizer.json").unwrap_or_else(
                    |_| {
                        panic!("Failed to load tokenizer and fallback also failed");
                    },
                )
            }
        };

        Self {
            vocabulary: HashMap::new(),
            idf_scores: HashMap::new(),
            tokenizer,
        }
    }

    fn fit(&mut self, chunks: &[Vec<String>]) {
        let mut word_doc_count: HashMap<String, usize> = HashMap::new();
        let total_docs = chunks.len() as f32;

        for chunk in chunks {
            let mut chunk_words = std::collections::HashSet::new();
            for line in chunk {
                match self.tokenizer.encode(line.as_str(), true) {
                    Ok(encoding) => {
                        for token in encoding.get_tokens() {
                            let token_str = token.to_string();
                            if token_str.len() > 2
                                && !token_str.starts_with('[')
                                && !token_str.ends_with(']')
                            {
                                chunk_words.insert(token_str);
                            }
                        }
                    }
                    Err(_) => {
                        for word in line.split_whitespace() {
                            if word.len() > 2 {
                                chunk_words.insert(word.to_string());
                            }
                        }
                    }
                }
            }
            for word in chunk_words {
                *word_doc_count.entry(word).or_insert(0) += 1;
            }
        }

        for (word, doc_count) in word_doc_count {
            self.vocabulary.insert(word.clone(), self.vocabulary.len());
            let idf = (total_docs / doc_count as f32).ln();
            self.idf_scores.insert(word, idf);
        }
    }

    fn embed(&self, chunk: &[String]) -> Vec<f32> {
        let mut embedding = vec![0.0; self.vocabulary.len()];
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        let mut total_words = 0;

        for line in chunk {
            match self.tokenizer.encode(line.as_str(), true) {
                Ok(encoding) => {
                    for token in encoding.get_tokens() {
                        let token_str = token.to_string();
                        if token_str.len() > 2
                            && !token_str.starts_with('[')
                            && !token_str.ends_with(']')
                        {
                            *word_counts.entry(token_str).or_insert(0) += 1;
                            total_words += 1;
                        }
                    }
                }
                Err(_) => {
                    for word in line.split_whitespace() {
                        if word.len() > 2 {
                            *word_counts.entry(word.to_string()).or_insert(0) += 1;
                            total_words += 1;
                        }
                    }
                }
            }
        }

        for (word, count) in word_counts {
            if let Some(&vocab_idx) = self.vocabulary.get(&word) {
                if let Some(&idf) = self.idf_scores.get(&word) {
                    let tf = count as f32 / total_words as f32;
                    embedding[vocab_idx] = tf * idf;
                }
            }
        }

        embedding
    }

    fn embed_query(&self, query: &str) -> Vec<f32> {
        let mut embedding = vec![0.0; self.vocabulary.len()];
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        let mut total_words = 0;

        match self.tokenizer.encode(query, true) {
            Ok(encoding) => {
                for token in encoding.get_tokens() {
                    let token_str = token.to_string();
                    if token_str.len() > 2
                        && !token_str.starts_with('[')
                        && !token_str.ends_with(']')
                    {
                        *word_counts.entry(token_str).or_insert(0) += 1;
                        total_words += 1;
                    }
                }
            }
            Err(_) => {
                for word in query.split_whitespace() {
                    if word.len() > 2 {
                        *word_counts.entry(word.to_string()).or_insert(0) += 1;
                        total_words += 1;
                    }
                }
            }
        }

        for (word, count) in word_counts {
            if let Some(&vocab_idx) = self.vocabulary.get(&word) {
                if let Some(&idf) = self.idf_scores.get(&word) {
                    let tf = count as f32 / total_words as f32;
                    embedding[vocab_idx] = tf * idf;
                }
            }
        }

        embedding
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

fn vector_search(
    conn: &Connection,
    query: &str,
    top_k: usize,
) -> Result<Vec<(ChunkEmbedding, f32)>> {
    // Load the embedder to get vocabulary and IDF scores
    let mut embedder = SimpleEmbedder::new();

    // Get all stored embeddings to rebuild vocabulary
    let mut stmt = conn.prepare("SELECT DISTINCT vocabulary_size FROM chunk_embeddings ORDER BY vocabulary_size DESC LIMIT 1")?;
    let vocab_size: usize = stmt.query_row(params![], |row| row.get(0))?;

    // Rebuild vocabulary from stored data (simplified approach)
    // In a real system, you'd want to store the vocabulary separately
    let mut all_chunks: Vec<Vec<String>> = Vec::new();
    let mut stmt = conn.prepare("SELECT text_content FROM chunk_embeddings")?;
    let chunks = stmt.query_map(params![], |row| {
        let text: String = row.get(0)?;
        Ok(text.lines().map(|s| s.to_string()).collect::<Vec<String>>())
    })?;

    for chunk in chunks {
        if let Ok(chunk_vec) = chunk {
            all_chunks.push(chunk_vec);
        }
    }

    if !all_chunks.is_empty() {
        embedder.fit(&all_chunks);
    }

    // Embed the query
    let query_embedding = embedder.embed_query(query);

    // Get all stored embeddings and compute similarities
    let mut stmt = conn.prepare("SELECT id, file_name, chunk_index, text_content, embedding, vocabulary_size, created_at FROM chunk_embeddings")?;
    let chunk_embeddings = stmt.query_map(params![], |row| {
        Ok(ChunkEmbedding {
            id: Some(row.get(0)?),
            file_name: row.get(1)?,
            chunk_index: row.get(2)?,
            text_content: row.get(3)?,
            embedding: row.get(4)?,
            vocabulary_size: row.get(5)?,
            created_at: row.get(6)?,
        })
    })?;

    let mut similarities: Vec<(ChunkEmbedding, f32)> = Vec::new();

    for embedding in chunk_embeddings {
        match embedding {
            Ok(chunk_embedding) => {
                // Parse the stored embedding
                match serde_json::from_str::<Vec<f32>>(&chunk_embedding.embedding) {
                    Ok(stored_embedding) => {
                        let similarity = cosine_similarity(&query_embedding, &stored_embedding);
                        similarities.push((chunk_embedding, similarity));
                    }
                    Err(e) => {
                        eprintln!("Error parsing embedding: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error processing embedding: {}", e);
            }
        }
    }

    // Sort by similarity (descending) and take top_k
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    similarities.truncate(top_k);

    Ok(similarities)
}

fn setup_database() -> Result<Connection> {
    let conn = Connection::open("embeddings.db")?;

    // Create tables if they don't exist
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunk_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text_content TEXT NOT NULL,
            embedding TEXT NOT NULL,
            vocabulary_size INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )",
        params![],
    )?;
    println!("Connected to SQLite database successfully");
    Ok(conn)
}

fn store_embeddings(
    conn: &Connection,
    file_name: &str,
    line_chunks: &[Vec<String>],
    all_embeddings: &[Vec<f32>],
    vocabulary_size: usize,
) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();

    for (i, (chunk, embedding)) in line_chunks.iter().zip(all_embeddings.iter()).enumerate() {
        let chunk_embedding = ChunkEmbedding {
            id: None, // SQLite will generate the ID
            file_name: file_name.to_string(),
            chunk_index: i,
            text_content: chunk.join("\n"), // Store text content as a single string
            embedding: serde_json::to_string(embedding)?, // Serialize embedding to JSON string
            vocabulary_size,
            created_at: now.clone(),
        };

        let result = conn.execute(
            "INSERT INTO chunk_embeddings (file_name, chunk_index, text_content, embedding, vocabulary_size, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                chunk_embedding.file_name,
                chunk_embedding.chunk_index,
                chunk_embedding.text_content,
                chunk_embedding.embedding,
                chunk_embedding.vocabulary_size,
                chunk_embedding.created_at
            ],
        );

        if let Err(e) = result {
            eprintln!("Error inserting chunk embedding: {}", e);
        } else {
            println!("Stored chunk {} with ID: {}", i, result.unwrap());
        }
    }

    println!(
        "Successfully stored {} embeddings in database",
        all_embeddings.len()
    );
    Ok(())
}

fn list_embeddings(conn: &Connection) -> Result<()> {
    let mut stmt = conn.prepare("SELECT id, file_name, chunk_index, text_content, embedding, vocabulary_size, created_at FROM chunk_embeddings ORDER BY id")?;
    let chunk_embeddings = stmt.query_map(params![], |row| {
        Ok(ChunkEmbedding {
            id: Some(row.get(0)?),
            file_name: row.get(1)?,
            chunk_index: row.get(2)?,
            text_content: row.get(3)?,
            embedding: row.get(4)?,
            vocabulary_size: row.get(5)?,
            created_at: row.get(6)?,
        })
    })?;

    let mut count = 0;
    for embedding in chunk_embeddings {
        match embedding {
            Ok(chunk_embedding) => {
                count += 1;
                println!("  ID: {:?}", chunk_embedding.id);
                println!("  File: {}", chunk_embedding.file_name);
                println!("  Chunk Index: {}", chunk_embedding.chunk_index);
                println!("  Vocabulary Size: {}", chunk_embedding.vocabulary_size);
                println!(
                    "  Embedding Dimensions: {}",
                    serde_json::from_str::<Vec<f32>>(&chunk_embedding.embedding)?.len()
                );
                println!("  Created: {}", chunk_embedding.created_at);
                println!(
                    "  Text Preview: {:?}",
                    &chunk_embedding.text_content[..chunk_embedding.text_content.len().min(20)] // Preview text
                );
                println!("---");
            }
            Err(e) => {
                eprintln!("Error processing embedding: {}", e);
            }
        }
    }

    println!("Found {} stored embeddings", count);
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Upload { file } => match fs::read_to_string(file) {
            Ok(contents) => {
                println!("File '{}' loaded successfully", file);
                let line_chunks = process_file(&contents, 10);
                println!(
                    "created {} chunks with {} lines",
                    line_chunks.len(),
                    line_chunks[0].len()
                );

                let mut embedder = SimpleEmbedder::new();
                embedder.fit(&line_chunks);

                println!("vocab size: {}", embedder.vocabulary.len());

                let mut all_embeddings: Vec<Vec<f32>> = Vec::new();

                for (i, chunk) in line_chunks.iter().enumerate() {
                    let embedding = embedder.embed(chunk);
                    println!("Chunk {}: {} dimensions", i + 1, embedding.len());
                    all_embeddings.push(embedding);
                }

                println!("Done {} embeddings!", all_embeddings.len());

                if !all_embeddings.is_empty() {
                    let avg_non_zero = all_embeddings
                        .iter()
                        .map(|emb| emb.iter().filter(|&&x| x > 0.0).count())
                        .sum::<usize>() as f32
                        / all_embeddings.len() as f32;
                    println!("non-zero elements: {:.2}", avg_non_zero);
                }

                // Store embeddings in database
                match setup_database() {
                    Ok(conn) => {
                        if let Err(e) = store_embeddings(
                            &conn,
                            file,
                            &line_chunks,
                            &all_embeddings,
                            embedder.vocabulary.len(),
                        ) {
                            eprintln!("Error storing embeddings in database: {}", e);
                        }
                    }
                    Err(e) => {
                        eprintln!("Error connecting to database: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error reading file '{}': {}", file, e);
            }
        },
        Commands::List => match setup_database() {
            Ok(conn) => {
                if let Err(e) = list_embeddings(&conn) {
                    eprintln!("Error listing embeddings: {}", e);
                }
            }
            Err(e) => {
                eprintln!("Error connecting to database: {}", e);
            }
        },
        Commands::Search { query } => {
            match setup_database() {
                Ok(conn) => {
                    let results = vector_search(&conn, &query, 5)?;
                    println!("Found {} results for query '{}':", results.len(), query);
                    for (chunk_embedding, similarity) in results {
                        println!("  ID: {:?}", chunk_embedding.id);
                        println!("  File: {}", chunk_embedding.file_name);
                        println!("  Chunk Index: {}", chunk_embedding.chunk_index);
                        println!("  Vocabulary Size: {}", chunk_embedding.vocabulary_size);
                        println!(
                            "  Embedding Dimensions: {}",
                            serde_json::from_str::<Vec<f32>>(&chunk_embedding.embedding)?.len()
                        );
                        println!("  Created: {}", chunk_embedding.created_at);
                        println!(
                            "  Text Preview: {:?}",
                            &chunk_embedding.text_content
                                [..chunk_embedding.text_content.len().min(20)] // Preview text
                        );
                        println!("  Similarity: {:.4}", similarity);
                        println!("---");
                    }
                }
                Err(e) => {
                    eprintln!("Error connecting to database: {}", e);
                }
            }
        }
        Commands::VectorSearch { query } => {
            match setup_database() {
                Ok(conn) => {
                    let results = vector_search(&conn, &query, 5)?;
                    println!("🔍 Vector Search Results for '{}':", query);
                    println!("Found {} most similar chunks:", results.len());
                    for (i, (chunk_embedding, similarity)) in results.iter().enumerate() {
                        println!("  🏆 Rank {} (Similarity: {:.4})", i + 1, similarity);
                        println!("  📄 File: {}", chunk_embedding.file_name);
                        println!("  📍 Chunk Index: {}", chunk_embedding.chunk_index);
                        println!("  📊 Vocabulary Size: {}", chunk_embedding.vocabulary_size);
                        println!(
                            "  🔢 Embedding Dimensions: {}",
                            serde_json::from_str::<Vec<f32>>(&chunk_embedding.embedding)?.len()
                        );
                        println!("  📅 Created: {}", chunk_embedding.created_at);
                        println!(
                            "  📝 Text Preview: {:?}",
                            &chunk_embedding.text_content
                                [..chunk_embedding.text_content.len().min(50)] // Preview text
                        );
                        println!("---");
                    }
                }
                Err(e) => {
                    eprintln!("Error connecting to database: {}", e);
                }
            }
        }
    }

    Ok(())
}
