use anyhow::Result;
use clap::{Parser, Subcommand};
use regex::Regex;
use reqwest::Client;
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
    #[command(about = "Upload and process a text file for embedding")]
    Upload { file: String },
    #[command(about = "List all stored embeddings")]
    List,
    #[command(about = "Search for similar text chunks")]
    Search { query: String },
    #[command(about = "Perform vector similarity search")]
    VectorSearch { query: String },
    #[command(about = "Retrieval-Augmented Generation with context")]
    RAG { question: String },
    #[command(about = "Interactive chat using Ollama models with RAG context")]
    Chat,
    #[command(about = "Store a note in memory")]
    Remember {
        #[arg(help = "The note to remember")]
        note: String,
    },
    #[command(about = "Delete a note by ID")]
    Forget {
        #[arg(help = "ID of the note to delete")]
        id: i64,
    },
    #[command(about = "List all stored notes")]
    ListNotes,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChunkEmbedding {
    id: Option<i64>,
    file_name: String,
    chunk_index: usize,
    text_content: String,
    embedding: String,
    vocabulary_size: usize,
    created_at: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct UserNote {
    id: Option<i64>,
    note: String,
    created_at: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

// ---- Ollama API types ----
#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModelTag>,
}

#[derive(Debug, Deserialize)]
struct OllamaModelTag {
    name: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    message: OllamaMessage,
    done: bool,
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
    let mut stmt = conn.prepare("SELECT COUNT(*) FROM chunk_embeddings")?;
    let count: i64 = stmt.query_row(params![], |row| row.get(0))?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut embedder = SimpleEmbedder::new();

    let mut stmt = conn.prepare("SELECT DISTINCT vocabulary_size FROM chunk_embeddings ORDER BY vocabulary_size DESC LIMIT 1")?;
    let _vocab_size: usize = stmt.query_row(params![], |row| row.get(0))?;
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
    let query_embedding = embedder.embed_query(query);
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

    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    similarities.truncate(top_k);

    Ok(similarities)
}

fn search_user_notes(conn: &Connection, query: &str) -> Result<Vec<(UserNote, f32)>> {
    let mut stmt = conn.prepare("SELECT id, note, created_at FROM user_notes")?;
    let notes = stmt.query_map(params![], |row| {
        Ok(UserNote {
            id: Some(row.get(0)?),
            note: row.get(1)?,
            created_at: row.get(2)?,
        })
    })?;

    let mut similarities: Vec<(UserNote, f32)> = Vec::new();

    for note in notes {
        match note {
            Ok(user_note) => {
                let query_lower = query.to_lowercase();
                let note_lower = user_note.note.to_lowercase();
                let query_words: std::collections::HashSet<&str> =
                    query_lower.split_whitespace().collect();
                let note_words: std::collections::HashSet<&str> =
                    note_lower.split_whitespace().collect();

                let intersection = query_words.intersection(&note_words).count();
                let union = query_words.union(&note_words).count();

                let similarity = if union > 0 {
                    intersection as f32 / union as f32
                } else {
                    0.0
                };

                if similarity > 0.0 {
                    similarities.push((user_note, similarity));
                }
            }
            Err(e) => {
                eprintln!("Error processing note: {}", e);
            }
        }
    }

    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(similarities)
}

fn build_rag_prompt(context: &str, question: &str) -> String {
    format!(
        "Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, say so.\n\nContext:\n{}\n\nQuestion: {}\n\nAnswer:",
        context, question
    )
}

async fn query_openai(prompt: &str, api_key: &str) -> Result<String> {
    let client = Client::new();
    let url = "https://api.openai.com/v1/chat/completions";

    let request = OpenAIRequest {
        model: "gpt-3.5-turbo".to_string(),
        messages: vec![OpenAIMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
        temperature: 0.7,
        max_tokens: 500,
    };

    let response = client
        .post(url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;

    if response.status().is_success() {
        let openai_response: OpenAIResponse = response.json().await?;
        if let Some(choice) = openai_response.choices.first() {
            Ok(choice.message.content.clone())
        } else {
            Err(anyhow::anyhow!("No response from OpenAI"))
        }
    } else {
        let error_text = response.text().await?;
        Err(anyhow::anyhow!("OpenAI API error: {}", error_text))
    }
}

async fn rag_search(conn: &Connection, question: &str, api_key: &str) -> Result<String> {
    let chunk_results = vector_search(conn, question, 3)?;

    let note_results = search_user_notes(conn, question)?;

    let mut all_contexts: Vec<String> = Vec::new();

    for (chunk, _) in chunk_results {
        all_contexts.push(format!("[Document Chunk]: {}", chunk.text_content));
    }

    for (note, _) in note_results.iter().take(3) {
        all_contexts.push(format!("[User Note]: {}", note.note));
    }

    if all_contexts.is_empty() {
        return Ok("No relevant context found to answer your question.".to_string());
    }

    let context: String = all_contexts.join("\n\n");

    let prompt = build_rag_prompt(&context, question);

    if api_key == "demo_key" {
        return Ok(format!("🔍 RAG Prompt Generated:\n\n{}", prompt));
    }

    let answer = query_openai(&prompt, api_key).await?;

    Ok(answer)
}

fn build_rag_context(conn: &Connection, question: &str) -> Result<Option<String>> {
    let chunk_results = vector_search(conn, question, 3)?;
    let note_results = search_user_notes(conn, question)?;

    let mut all_contexts: Vec<String> = Vec::new();
    for (chunk, _) in chunk_results {
        all_contexts.push(format!("[Document Chunk]: {}", chunk.text_content));
    }
    for (note, _) in note_results.iter().take(3) {
        all_contexts.push(format!("[User Note]: {}", note.note));
    }

    if all_contexts.is_empty() {
        Ok(None)
    } else {
        Ok(Some(all_contexts.join("\n\n")))
    }
}

async fn list_ollama_models(client: &Client) -> Result<Vec<String>> {
    let url = "http://localhost:11434/api/tags";
    let resp = client.get(url).send().await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let txt = resp.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!(format!(
            "Failed to list Ollama models: {} - {}",
            status, txt
        )));
    }
    let tags: OllamaTagsResponse = resp.json().await?;
    Ok(tags.models.into_iter().map(|m| m.name).collect())
}

async fn ollama_chat(client: &Client, model: &str, messages: &[OllamaMessage]) -> Result<String> {
    let url = "http://localhost:11434/api/chat";
    let req = OllamaChatRequest {
        model: model.to_string(),
        messages: messages.to_vec(),
        stream: false,
    };
    let resp = client.post(url).json(&req).send().await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let txt = resp.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!(format!(
            "Ollama chat failed: {} - {}",
            status, txt
        )));
    }
    let body: OllamaChatResponse = resp.json().await?;
    Ok(body.message.content)
}

fn setup_database() -> Result<Connection> {
    let conn = Connection::open("embeddings.db")?;

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

    conn.execute(
        "CREATE TABLE IF NOT EXISTS user_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note TEXT NOT NULL,
            created_at TEXT NOT NULL
        )",
        params![],
    )?;

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
            id: None,
            file_name: file_name.to_string(),
            chunk_index: i,
            text_content: chunk.join("\n"),
            embedding: serde_json::to_string(embedding)?,
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
                    &chunk_embedding.text_content[..chunk_embedding.text_content.len().min(20)]
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

#[tokio::main]
async fn main() -> Result<()> {
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
                    println!("Vector Search Results for '{}':", query);
                    println!("Found {} most similar chunks:", results.len());
                    for (i, (chunk_embedding, similarity)) in results.iter().enumerate() {
                        println!("   Rank {} (Similarity: {:.4})", i + 1, similarity);
                        println!("   File: {}", chunk_embedding.file_name);
                        println!("   Chunk Index: {}", chunk_embedding.chunk_index);
                        println!("   Vocabulary Size: {}", chunk_embedding.vocabulary_size);
                        println!(
                            "   Embedding Dimensions: {}",
                            serde_json::from_str::<Vec<f32>>(&chunk_embedding.embedding)?.len()
                        );
                        println!("   Created: {}", chunk_embedding.created_at);
                        println!(
                            "   Text Preview: {:?}",
                            &chunk_embedding.text_content
                                [..chunk_embedding.text_content.len().min(50)]
                        );
                        println!("---");
                    }
                }
                Err(e) => {
                    eprintln!("Error connecting to database: {}", e);
                }
            }
        }
        Commands::RAG { question } => {
            let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
                eprintln!("Warning: OPENAI_API_KEY environment variable not set. Using demo mode.");
                "demo_key".to_string()
            });

            match setup_database() {
                Ok(conn) => {
                    println!("RAG Search for: '{}'", question);
                    println!("Retrieving relevant context...");

                    match rag_search(&conn, question, &api_key).await {
                        Ok(answer) => {
                            println!("Answer:");
                            println!("{}", answer);
                        }
                        Err(e) => {
                            eprintln!("Error during RAG search: {}", e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error connecting to database: {}", e);
                }
            }
        }
        Commands::Chat => {
            // Initialize HTTP client and database
            let client = Client::new();
            println!("Querying Ollama for available models on http://localhost:11434 ...");
            let models = match list_ollama_models(&client).await {
                Ok(list) if !list.is_empty() => list,
                Ok(_) => {
                    eprintln!("No models found. Pull a model with 'ollama pull llama3.1' (for example). ");
                    return Ok(());
                }
                Err(e) => {
                    eprintln!("Error listing models from Ollama: {}", e);
                    eprintln!("Ensure Ollama is running: 'ollama serve' or the macOS app is open.");
                    return Ok(());
                }
            };

            println!("Available models:");
            for (i, name) in models.iter().enumerate() {
                println!("  {}. {}", i + 1, name);
            }
            println!("Enter the number of the model to use:");

            let mut input = String::new();
            std::io::stdin().read_line(&mut input).expect("failed to read input");
            let selection: usize = match input.trim().parse::<usize>() {
                Ok(n) if n >= 1 && n <= models.len() => n - 1,
                _ => {
                    eprintln!("Invalid selection.");
                    return Ok(());
                }
            };
            let model_name = models[selection].clone();
            println!("Using model: {}", model_name);

            let conn = match setup_database() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Error connecting to database: {}", e);
                    return Ok(());
                }
            };

            println!("💬 Entering chat. Type 'exit' or 'quit' to leave.\n");

            let mut history: Vec<OllamaMessage> = Vec::new();

            loop {
                print!("You: ");
                use std::io::Write;
                std::io::stdout().flush().ok();

                let mut user_input = String::new();
                if std::io::stdin().read_line(&mut user_input).is_err() {
                    eprintln!("Failed to read input");
                    continue;
                }
                let user_input = user_input.trim();
                if user_input.is_empty() {
                    continue;
                }
                if matches!(user_input, "exit" | "quit" | ":q") {
                    println!("Exiting chat.");
                    break;
                }

                let context_opt = match build_rag_context(&conn, user_input) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("RAG context error: {}", e);
                        None
                    }
                };

                let mut messages: Vec<OllamaMessage> = Vec::new();
                if let Some(context) = context_opt {
                    let system_content = format!(
                        "You are an assistant using the provided context to answer. If the context is insufficient, say so.\n\nContext:\n{}",
                        context
                    );
                    messages.push(OllamaMessage { role: "system".to_string(), content: system_content });
                }
                messages.extend(history.clone());
                messages.push(OllamaMessage { role: "user".to_string(), content: user_input.to_string() });

                match ollama_chat(&client, &model_name, &messages).await {
                    Ok(answer) => {
                        println!("Assistant: {}\n", answer.trim());
                        history.push(OllamaMessage { role: "user".to_string(), content: user_input.to_string() });
                        history.push(OllamaMessage { role: "assistant".to_string(), content: answer });
                    }
                    Err(e) => {
                        eprintln!("Chat error: {}", e);
                    }
                }
            }
        }
        Commands::Remember { note } => match setup_database() {
            Ok(conn) => {
                let now = chrono::Utc::now().to_rfc3339();
                let result = conn.execute(
                    "INSERT INTO user_notes (note, created_at) VALUES (?1, ?2)",
                    params![note, now],
                );
                if let Err(e) = result {
                    eprintln!("Error storing note: {}", e);
                } else {
                    println!("Noted ✔️");
                }
            }
            Err(e) => {
                eprintln!("Error connecting to database: {}", e);
            }
        },
        Commands::Forget { id } => match setup_database() {
            Ok(conn) => {
                let result = conn.execute("DELETE FROM user_notes WHERE id = ?1", params![id]);
                if let Err(e) = result {
                    eprintln!("Error deleting note: {}", e);
                } else {
                    println!("Note with ID {} deleted successfully.", id);
                }
            }
            Err(e) => {
                eprintln!("Error connecting to database: {}", e);
            }
        },
        Commands::ListNotes => match setup_database() {
            Ok(conn) => {
                let mut stmt = conn.prepare(
                    "SELECT id, note, created_at FROM user_notes ORDER BY created_at DESC",
                )?;
                let notes = stmt.query_map(params![], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })?;

                let mut count = 0;
                for note in notes {
                    match note {
                        Ok((id, note, created_at)) => {
                            count += 1;
                            println!("ID: {}", id);
                            println!("   Note: {}", note);
                            println!("   Created: {}", created_at);
                            println!("---");
                        }
                        Err(e) => {
                            eprintln!("Error processing note: {}", e);
                        }
                    }
                }
                println!("Found {} notes", count);
            }
            Err(e) => {
                eprintln!("Error connecting to database: {}", e);
            }
        },
    }

    Ok(())
}
