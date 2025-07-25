use anyhow::Result;
use clap::{Parser, Subcommand};
use regex::Regex;
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
            }
            Err(e) => {
                eprintln!("Error reading file '{}': {}", file, e);
            }
        },
    }

    Ok(())
}
