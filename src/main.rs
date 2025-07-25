use clap::{Parser, Subcommand};
use regex::Regex;
use std::fs;

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

fn process_file(file: &str, no_lines: i32) -> Vec<Vec<&str>> {
    let text = file.to_lowercase();
    let re = Regex::new(r"[^\w\s\.\,\!\?]").unwrap();
    let text = re.replace_all(&text, "");
    let lines: Vec<&str> = text.lines().collect();
    let line_chunks: Vec<Vec<&str>> = lines.chunks(no_lines).map(|slice| slice.to_vec()).collect();

    line_chunks
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Upload { file } => match fs::read_to_string(file) {
            Ok(contents) => {
                println!("✅ File '{}' loaded successfully", file);
                let line_chunks = process_file(&contents, 10);
                println!(
                    "📦 Created {} chunks of max {} words each",
                    line_chunks.len(),
                    line_chunks[0].len()
                );
            }
            Err(e) => {
                eprintln!("Error reading file '{}': {}", file, e);
            }
        },
    }
}
