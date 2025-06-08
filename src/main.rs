//! memvid-rs CLI application
//!
//! Command-line interface for the memvid-rs library.

use clap::{Parser, Subcommand};
use memvid_rs::{MemvidEncoder, MemvidRetriever};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "memvid-rs")]
#[command(about = "A high-performance QR code video encoder for text storage and semantic retrieval")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode documents into a QR code video
    Encode {
        /// Input file(s) to encode
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        
        /// Output video file
        #[arg(short, long, default_value = "memory.mp4")]
        output: PathBuf,
        
        /// Output index file
        #[arg(short, long, default_value = "memory_index.json")]
        index: PathBuf,
        
        /// Chunk size in characters
        #[arg(long, default_value = "1024")]
        chunk_size: usize,
        
        /// Overlap between chunks
        #[arg(long, default_value = "32")]
        overlap: usize,
    },
    
    /// Search within a QR code video
    Search {
        /// Video file to search
        #[arg(short, long)]
        video: PathBuf,
        
        /// Index file
        #[arg(short, long)]
        index: PathBuf,
        
        /// Search query
        query: String,
        
        /// Number of results to return
        #[arg(short = 'k', long, default_value = "5")]
        top_k: usize,
    },
    
    /// Interactive chat with your documents
    Chat {
        /// Video file
        #[arg(short, long)]
        video: PathBuf,
        
        /// Index file
        #[arg(short, long)]
        index: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Encode { inputs, output, index, chunk_size: _, overlap: _ } => {
            encode_command(inputs, output, index).await?;
        }
        Commands::Search { video, index, query, top_k } => {
            search_command(video, index, query, top_k).await?;
        }
        Commands::Chat { video, index } => {
            chat_command(video, index).await?;
        }
    }
    
    Ok(())
}

async fn encode_command(
    inputs: Vec<PathBuf>,
    output: PathBuf,
    index: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎬 Starting memvid encoding...");
    
    let mut encoder = MemvidEncoder::new(None).await?;
    
    for input in inputs {
        println!("📄 Processing: {}", input.display());
        
        if !input.exists() {
            eprintln!("❌ File not found: {}", input.display());
            continue;
        }
        
        match input.extension().and_then(|ext| ext.to_str()) {
            Some("pdf") => {
                encoder.add_pdf(&input).await?;
            }
            Some("txt") | Some("md") | Some("markdown") => {
                encoder.add_text_file(&input).await?;
            }
            _ => {
                // Try to read as text file
                match encoder.add_text_file(&input).await {
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("❌ Failed to process {}: {}", input.display(), e);
                        continue;
                    }
                }
            }
        }
    }
    
    if encoder.chunk_count() == 0 {
        eprintln!("❌ No content was successfully processed");
        return Ok(());
    }
    
    println!("🔧 Building video with {} chunks...", encoder.chunk_count());
    
    let stats = encoder.build_video(
        output.to_str().unwrap(),
        index.to_str().unwrap(),
    ).await?;
    
    println!("✅ Encoding complete!");
    println!("   📊 Chunks: {}", stats.total_chunks);
    println!("   🎞️  Frames: {}", stats.total_frames);
    println!("   ⏱️  Time: {:.2}s", stats.processing_time);
    println!("   📹 Video: {}", output.display());
    println!("   📋 Index: {}", index.display());
    
    Ok(())
}

async fn search_command(
    video: PathBuf,
    index: PathBuf,
    query: String,
    top_k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Searching for: \"{}\"", query);
    
    let retriever = MemvidRetriever::new(&video, &index).await?;
    let results = retriever.search(&query, top_k).await?;
    
    if results.is_empty() {
        println!("❌ No results found");
        return Ok(());
    }
    
    println!("📋 Found {} results:", results.len());
    println!();
    
    for (i, (score, text)) in results.iter().enumerate() {
        println!("{}. Score: {:.3}", i + 1, score);
        println!("   {}", text);
        println!();
    }
    
    Ok(())
}

async fn chat_command(
    video: PathBuf,
    index: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("💬 Starting interactive chat mode...");
    println!("   Type 'quit' or 'exit' to end the session");
    println!();
    
    let retriever = MemvidRetriever::new(&video, &index).await?;
    
    loop {
        print!("❓ Query: ");
        use std::io::{self, Write};
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input.is_empty() {
            continue;
        }
        
        if input == "quit" || input == "exit" {
            println!("👋 Goodbye!");
            break;
        }
        
        let results = retriever.search(input, 3).await?;
        
        if results.is_empty() {
            println!("❌ No results found for: \"{}\"", input);
        } else {
            println!("📋 Results:");
            for (i, (score, text)) in results.iter().enumerate() {
                println!("{}. (Score: {:.3}) {}", i + 1, score, text);
            }
        }
        println!();
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing() {
        // Test that CLI parsing works
        let cli = Cli::try_parse_from(&["memvid-rs", "encode", "test.txt"]);
        assert!(cli.is_ok());
    }
} 