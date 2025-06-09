//! memvid-rs CLI application
//!
//! Command-line interface for the memvid-rs library.

use clap::{Parser, Subcommand};
use memvid_rs::{MemvidEncoder, MemvidRetriever};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "memvid-rs")]
#[command(
    about = "A high-performance QR code video encoder for text storage and semantic retrieval"
)]
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

        /// Output index file (SQLite database)
        #[arg(short, long, default_value = "memory_index.db")]
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

        /// Index file (SQLite database)
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

        /// Index file (SQLite database)
        #[arg(short, long)]
        index: PathBuf,
    },

    /// Add new content to existing knowledge base (incremental update)
    Append {
        /// Existing video file to append to
        #[arg(short, long)]
        video: PathBuf,

        /// Existing index file (SQLite database)
        #[arg(short, long)]
        index: PathBuf,

        /// New input file(s) to add
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
    },

    /// Store LLM conversation history to knowledge base
    AppendConversation {
        /// Existing video file to append to
        #[arg(short, long)]
        video: PathBuf,

        /// Existing index file (SQLite database)
        #[arg(short, long)]
        index: PathBuf,

        /// Conversation history file (JSON format: [{"human": "...", "assistant": "..."}])
        #[arg(short, long)]
        conversation_file: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Encode {
            inputs,
            output,
            index,
            chunk_size: _,
            overlap: _,
        } => {
            encode_command(inputs, output, index).await?;
        }
        Commands::Search {
            video,
            index,
            query,
            top_k,
        } => {
            search_command(video, index, query, top_k).await?;
        }
        Commands::Chat { video, index } => {
            chat_command(video, index).await?;
        }
        Commands::Append {
            video,
            index,
            inputs,
        } => {
            append_command(video, index, inputs).await?;
        }
        Commands::AppendConversation {
            video,
            index,
            conversation_file,
        } => {
            append_conversation_command(video, index, conversation_file).await?;
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

    let stats = encoder
        .build_video(output.to_str().unwrap(), index.to_str().unwrap())
        .await?;

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

    let mut retriever = MemvidRetriever::new(&video, &index).await?;
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

async fn chat_command(video: PathBuf, index: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("💬 Starting interactive chat mode...");
    println!("   Type 'quit' or 'exit' to end the session");
    println!();

    let mut retriever = MemvidRetriever::new(&video, &index).await?;

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

async fn append_command(
    video: PathBuf,
    index: PathBuf,
    inputs: Vec<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎬 Starting incremental update...");

    // Verify existing files exist
    if !video.exists() {
        eprintln!("❌ Existing video file not found: {}", video.display());
        return Ok(());
    }
    if !index.exists() {
        eprintln!("❌ Existing index file not found: {}", index.display());
        return Ok(());
    }

    let mut encoder = MemvidEncoder::new(None).await?;
    let mut total_added_chunks = 0;
    let mut total_processing_time = 0.0;

    for input in inputs {
        println!("📄 Processing: {}", input.display());

        if !input.exists() {
            eprintln!("❌ File not found: {}", input.display());
            continue;
        }

        let start_time = std::time::Instant::now();

        // Use append_document_chunks for each file - this handles the incremental update correctly
        let stats = match encoder
            .append_document_chunks(
                video.to_str().unwrap(),
                index.to_str().unwrap(),
                input.to_str().unwrap(),
            )
            .await
        {
            Ok(stats) => stats,
            Err(e) => {
                eprintln!("❌ Failed to process {}: {}", input.display(), e);
                continue;
            }
        };

        total_added_chunks += stats.total_chunks;
        total_processing_time += stats.processing_time;

        println!(
            "   ✅ Added {} chunks from {} in {:.2}s",
            stats.total_chunks,
            input.display(),
            start_time.elapsed().as_secs_f64()
        );
    }

    if total_added_chunks == 0 {
        eprintln!("❌ No content was successfully processed");
        return Ok(());
    }

    println!("✅ Incremental update complete!");
    println!("   📊 Total added chunks: {}", total_added_chunks);
    println!("   🎞️  Total added frames: {}", total_added_chunks);
    println!("   ⏱️  Total time: {:.2}s", total_processing_time);
    println!("   📹 Updated video: {}", video.display());
    println!("   📋 Updated index: {}", index.display());

    Ok(())
}

async fn append_conversation_command(
    video: PathBuf,
    index: PathBuf,
    conversation_file: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎬 Starting conversation history append...");

    // Verify existing files exist
    if !video.exists() {
        eprintln!("❌ Existing video file not found: {}", video.display());
        return Ok(());
    }
    if !index.exists() {
        eprintln!("❌ Existing index file not found: {}", index.display());
        return Ok(());
    }

    if !conversation_file.exists() {
        eprintln!(
            "❌ Conversation history file not found: {}",
            conversation_file.display()
        );
        return Ok(());
    }

    let mut encoder = MemvidEncoder::new(None).await?;

    println!("📄 Processing conversation history file...");

    // Try to parse as JSON conversation format first
    let file_content = std::fs::read_to_string(&conversation_file)?;

    // TODO: For now, parse JSON conversations
    // Expected format: [{"human": "...", "assistant": "..."}, ...]
    if let Ok(json_conversations) = serde_json::from_str::<Vec<serde_json::Value>>(&file_content) {
        let mut conversations = Vec::new();

        for conv in json_conversations {
            if let (Some(human), Some(assistant)) = (
                conv.get("human").and_then(|v| v.as_str()),
                conv.get("assistant").and_then(|v| v.as_str()),
            ) {
                conversations.push((human.to_string(), assistant.to_string()));
            }
        }

        if !conversations.is_empty() {
            let stats = encoder
                .append_conversation_history(
                    video.to_str().unwrap(),
                    index.to_str().unwrap(),
                    conversations,
                )
                .await?;

            println!("✅ Conversation history append complete!");
            println!("   💬 Conversation turns: {}", stats.total_chunks / 2);
            println!("   📊 Total chunks: {}", stats.total_chunks);
            println!("   🎞️  Total frames: {}", stats.total_frames);
            println!("   ⏱️  Time: {:.2}s", stats.processing_time);
            println!("   📹 Updated video: {}", video.display());
            println!("   📋 Updated index: {}", index.display());
        } else {
            eprintln!("❌ No valid conversations found in JSON file");
        }
    } else {
        // Fallback: treat as plain text file
        println!("📄 JSON parsing failed, treating as plain text file...");
        let stats = encoder
            .append_document_chunks(
                video.to_str().unwrap(),
                index.to_str().unwrap(),
                conversation_file.to_str().unwrap(),
            )
            .await?;

        println!("✅ Conversation history append complete!");
        println!("   📊 Chunks: {}", stats.total_chunks);
        println!("   🎞️  Frames: {}", stats.total_frames);
        println!("   ⏱️  Time: {:.2}s", stats.processing_time);
        println!("   📹 Updated video: {}", video.display());
        println!("   📋 Updated index: {}", index.display());
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
