//! Basic encoding example demonstrating the memvid-rs video encoding functionality
//!
//! This example shows how to:
//! 1. Create text chunks
//! 2. Generate QR codes
//! 3. Encode them into a video file
//! 4. Save an index for later retrieval

use memvid_rs::MemvidEncoder;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("🚀 Memvid-rs Basic Encoding Example");
    println!("====================================");

    // Sample text content to encode
    let sample_texts = vec![
        "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.",
        "WebAssembly (abbreviated Wasm) is a binary instruction format for a stack-based virtual machine.",
        "Machine learning is a method of data analysis that automates analytical model building.",
        "Quantum computing harnesses the phenomena of quantum mechanics to deliver a huge leap forward in computation.",
        "Blockchain is a distributed ledger technology that maintains a continuously growing list of records.",
        "Artificial intelligence is intelligence demonstrated by machines, in contrast to human intelligence.",
        "The Internet of Things describes the network of physical objects embedded with sensors and software.",
        "Cloud computing is the on-demand availability of computer system resources without direct active management.",
        "DevOps is a set of practices that combines software development and IT operations.",
        "Cryptocurrency is a digital currency in which transactions are verified and records maintained by a decentralized system.",
    ];

    // Create output directory
    let output_dir = Path::new("output");
    std::fs::create_dir_all(output_dir)?;

    // Initialize encoder with default configuration
    println!("📝 Initializing encoder...");
    let mut encoder = MemvidEncoder::new(None).await?;

    // Add sample texts
    println!("✏️  Adding {} text samples...", sample_texts.len());
    for (i, text) in sample_texts.iter().enumerate() {
        encoder.add_text(text, 512, 32).await?;
        println!("   Added text {}: \"{}...\"", i + 1, &text[..50.min(text.len())]);
    }

    // Check stats before encoding
    let stats = encoder.get_stats();
    println!("\n📊 Pre-encoding statistics:");
    println!("   Total chunks: {}", stats.total_chunks);
    println!("   Total frames: {}", stats.total_frames);

    // Build video and index
    let video_path = output_dir.join("sample_memory.mp4");
    let index_path = output_dir.join("sample_memory.db");

    println!("\n🎬 Encoding video...");
    println!("   Video: {}", video_path.display());
    println!("   Database: {}", index_path.display());

    let encoding_stats = encoder.build_video(
        video_path.to_str().unwrap(),
        index_path.to_str().unwrap(),
    ).await?;

    // Display results
    println!("\n✅ Encoding completed successfully!");
    println!("📈 Final statistics:");
    println!("   Chunks processed: {}", encoding_stats.total_chunks);
    println!("   Frames created: {}", encoding_stats.total_frames);
    println!("   Processing time: {:.2}s", encoding_stats.processing_time);
    println!("   Video file size: {:.2} MB", encoding_stats.video_file_size as f64 / 1_048_576.0);

    // Verify files exist
    if video_path.exists() {
        let video_size = std::fs::metadata(&video_path)?.len();
        println!("   Video file: {} ({:.2} MB)", video_path.display(), video_size as f64 / 1_048_576.0);
    } else {
        println!("   ⚠️  Video file not found (encoding may have failed)");
    }

    if index_path.exists() {
        let index_size = std::fs::metadata(&index_path)?.len();
        println!("   Database file: {} ({:.2} KB)", index_path.display(), index_size as f64 / 1024.0);
    } else {
        println!("   ⚠️  Database file not found");
    }

    println!("\n🎯 Next steps:");
    println!("   1. Play the video file to see QR codes: {}", video_path.display());
    println!("   2. Use the retriever API to search the content");
    println!("   3. Implement video decoding to extract QR codes back to text");

    Ok(())
} 