//! Round-trip test example demonstrating complete memvid-rs functionality
//!
//! This example shows the full pipeline:
//! 1. Text → Chunks → QR codes → Video (encoding)
//! 2. Video → QR codes → Text (decoding)
//! 3. Verification that original text matches recovered text

use memvid_rs::{MemvidEncoder, MemvidRetriever};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("🔄 Memvid-rs Round-Trip Test");
    println!("============================");

    // Test data - various types of content
    let test_texts = vec![
        "The quick brown fox jumps over the lazy dog. This is a classic pangram used for testing.",
        "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Machine learning is revolutionizing how we process and understand data in the modern world.",
        "Blockchain technology provides a decentralized approach to data storage and verification.",
        "The Internet of Things (IoT) connects everyday devices to create smart, interconnected systems.",
        "Quantum computing promises to solve complex problems that are intractable for classical computers.",
        "Artificial intelligence is transforming industries from healthcare to autonomous vehicles.",
        "Cloud computing enables scalable, on-demand access to computing resources and services.",
        "Cybersecurity is critical for protecting digital assets and maintaining privacy in our connected world.",
    ];

    // Create output directory
    let output_dir = Path::new("output");
    std::fs::create_dir_all(output_dir)?;

    println!("\n📝 Phase 1: Encoding (Text → Video)");
    println!("=====================================");

    // Initialize encoder
    let mut encoder = MemvidEncoder::new(None).await?;

    // Add test texts
    println!("Adding {} text samples...", test_texts.len());
    for (i, text) in test_texts.iter().enumerate() {
        encoder.add_text(text, 512, 32).await?;
        println!("  ✓ Added text {}: \"{}...\"", i + 1, &text[..50.min(text.len())]);
    }

    // Build video and database
    let video_path = output_dir.join("round_trip_test.mp4");
    let database_path = output_dir.join("round_trip_test.db");

    println!("\n🎬 Encoding video...");
    let encoding_stats = encoder.build_video(
        video_path.to_str().unwrap(),
        database_path.to_str().unwrap(),
    ).await?;

    println!("✅ Encoding completed!");
    println!("   Chunks: {}", encoding_stats.total_chunks);
    println!("   Frames: {}", encoding_stats.total_frames);
    println!("   Time: {:.2}s", encoding_stats.processing_time);
    println!("   Video size: {:.2} KB", encoding_stats.video_file_size as f64 / 1024.0);

    println!("\n📖 Phase 2: Decoding (Video → Text)");
    println!("=====================================");

    // Initialize retriever
    let mut retriever = MemvidRetriever::new(&video_path, &database_path).await?;

    // Get video information
    let video_info = retriever.get_video_info().await?;
    println!("Video info:");
    println!("   Dimensions: {}x{}", video_info.width, video_info.height);
    println!("   FPS: {:.2}", video_info.fps);
    println!("   Duration: {:.2}s", video_info.duration_seconds);
    println!("   Frames: {}", video_info.frame_count);
    println!("   Codec: {}", video_info.codec);

    // Get database statistics
    let db_stats = retriever.get_stats()?;
    println!("\nDatabase stats:");
    println!("   Total chunks: {}", db_stats.total_chunks);
    println!("   Total frames: {}", db_stats.total_frames);
    println!("   Database size: {:.2} KB", db_stats.database_size_bytes as f64 / 1024.0);

    println!("\n🔍 Phase 3: Content Verification");
    println!("=================================");

    // Test 1: Retrieve chunks by ID
    println!("Testing chunk retrieval by ID...");
    let mut recovered_texts = Vec::new();
    
    for chunk_id in 0..test_texts.len() {
        if let Some(chunk_text) = retriever.get_chunk_by_id(chunk_id).await? {
            recovered_texts.push(chunk_text);
            println!("  ✓ Chunk {}: \"{}...\"", chunk_id, &recovered_texts[chunk_id][..50.min(recovered_texts[chunk_id].len())]);
        } else {
            println!("  ✗ Chunk {} not found", chunk_id);
        }
    }

    // Test 2: Direct frame decoding
    println!("\nTesting direct frame decoding...");
    let mut frame_texts = Vec::new();
    
    for frame_num in 0..test_texts.len() as u32 {
        match retriever.decode_frame(frame_num).await {
            Ok(frame_text) => {
                frame_texts.push(frame_text);
                println!("  ✓ Frame {}: \"{}...\"", frame_num, &frame_texts[frame_num as usize][..50.min(frame_texts[frame_num as usize].len())]);
            }
            Err(e) => {
                println!("  ✗ Frame {} decode failed: {}", frame_num, e);
            }
        }
    }

    // Test 3: Search functionality
    println!("\nTesting search functionality...");
    let search_queries = ["Rust", "machine learning", "blockchain"];
    
    for query in &search_queries {
        let results = retriever.search(query, 3).await?;
        println!("  Search '{}': {} results", query, results.len());
        for (i, (score, text)) in results.iter().enumerate() {
            println!("    {}. Score: {:.2}, Text: \"{}...\"", i + 1, score, &text[..50.min(text.len())]);
        }
    }

    println!("\n✅ Phase 4: Round-Trip Validation");
    println!("==================================");

    // Verify that original texts match recovered texts
    let mut success_count = 0;
    let mut total_tests = 0;

    // Test database retrieval
    println!("Database retrieval validation:");
    for (i, original) in test_texts.iter().enumerate() {
        total_tests += 1;
        if i < recovered_texts.len() && recovered_texts[i] == *original {
            success_count += 1;
            println!("  ✓ Chunk {}: MATCH", i);
        } else {
            println!("  ✗ Chunk {}: MISMATCH", i);
            if i < recovered_texts.len() {
                println!("    Original: \"{}\"", original);
                println!("    Recovered: \"{}\"", recovered_texts[i]);
            }
        }
    }

    // Test frame decoding
    println!("\nFrame decoding validation:");
    for (i, original) in test_texts.iter().enumerate() {
        total_tests += 1;
        if i < frame_texts.len() && frame_texts[i] == *original {
            success_count += 1;
            println!("  ✓ Frame {}: MATCH", i);
        } else {
            println!("  ✗ Frame {}: MISMATCH", i);
            if i < frame_texts.len() {
                println!("    Original: \"{}\"", original);
                println!("    Recovered: \"{}\"", frame_texts[i]);
            }
        }
    }

    // Final results
    let success_rate = (success_count as f64 / total_tests as f64) * 100.0;
    println!("\n🎯 Final Results:");
    println!("   Success rate: {:.1}% ({}/{} tests passed)", success_rate, success_count, total_tests);
    
    if success_rate >= 100.0 {
        println!("   🎉 PERFECT! Complete round-trip functionality working!");
    } else if success_rate >= 80.0 {
        println!("   ✅ GOOD! Most functionality working with minor issues.");
    } else {
        println!("   ⚠️  PARTIAL! Some functionality needs improvement.");
    }

    // Performance summary
    println!("\n📊 Performance Summary:");
    println!("   Encoding: {:.2}s for {} chunks", encoding_stats.processing_time, encoding_stats.total_chunks);
    println!("   Video size: {:.2} KB ({:.1} bytes/chunk)", 
             encoding_stats.video_file_size as f64 / 1024.0,
             encoding_stats.video_file_size as f64 / encoding_stats.total_chunks as f64);
    println!("   Database size: {:.2} KB", db_stats.database_size_bytes as f64 / 1024.0);
    println!("   Total storage: {:.2} KB", 
             (encoding_stats.video_file_size + db_stats.database_size_bytes as u64) as f64 / 1024.0);

    println!("\n🚀 Round-trip test completed!");
    println!("Files created:");
    println!("   Video: {}", video_path.display());
    println!("   Database: {}", database_path.display());

    Ok(())
} 