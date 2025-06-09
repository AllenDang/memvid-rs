//! Basic encoding integration tests
//!
//! This module tests core video encoding functionality

use memvid_rs::MemvidEncoder;
use tempfile;

#[tokio::test]
async fn test_basic_encoding() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // Sample text content to encode
    let sample_texts = vec![
        "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.",
        "WebAssembly (abbreviated Wasm) is a binary instruction format for a stack-based virtual machine.",
        "Machine learning is a method of data analysis that automates analytical model building.",
        "Quantum computing harnesses the phenomena of quantum mechanics to deliver a huge leap forward in computation.",
        "Blockchain is a distributed ledger technology that maintains a continuously growing list of records.",
    ];

    // Create temporary directory
    let temp_dir = tempfile::tempdir()?;

    // Initialize encoder with default configuration
    let mut encoder = MemvidEncoder::new(None).await?;

    // Add sample texts
    for text in &sample_texts {
        encoder.add_text(text, 512, 32).await?;
    }

    // Check stats before encoding (frames aren't created until build_video)
    let stats = encoder.get_stats();
    assert!(stats.total_chunks > 0);

    // Build video and index
    let video_path = temp_dir.path().join("sample_memory.mp4");
    let index_path = temp_dir.path().join("sample_memory.db");

    let encoding_stats = encoder
        .build_video(video_path.to_str().unwrap(), index_path.to_str().unwrap())
        .await?;

    // Verify encoding results
    assert!(encoding_stats.total_chunks > 0);
    assert!(encoding_stats.total_frames > 0);
    assert!(encoding_stats.processing_time >= 0.0);
    assert!(encoding_stats.video_file_size > 0);

    // Verify files exist
    assert!(video_path.exists());
    assert!(index_path.exists());

    // Verify file sizes are reasonable
    let video_size = std::fs::metadata(&video_path)?.len();
    let index_size = std::fs::metadata(&index_path)?.len();

    assert!(video_size > 0);
    assert!(index_size > 0);

    // Video should be reasonably sized for our test data
    assert!(video_size > 1000); // At least 1KB
    assert!(video_size < 10_000_000); // Less than 10MB for this small test

    Ok(())
}

#[tokio::test]
async fn test_encoding_with_chunks() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = tempfile::tempdir()?;

    // Initialize encoder
    let mut encoder = MemvidEncoder::new(None).await?;

    // Add chunks directly
    let chunks = vec![
        "Test chunk one with some content".to_string(),
        "Test chunk two with different content".to_string(),
        "Test chunk three with more content".to_string(),
    ];

    encoder.add_chunks(chunks.clone())?;

    let stats = encoder.get_stats();
    assert_eq!(stats.total_chunks, chunks.len());

    // Build video
    let video_path = temp_dir.path().join("chunks_test.mp4");
    let index_path = temp_dir.path().join("chunks_test.db");

    let encoding_stats = encoder
        .build_video(video_path.to_str().unwrap(), index_path.to_str().unwrap())
        .await?;

    assert_eq!(encoding_stats.total_chunks, chunks.len());
    assert!(encoding_stats.total_frames > 0);
    assert!(video_path.exists());
    assert!(index_path.exists());

    Ok(())
}

#[tokio::test]
async fn test_encoder_clear() -> Result<(), Box<dyn std::error::Error>> {
    let mut encoder = MemvidEncoder::new(None).await?;

    // Add some content
    encoder
        .add_text("Test content for clearing", 100, 10)
        .await?;

    let stats_before = encoder.get_stats();
    assert!(stats_before.total_chunks > 0);

    // Clear encoder
    encoder.clear();

    let stats_after = encoder.get_stats();
    assert_eq!(stats_after.total_chunks, 0);
    assert_eq!(stats_after.total_frames, 0);

    Ok(())
}
