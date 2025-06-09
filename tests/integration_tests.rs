//! Intergration tests demonstration
//!
//! This example demonstrates how memvid-rs provides equivalent functionality
//! to the Python version, with comprehensive test coverage for all major features.

use memvid_rs::api::{MemvidEncoder, MemvidRetriever};
use memvid_rs::qr::{QrEncoder, QrDecoder};
use memvid_rs::text::TextChunker;
use memvid_rs::config::{QrConfig, ChunkingConfig};
use memvid_rs::text::ChunkingStrategy;
use tempfile;

#[tokio::test]
async fn test_encoder_functionality() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    // Test encoder initialization
    let mut encoder = MemvidEncoder::new(None).await?;
    
    // Test add_chunks (equivalent to Python add_chunks)
    let chunks = vec![
        "Quantum computing uses qubits for parallel processing".to_string(),
        "Machine learning models require large datasets".to_string(),
        "Neural networks mimic brain structure".to_string(),
    ];
    encoder.add_chunks(chunks.clone())?;
    
    // Test add_text with chunking (equivalent to Python add_text)
    let long_text = "This is a test. ".repeat(50); // 800 characters
    encoder.add_text(&long_text, 100, 20).await?;
    
    // Test build_video (equivalent to Python build_video)
    let temp_dir = tempfile::tempdir()?;
    let video_file = temp_dir.path().join("demo.mp4");
    let index_file = temp_dir.path().join("demo_index.db");
    
    let stats = encoder.build_video(
        video_file.to_str().unwrap(),
        index_file.to_str().unwrap()
    ).await?;
    
    // Verify stats
    assert!(stats.total_chunks > 0);
    assert!(stats.total_frames > 0);
    assert!(stats.video_file_size > 0);
    assert!(stats.processing_time >= 0.0);
    
    // Test encoder stats (equivalent to Python get_stats)
    let encoder_stats = encoder.get_stats();
    assert!(encoder_stats.total_chunks > 0);
    
    // Test clear (equivalent to Python clear)
    encoder.clear();
    let stats_after_clear = encoder.get_stats();
    assert_eq!(stats_after_clear.total_chunks, 0);
    
    Ok(())
}

#[tokio::test]
async fn test_retriever_functionality() -> Result<(), Box<dyn std::error::Error>> {
    // Setup test data
    let chunks = vec![
        "Quantum computing uses qubits for parallel processing".to_string(),
        "Machine learning models require large datasets".to_string(),
        "Neural networks mimic brain structure".to_string(),
        "Cloud computing provides scalable resources".to_string(),
        "Blockchain ensures data immutability".to_string(),
    ];
    
    let mut encoder = MemvidEncoder::new(None).await?;
    encoder.add_chunks(chunks.clone())?;
    
    let temp_dir = tempfile::tempdir()?;
    let video_file = temp_dir.path().join("retriever_demo.mp4");
    let index_file = temp_dir.path().join("retriever_demo_index.db");
    
    encoder.build_video(
        video_file.to_str().unwrap(),
        index_file.to_str().unwrap()
    ).await?;
    
    // Test retriever initialization (equivalent to Python MemvidRetriever)
    let retriever = MemvidRetriever::new(
        video_file.to_str().unwrap(),
        index_file.to_str().unwrap()
    ).await?;
    
    // Test search (equivalent to Python search)
    let results = retriever.search("quantum computing", 3).await?;
    assert!(!results.is_empty());
    assert!(results.len() <= 3);
    
    // Test search_with_metadata (equivalent to Python search_with_metadata)
    let metadata_results = retriever.search_with_metadata("machine learning", 2).await?;
    assert!(!metadata_results.is_empty());
    assert!(metadata_results.len() <= 2);
    
    // Test get_chunk_by_id (equivalent to Python get_chunk_by_id)
    let chunk = retriever.get_chunk_by_id(0).await?;
    assert!(chunk.is_some());
    
    // Test stats (equivalent to Python stats)
    let stats = retriever.get_stats()?;
    assert!(stats.total_frames > 0);
    assert!(stats.database_size_bytes > 0);
    
    // Test video info (equivalent to Python get_video_info)
    let video_info = retriever.get_video_info().await?;
    assert!(video_info.width > 0);
    assert!(video_info.height > 0);
    assert!(video_info.fps > 0.0);
    
    Ok(())
}

#[test]
fn test_qr_functionality() -> Result<(), Box<dyn std::error::Error>> {
    let encoder = QrEncoder::new(QrConfig::default());
    let decoder = QrDecoder::new();
    
    // Test basic encode/decode (equivalent to Python test_qr_encode_decode)
    let test_data = "Test data for QR encoding and decoding round trip";
    let qr_frame = encoder.encode_text(test_data)?;
    let decode_result = decoder.decode_image(&qr_frame.image)?;
    assert_eq!(decode_result.text, test_data);
    
    // Test various sizes (equivalent to Python test_qr_round_trip_various_sizes)
    let test_cases = vec![
        "Short",
        "Medium length test data that spans multiple words",
        "This is a much longer test string that will test the QR code capabilities with more substantial content.",
    ];
    
    for test_data in test_cases.iter() {
        let qr_frame = encoder.encode_text(test_data)?;
        let decode_result = decoder.decode_image(&qr_frame.image)?;
        assert_eq!(decode_result.text, *test_data);
    }
    
    // Test Unicode data (equivalent to Python Unicode tests)
    let unicode_data = "Hello 世界! 🚀 Testing Unicode: åäö, éèê, ñ, ü";
    let qr_frame = encoder.encode_text(unicode_data)?;
    let decode_result = decoder.decode_image(&qr_frame.image)?;
    assert_eq!(decode_result.text, unicode_data);
    
    // Test JSON data (equivalent to Python JSON tests)
    let json_data = r#"{"id": 123, "text": "Sample chunk", "metadata": {"source": "test", "page": 1}}"#;
    let qr_frame = encoder.encode_text(json_data)?;
    let decode_result = decoder.decode_image(&qr_frame.image)?;
    assert_eq!(decode_result.text, json_data);
    
    // Verify it's valid JSON
    let _: serde_json::Value = serde_json::from_str(&decode_result.text)?;
    
    Ok(())
}

#[test]
fn test_text_chunking() -> Result<(), Box<dyn std::error::Error>> {
    // Test basic chunking (equivalent to Python test_chunk_text)
    let config = ChunkingConfig {
        chunk_size: 50,
        overlap: 10,
        min_chunk_size: 10,
        max_chunk_size: 100,
    };
    let chunker = TextChunker::new(config.clone(), ChunkingStrategy::Character)?;
    
    let text = "This is a test document. It has multiple sentences. Each sentence should be processed correctly.";
    let chunks = chunker.chunk_text(text, Some("test_doc".to_string()))?;
    
    assert!(!chunks.is_empty());
    for chunk in &chunks {
        assert!(chunk.text.len() <= config.max_chunk_size);
        assert!(chunk.length > 0);
    }
    
    // Test long text with overlap (equivalent to Python test_chunk_text_with_overlap)
    let long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(20);
    let overlap_config = ChunkingConfig {
        chunk_size: 100,
        overlap: 20,
        min_chunk_size: 10,
        max_chunk_size: 150,
    };
    let overlap_chunker = TextChunker::new(overlap_config.clone(), ChunkingStrategy::Character)?;
    
    let overlapped_chunks = overlap_chunker.chunk_text(&long_text, Some("long_doc".to_string()))?;
    assert!(overlapped_chunks.len() > 1);
    
    // Verify overlaps exist
    for i in 1..overlapped_chunks.len() {
        let prev_chunk = &overlapped_chunks[i-1];
        let curr_chunk = &overlapped_chunks[i];
        
        // Check that chunks have reasonable offsets
        assert!(curr_chunk.offset >= prev_chunk.offset);
    }
    
    // Test metadata preservation (equivalent to Python metadata tests)
    for (i, chunk) in overlapped_chunks.iter().enumerate() {
        assert_eq!(chunk.id, i);
        assert!(!chunk.text.is_empty());
        assert!(chunk.length > 0);
    }
    
    // Test sentence-aware chunking (equivalent to Python sentence chunking)
    let sentence_config = ChunkingConfig {
        chunk_size: 80,
        overlap: 0,
        min_chunk_size: 10,
        max_chunk_size: 120,
    };
    let sentence_chunker = TextChunker::new(sentence_config.clone(), ChunkingStrategy::Sentence)?;
    
    let sentence_text = "First sentence here. Second sentence follows. Third sentence completes the test.";
    let sentence_chunks = sentence_chunker.chunk_text(sentence_text, Some("sentence_test".to_string()))?;
    
    assert!(!sentence_chunks.is_empty());
    for chunk in &sentence_chunks {
        assert!(chunk.text.len() <= sentence_config.max_chunk_size);
    }
    
    Ok(())
} 