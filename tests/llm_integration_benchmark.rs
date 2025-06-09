//! LLM Integration Performance Benchmark
//!
//! This test simulates real LLM usage patterns to measure the effectiveness
//! of performance optimizations in actual chat/search scenarios.

use memvid_rs::api::encoder::MemvidEncoder;
use memvid_rs::api::retriever::MemvidRetriever;
use std::time::Instant;
use tempfile::tempdir;

#[tokio::test]
async fn benchmark_llm_conversation_pattern() -> Result<(), Box<dyn std::error::Error>> {
    // Create a knowledge base similar to real use cases
    let temp_dir = tempdir()?;
    let video_file = temp_dir.path().join("llm_kb.mp4");
    let index_file = temp_dir.path().join("llm_kb.db");

    let knowledge_base: Vec<String> = vec![
        "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information.".to_string(),
        "Machine learning algorithms learn patterns from data without being explicitly programmed for specific tasks.".to_string(),
        "Blockchain technology provides a decentralized ledger system for secure and transparent transactions.".to_string(),
        "Artificial intelligence encompasses machine learning, natural language processing, and computer vision.".to_string(),
        "Neural networks are computing systems inspired by biological neural networks in animal brains.".to_string(),
        "Deep learning is a subset of machine learning using artificial neural networks with multiple layers.".to_string(),
        "Cryptocurrency relies on blockchain technology to enable peer-to-peer digital transactions.".to_string(),
        "Natural language processing enables computers to understand and generate human language.".to_string(),
        "Computer vision allows machines to interpret and understand visual information from the world.".to_string(),
        "Quantum supremacy refers to quantum computers outperforming classical computers on specific tasks.".to_string(),
    ];

    // Encode the knowledge base
    let mut encoder = MemvidEncoder::new(None).await?;
    encoder.add_chunks(knowledge_base)?;
    encoder.build_video(video_file.to_str().unwrap(), index_file.to_str().unwrap()).await?;

    let mut retriever = MemvidRetriever::new(&video_file, &index_file).await?;

    println!("\n🤖 LLM CONVERSATION PERFORMANCE TEST");
    println!("====================================");

    // Simulate realistic LLM conversation patterns
    let conversation_queries = [
        "What is quantum computing?",
        "How does machine learning work?", 
        "Explain blockchain technology",
        "What are neural networks?",
        "Tell me about quantum supremacy",
        "How is AI different from machine learning?", // Follow-up question
        "What applications use blockchain?", // Related follow-up
    ];

    let mut total_search_time = std::time::Duration::from_millis(0);
    let conversation_start = Instant::now();

    println!("Simulating LLM conversation with {} queries...", conversation_queries.len());
    println!();

    for (i, query) in conversation_queries.iter().enumerate() {
        let query_start = Instant::now();
        
        // This simulates what the LLM chat API does
        let results = retriever.search(query, 5).await?;
        let search_duration = query_start.elapsed();
        total_search_time += search_duration;

        // Extract context like the chat API does (but don't actually call LLM)
        let _context = results
            .iter()
            .take(3)
            .map(|(_score, text)| format!("[Context]: {}", text))
            .collect::<Vec<_>>()
            .join("\n\n");

        println!("Query {}: \"{}\"", i + 1, query);
        println!("  📊 Found {} results in {:?}", results.len(), search_duration);
        println!("  💾 Cache status: {} frames", retriever.get_stats()?.cached_frames);
        
        // Simulate LLM processing time (typical: 1-3 seconds)
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        println!();
    }

    let total_conversation_time = conversation_start.elapsed();

    println!("📈 LLM PERFORMANCE SUMMARY");
    println!("=========================");
    println!("Total queries: {}", conversation_queries.len());
    println!("Total search time: {:?}", total_search_time);
    println!("Average search time: {:?}", total_search_time / conversation_queries.len() as u32);
    println!("Total conversation time: {:?}", total_conversation_time);
    println!("Search overhead: {:.1}% of conversation", 
        (total_search_time.as_millis() as f64 / total_conversation_time.as_millis() as f64) * 100.0
    );

    let final_stats = retriever.get_stats()?;
    println!("Final cache efficiency: {} frames cached", final_stats.cached_frames);

    // Verify performance is suitable for LLM scenarios
    let avg_search_ms = total_search_time.as_millis() / conversation_queries.len() as u128;
    assert!(avg_search_ms < 500, "Average search time should be under 500ms for smooth LLM interaction");
    
    // Test cache effectiveness in follow-up queries
    let follow_up_start = Instant::now();
    let _follow_up_results = retriever.search("quantum machine learning", 3).await?;
    let follow_up_duration = follow_up_start.elapsed();
    
    println!("Follow-up query performance: {:?}", follow_up_duration);
    assert!(follow_up_duration.as_millis() < 300, "Follow-up queries should be faster due to caching");
    
    println!("✅ Performance suitable for LLM integration!");

    Ok(())
} 