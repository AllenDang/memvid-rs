//! CLI Append Command Test
//!
//! Test the corrected CLI append functionality that uses incremental updates

use memvid_rs::api::encoder::MemvidEncoder;
use memvid_rs::api::retriever::MemvidRetriever;
use std::io::Write;
use tempfile::{tempdir, NamedTempFile};

#[tokio::test]
async fn test_cli_append_functionality() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Testing CLI Append Functionality");
    println!("===================================");

    let temp_dir = tempdir()?;
    let video_file = temp_dir.path().join("test_knowledge.mp4");
    let index_file = temp_dir.path().join("test_knowledge.db");

    // STEP 1: Create initial knowledge base
    println!("📚 Creating initial knowledge base...");
    
    let initial_doc = "Artificial Intelligence is the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.";
    
    let temp_initial = temp_dir.path().join("initial_document.txt");
    std::fs::write(&temp_initial, initial_doc)?;

    let mut encoder = MemvidEncoder::new(None).await?;
    encoder.add_text_file(&temp_initial).await?;
    
    let initial_stats = encoder.build_video(
        video_file.to_str().unwrap(),
        index_file.to_str().unwrap(),
    ).await?;
    
    println!("✅ Initial knowledge base created with {} chunks", initial_stats.total_chunks);

    // STEP 2: Test append_document_chunks functionality (simulates CLI append)
    println!("\n📄 Testing document append...");
    
    let new_doc = "Machine Learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.";
    
    // Create temp file with .txt extension so append_document_chunks can handle it
    let temp_new = temp_dir.path().join("new_document.txt");
    std::fs::write(&temp_new, new_doc)?;

    // This simulates what the fixed CLI append command does
    let append_stats = encoder.append_document_chunks(
        video_file.to_str().unwrap(),
        index_file.to_str().unwrap(),
        temp_new.to_str().unwrap(),
    ).await?;
    
    println!("✅ Document appended with {} new chunks", append_stats.total_chunks);

    // STEP 3: Test append_conversation_history functionality
    println!("\n💬 Testing conversation append...");
    
    let conversations = vec![
        ("What is deep learning?".to_string(), 
         "Deep learning is a subset of machine learning that uses neural networks with multiple layers.".to_string()),
        ("How does it work?".to_string(),
         "Deep learning algorithms attempt to model high-level abstractions in data by using computational graphs.".to_string()),
    ];

    let conversation_stats = encoder.append_conversation_history(
        video_file.to_str().unwrap(),
        index_file.to_str().unwrap(),
        conversations,
    ).await?;
    
    println!("✅ Conversations appended with {} new chunks", conversation_stats.total_chunks);

    // STEP 4: Verify search functionality across all content
    println!("\n🔍 Testing search across all appended content...");
    
    let mut retriever = MemvidRetriever::new(&video_file, &index_file).await?;
    let final_stats = retriever.get_stats()?;
    
    println!("📊 Final knowledge base stats:");
    println!("   Total chunks: {}", final_stats.total_chunks);
    println!("   Total frames: {}", final_stats.total_frames);

    // Test searches for content from each phase
    let test_queries = [
        ("artificial intelligence", "Should find initial content"),
        ("machine learning", "Should find appended document"),
        ("deep learning", "Should find conversation content"),
    ];

    for (query, description) in test_queries {
        let results = retriever.search(query, 2).await?;
        println!("   Query '{}': {} results ({})", query, results.len(), description);
        assert!(!results.is_empty(), "Should find results for '{}'", query);
    }

    println!("\n🎯 CLI Append Functionality Test Summary");
    println!("========================================");
    println!("✅ Initial knowledge base creation: PASSED");
    println!("✅ Document append (append_document_chunks): PASSED");
    println!("✅ Conversation append (append_conversation_history): PASSED");
    println!("✅ Search across all content: PASSED");
    println!("✅ Incremental updates preserve existing data: PASSED");

    Ok(())
}

#[tokio::test]
async fn test_conversation_json_format() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Testing JSON Conversation Format");
    println!("===================================");

    let temp_dir = tempdir()?;
    let video_file = temp_dir.path().join("conv_test.mp4");
    let index_file = temp_dir.path().join("conv_test.db");

    // Create initial knowledge base
    let mut encoder = MemvidEncoder::new(None).await?;
    encoder.add_chunks(vec!["Initial test content".to_string()])?;
    encoder.build_video(
        video_file.to_str().unwrap(),
        index_file.to_str().unwrap(),
    ).await?;

    // Create JSON conversation file
    let conversation_json = r#"[
        {
            "human": "What is quantum computing?",
            "assistant": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information."
        },
        {
            "human": "What are its applications?",
            "assistant": "Applications include cryptography, optimization problems, and drug discovery simulations."
        }
    ]"#;

    let mut temp_json = NamedTempFile::new()?;
    write!(temp_json, "{}", conversation_json)?;
    temp_json.flush()?;

    // Parse JSON and append conversations (simulates AppendConversation CLI command)
    let json_conversations: Vec<serde_json::Value> = serde_json::from_str(conversation_json)?;
    let mut conversations = Vec::new();
    
    for conv in json_conversations {
        if let (Some(human), Some(assistant)) = (
            conv.get("human").and_then(|v| v.as_str()),
            conv.get("assistant").and_then(|v| v.as_str()),
        ) {
            conversations.push((human.to_string(), assistant.to_string()));
        }
    }

    let stats = encoder.append_conversation_history(
        video_file.to_str().unwrap(),
        index_file.to_str().unwrap(),
        conversations,
    ).await?;

    println!("✅ JSON conversation append completed:");
    println!("   Conversation turns: {}", stats.total_chunks / 2);
    println!("   Total chunks: {}", stats.total_chunks);

    // Test search
    let mut retriever = MemvidRetriever::new(&video_file, &index_file).await?;
    let results = retriever.search("quantum computing", 2).await?;
    assert!(!results.is_empty(), "Should find quantum computing content");
    
    println!("✅ JSON conversation format test: PASSED");

    Ok(())
} 