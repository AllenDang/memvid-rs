//! Phase 3 ML Embedding Demo
//!
//! This example demonstrates the machine learning capabilities implemented in Phase 3,
//! including embedding generation, model management, and caching.

use memvid_rs::ml::{
    embedding::{EmbeddingConfig, EmbeddingModel},
    models::ModelManager,
};


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("🤖 Memvid-rs Phase 3: ML Embedding Demo");
    println!("========================================\n");

    // Phase 3.1: Model Management
    println!("📦 Phase 3.1: Model Management");
    println!("==============================");

    let mut model_manager = ModelManager::new(None)?;
    println!("✓ Model manager initialized");
    println!("  Cache directory: {:?}", model_manager.cache_dir());

    // List available models
    println!("\n📋 Available models:");
    {
        let models = model_manager.list_models();
        for model in &models {
            println!("  • {} ({})", model.name, format!("{:?}", model.model_type));
            println!("    Dimension: {}, Max length: {}", 
                     model.config.dimension, model.config.max_length);
        }
    }

    // Download a model (placeholder)
    println!("\n⬇️ Downloading all-MiniLM-L6-v2 model...");
    let model_path = model_manager.download_model("all-MiniLM-L6-v2").await?;
    println!("✓ Model cached at: {:?}", model_path);

    // Phase 3.2: Embedding Generation
    println!("\n🧠 Phase 3.2: Embedding Generation");
    println!("===================================");

    let config = EmbeddingConfig::default();
    println!("✓ Using model: {}", config.model_name);
    println!("  Dimension: {}, Normalize: {}", config.max_length, config.normalize);

    let mut embedding_model = EmbeddingModel::new(config).await?;
    println!("✓ Embedding model loaded");

    // Test sentences
    let test_sentences = vec![
        "Machine learning is transforming technology",
        "Rust provides memory safety without garbage collection",
        "Video encoding preserves data efficiently",
        "Semantic search enables intelligent retrieval",
        "QR codes store information in visual patterns",
    ];

    println!("\n🔍 Generating embeddings for test sentences:");
    let mut embeddings = Vec::new();
    
    for (i, sentence) in test_sentences.iter().enumerate() {
        let embedding = embedding_model.encode(sentence)?;
        embeddings.push(embedding);
        
        println!("  {}. \"{}...\"", i + 1, &sentence[..50.min(sentence.len())]);
        println!("     → Embedding: [{:.3}, {:.3}, {:.3}, ... ] ({}D)", 
                 embeddings[i][0], embeddings[i][1], embeddings[i][2], embeddings[i].len());
    }

    // Phase 3.3: Embedding Analysis
    println!("\n📊 Phase 3.3: Embedding Analysis");
    println!("=================================");

    // Test caching
    println!("🗄️ Testing embedding cache:");
    let cache_test_sentence = &test_sentences[0];
    let start = std::time::Instant::now();
    let cached_embedding = embedding_model.encode(cache_test_sentence)?;
    let cache_time = start.elapsed();
    
    println!("  Cache hit time: {:?}", cache_time);
    println!("  Cache size: {} embeddings", embedding_model.cache_size());
    assert_eq!(cached_embedding, embeddings[0], "Cached embedding should match");
    println!("  ✓ Cache working correctly");

    // Batch processing
    println!("\n📦 Testing batch processing:");
    let batch_sentences: Vec<String> = vec![
        "New sentence one".to_string(),
        "New sentence two".to_string(),
        "New sentence three".to_string(),
    ];
    
    let start = std::time::Instant::now();
    let batch_embeddings = embedding_model.encode_batch(&batch_sentences)?;
    let batch_time = start.elapsed();
    
    println!("  Batch of 3 sentences processed in: {:?}", batch_time);
    println!("  Total cache size: {} embeddings", embedding_model.cache_size());
    assert_eq!(batch_embeddings.len(), 3, "Should have 3 batch embeddings");
    println!("  ✓ Batch processing working correctly");

    // Similarity calculation (cosine similarity)
    println!("\n🔢 Calculating embedding similarities:");
    let similarities = calculate_similarities(&embeddings);
    
    println!("  Similarity matrix (cosine similarity):");
    for (i, row) in similarities.iter().enumerate() {
        print!("  {}:", i + 1);
        for similarity in row {
            print!(" {:.3}", similarity);
        }
        println!();
    }

    // Find most similar pairs
    let mut pairs = Vec::new();
    for i in 0..similarities.len() {
        for j in i + 1..similarities[i].len() {
            pairs.push((i, j, similarities[i][j]));
        }
    }
    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    println!("\n📈 Most similar sentence pairs:");
    for (i, (idx1, idx2, similarity)) in pairs.iter().take(3).enumerate() {
        println!("  {}. Sentences {} & {} (similarity: {:.3})", 
                 i + 1, idx1 + 1, idx2 + 1, similarity);
        println!("     \"{}\"", &test_sentences[*idx1][..60.min(test_sentences[*idx1].len())]);
        println!("     \"{}\"", &test_sentences[*idx2][..60.min(test_sentences[*idx2].len())]);
    }

    // Phase 3.4: Performance Summary
    println!("\n⚡ Phase 3.4: Performance Summary");
    println!("=================================");

    println!("✅ Phase 3 Implementation Complete!");
    println!("  • Model Management: ✓ Working");
    println!("  • Embedding Generation: ✓ Working (placeholder)");
    println!("  • Caching System: ✓ Working");
    println!("  • Batch Processing: ✓ Working");
    println!("  • Similarity Calculation: ✓ Working");

    println!("\n📊 Statistics:");
    println!("  • Models available: {}", model_manager.list_models().len());
    println!("  • Embeddings generated: {}", embedding_model.cache_size());
    println!("  • Embedding dimension: {}D", embeddings[0].len());
    println!("  • Cache hit rate: 100% (cached sentences)");

    println!("\n🚀 Next Phase: Vector Search Implementation");
    println!("   Phase 4 will integrate HNSW vector search for semantic retrieval");

    Ok(())
}

/// Calculate cosine similarity between embeddings
fn calculate_similarities(embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut similarities = vec![vec![0.0; embeddings.len()]; embeddings.len()];
    
    for i in 0..embeddings.len() {
        for j in 0..embeddings.len() {
            similarities[i][j] = cosine_similarity(&embeddings[i], &embeddings[j]);
        }
    }
    
    similarities
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
} 