//! Machine Learning module for memvid-rs
//!
//! This module provides comprehensive pure Rust machine learning capabilities using the Candle framework
//! for embedding generation, vector search, and semantic retrieval without Python dependencies.

pub mod device;
pub mod embedding;
pub mod index;
pub mod models;
pub mod search;
pub mod text;

// Re-export main types and functions
pub use device::{DeviceInfo, DeviceManager, DeviceType};
pub use embedding::{EmbeddingModel, EmbeddingConfig, Embedding};
pub use index::{IndexManager, ChunkMetadata, IndexStats};
pub use models::{ModelManager, ModelType, ModelInfo, ModelConfig};
pub use search::{VectorSearchIndex, SearchConfig, SearchResult, DistanceMetric};
pub use text::{TextProcessor, TextConfig, TokenizedText};

use crate::error::Result;

/// Initialize the ML system with automatic device detection
pub async fn initialize() -> Result<()> {
    log::info!("Initializing comprehensive ML system...");
    
    // Initialize device manager with auto-detection
    device::initialize()?;
    let device_info = device::current_device()?;
    log::info!("Using optimal device: {} (score: {})", device_info.name, device_info.compute_score);
    
    log::info!("ML system initialized successfully");
    Ok(())
}

/// Get current ML device information
pub fn current_device() -> Result<&'static DeviceInfo> {
    device::current_device()
}

/// Get all available ML devices
pub fn available_devices() -> Result<&'static [DeviceInfo]> {
    device::available_devices()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_complete_ml_pipeline() {
        // Initialize ML system
        initialize().await.unwrap();
        
        // Test device detection
        let device_info = current_device().unwrap();
        assert!(!device_info.name.is_empty());
        assert!(device_info.compute_score > 0.0);
        
        // Create embedding model (placeholder for now)
        let config = EmbeddingConfig::default();
        let mut embedding_model = EmbeddingModel::new(config).await.unwrap();
        
        // Create index manager
        let mut index_manager = IndexManager::new(384, None).unwrap();
        
        // Test data
        let texts = vec![
            "Machine learning is a subset of artificial intelligence".to_string(),
            "Deep learning uses neural networks with multiple layers".to_string(),
            "Natural language processing helps computers understand text".to_string(),
        ];
        
        // Generate embeddings (placeholder implementation)
        let mut embeddings = Vec::new();
        for text in &texts {
            let embedding = embedding_model.encode(text).unwrap();
            embeddings.push(embedding);
        }
        
        // Add to index
        let frame_numbers = vec![0, 1, 2];
        let chunk_ids = index_manager.add_chunks(&texts, &embeddings, &frame_numbers).unwrap();
        assert_eq!(chunk_ids.len(), 3);
        
        // Build index
        index_manager.build().unwrap();
        
        // Test search
        let query_embedding = embedding_model.encode("artificial intelligence").unwrap();
        let search_results = index_manager.search(&query_embedding, 2).unwrap();
        assert_eq!(search_results.len(), 2);
        
        // Test frame-chunk mapping
        let frame_0_chunks = index_manager.get_chunks_by_frame(0);
        assert_eq!(frame_0_chunks.len(), 1);
        assert_eq!(frame_0_chunks[0].text, texts[0]);
        
        // Test statistics
        let stats = index_manager.get_stats();
        assert_eq!(stats.total_chunks, 3);
        assert_eq!(stats.total_frames, 3);
        assert_eq!(stats.dimension, 384);
        
        // Test save/load
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");
        
        index_manager.save(&index_path).unwrap();
        let loaded_manager = IndexManager::load(&index_path).unwrap();
        assert_eq!(loaded_manager.chunk_count(), 3);
        
        log::info!("Complete ML pipeline test passed successfully!");
    }

    #[test]
    fn test_ml_module_exports() {
        // Test that all main types are properly exported
        let _config = EmbeddingConfig::default();
        let _search_config = SearchConfig::default();
        
        // Test device types
        let device_types = vec![
            DeviceType::Cpu,
            DeviceType::Metal,
        ];
        assert_eq!(device_types.len(), 2);
        
        // Test distance metrics
        let metrics = vec![
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
            DistanceMetric::DotProduct,
        ];
        assert_eq!(metrics.len(), 4);
    }
} 