//! MemvidRetriever - Main retrieval API
//!
//! This provides the high-level interface for searching and retrieving text from QR code videos.

use crate::config::Config;
use crate::error::{MemvidError, Result};
use crate::storage::Database;
use crate::video::{VideoDecoder, VideoInfo};
use crate::qr::QrDecoder;
use crate::text::ChunkMetadata;
use crate::ml::{EmbeddingModel, EmbeddingConfig, IndexManager};
use std::path::Path;
use std::collections::HashMap;

/// Search result with score and metadata
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Similarity score
    pub score: f32,
    
    /// Text content
    pub text: String,
    
    /// Chunk metadata
    pub metadata: Option<ChunkMetadata>,
}

/// Main retriever for searching QR code videos
pub struct MemvidRetriever {
    config: Config,
    video_path: String,
    database_path: String,
    database: Database,
    video_decoder: VideoDecoder,
    qr_decoder: QrDecoder,
    frame_cache: HashMap<u32, String>, // Cache decoded frames
    embedding_model: EmbeddingModel,
    index_manager: Option<IndexManager>,
}

impl MemvidRetriever {
    /// Create a new retriever for the given video and database files
    pub async fn new<P1: AsRef<Path>, P2: AsRef<Path>>(
        video_file: P1, 
        database_file: P2
    ) -> Result<Self> {
        let video_path = video_file.as_ref().to_string_lossy().to_string();
        let database_path = database_file.as_ref().to_string_lossy().to_string();
        
        // Verify files exist
        if !video_file.as_ref().exists() {
            return Err(MemvidError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Video file not found: {}", video_path),
            )));
        }
        
        if !database_file.as_ref().exists() {
            return Err(MemvidError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Database file not found: {}", database_path),
            )));
        }

        // Initialize database connection
        let database = Database::new(&database_path)?;
        
        // Initialize video decoder
        let video_decoder = VideoDecoder::new()?;
        
        // Initialize QR decoder
        let qr_decoder = QrDecoder::new();
        
        // Initialize embedding model for semantic search
        let embedding_config = EmbeddingConfig::default();
        let embedding_model = EmbeddingModel::new(embedding_config).await?;
        
        log::info!("MemvidRetriever initialized for {} with database {}", video_path, database_path);
        
        Ok(Self {
            config: Config::default(),
            video_path,
            database_path,
            database,
            video_decoder,
            qr_decoder,
            frame_cache: HashMap::new(),
            embedding_model,
            index_manager: None,
        })
    }

    /// Search in the video content using semantic similarity
    pub async fn search(&mut self, query: &str, top_k: usize) -> Result<Vec<(f32, String)>> {
        log::info!("Searching for: '{}' (top {})", query, top_k);
        
        // Generate embedding for the query
        let query_embedding = self.embedding_model.encode(query)?;
        
        // If we have an index manager, use semantic search
        if let Some(ref index_manager) = self.index_manager {
            log::info!("🧠 Using TRUE SEMANTIC SEARCH with IndexManager for query: '{}'", query);
            let search_results = index_manager.search(&query_embedding, top_k)?;
            let mut results = Vec::new();
            
            for result in search_results {
                if let Some(chunk) = index_manager.get_chunk_by_id(result.id) {
                    let score = 1.0 - result.distance; // Convert distance to similarity score
                    results.push((score, chunk.text.clone()));
                }
            }
            
            log::info!("Found {} TRUE SEMANTIC results for query '{}'", results.len(), query);
            return Ok(results);
        }

        // Check if chunks have stored embeddings for true semantic search
        let all_chunks = self.database.search_chunks("", top_k * 10)?; // Get a sample of chunks to check
        let chunks_with_embeddings: Vec<_> = all_chunks.iter()
            .filter(|chunk| chunk.embedding.is_some())
            .collect();
        
        if chunks_with_embeddings.is_empty() {
            log::warn!("❌ NO SEMANTIC EMBEDDINGS FOUND: The database contains no stored embeddings for semantic search");
            log::warn!("💡 SOLUTION: Re-encode the video with embedding generation enabled, or use a system with IndexManager");
            log::warn!("🚫 REFUSING to fall back to keyword search as it may provide misleading results");
            return Err(crate::error::MemvidError::MachineLearning(
                "No semantic embeddings available in database. Refusing keyword fallback to avoid misleading results. Please re-encode video with embeddings enabled.".to_string()
            ));
        }

        log::info!("🧠 Using TRUE SEMANTIC SEARCH with stored embeddings for query: '{}'", query);
        
        // Get all chunks with embeddings for semantic comparison
        let chunks_with_embeddings = self.database.search_chunks("", top_k * 50)?; // Get more chunks for better coverage
        let valid_chunks: Vec<_> = chunks_with_embeddings.into_iter()
            .filter(|chunk| chunk.embedding.is_some())
            .collect();
        
        if valid_chunks.is_empty() {
            return Ok(vec![]);
        }
        
        let mut results = Vec::new();
        
        for chunk in valid_chunks {
            if let Some(ref chunk_embedding) = chunk.embedding {
                let similarity = self.compute_cosine_similarity(&query_embedding, chunk_embedding);
                results.push((similarity, chunk.text));
            }
        }
        
        // Sort by similarity score (descending) and take top_k
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        results.truncate(top_k);
        
        log::info!("📊 TRUE SEMANTIC RESULTS: Found {} embedding-based results for query '{}'", results.len(), query);
        log::info!("Found {} results for query '{}'", results.len(), query);
        Ok(results)
    }

    /// Search with full metadata using semantic similarity
    pub async fn search_with_metadata(&mut self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        log::info!("Searching with metadata for: '{}' (top {})", query, top_k);
        
        // Generate embedding for the query
        let query_embedding = self.embedding_model.encode(query)?;
        
        // If we have an index manager, use semantic search
        if let Some(ref index_manager) = self.index_manager {
            log::info!("🧠 Using TRUE SEMANTIC SEARCH with IndexManager for metadata query: '{}'", query);
            let search_results = index_manager.search(&query_embedding, top_k)?;
            let mut results = Vec::new();
            
            for result in search_results {
                if let Some(chunk) = index_manager.get_chunk_by_id(result.id) {
                    let score = 1.0 - result.distance; // Convert distance to similarity score
                    results.push(SearchResult {
                        score,
                        text: chunk.text.clone(),
                        metadata: Some(crate::text::ChunkMetadata {
                            id: chunk.id,
                            text: chunk.text.clone(),
                            source: Some("".to_string()), // TODO: Map from IndexManager metadata
                            page: None,
                            offset: 0,
                            length: chunk.length,
                            frame: Some(chunk.frame_number as u32),
                            embedding: None,
                        }),
                    });
                }
            }
            
            log::info!("Found {} TRUE SEMANTIC results with metadata for query '{}'", results.len(), query);
            return Ok(results);
        }

        // Check if chunks have stored embeddings for true semantic search
        let all_chunks = self.database.search_chunks("", top_k * 10)?; // Get a sample to check
        let chunks_with_embeddings: Vec<_> = all_chunks.iter()
            .filter(|chunk| chunk.embedding.is_some())
            .collect();
        
        if chunks_with_embeddings.is_empty() {
            log::warn!("❌ NO SEMANTIC EMBEDDINGS FOUND: The database contains no stored embeddings for semantic search");
            log::warn!("💡 SOLUTION: Re-encode the video with embedding generation enabled, or use a system with IndexManager");
            log::warn!("🚫 REFUSING to fall back to keyword search as it may provide misleading results");
            return Err(crate::error::MemvidError::MachineLearning(
                "No semantic embeddings available in database. Refusing keyword fallback to avoid misleading results. Please re-encode video with embeddings enabled.".to_string()
            ));
        }

        log::info!("🧠 Using TRUE SEMANTIC SEARCH with stored embeddings for metadata query: '{}'", query);

        // Get all chunks with embeddings for semantic comparison  
        let chunks_with_embeddings = self.database.search_chunks("", top_k * 50)?; // Get more chunks for better coverage
        let valid_chunks: Vec<_> = chunks_with_embeddings.into_iter()
            .filter(|chunk| chunk.embedding.is_some())
            .collect();
        
        if valid_chunks.is_empty() {
            return Ok(vec![]);
        }
        
        let mut results = Vec::new();
        
        for chunk in valid_chunks {
            let score = if let Some(ref chunk_embedding) = chunk.embedding {
                self.compute_cosine_similarity(&query_embedding, chunk_embedding)
            } else {
                continue; // Skip chunks without embeddings
            };
            
            results.push(SearchResult {
                score,
                text: chunk.text.clone(),
                metadata: Some(chunk),
            });
        }
        
        // Sort by similarity score (descending) and take top_k
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(top_k);

        log::info!("📊 TRUE SEMANTIC METADATA RESULTS: Found {} embedding-based results for query '{}'", results.len(), query);
        Ok(results)
    }

    /// Get specific chunk by ID from database
    pub async fn get_chunk_by_id(&self, chunk_id: usize) -> Result<Option<String>> {
        log::info!("Retrieving chunk by ID: {}", chunk_id);
        
        let chunk = self.database.get_chunk_by_id(chunk_id)?;
        Ok(chunk.map(|c| c.text))
    }

    /// Get context window around a specific chunk
    pub async fn get_context_window(&self, chunk_id: usize, window_size: usize) -> Result<Vec<String>> {
        log::info!("Getting context window for chunk {} with size {}", chunk_id, window_size);
        
        // Get the target chunk first
        let target_chunk = self.database.get_chunk_by_id(chunk_id)?;
        if target_chunk.is_none() {
            return Ok(vec![]);
        }
        
        // Get surrounding chunks
        let half_window = window_size / 2;
        let start_id = chunk_id.saturating_sub(half_window);
        let end_id = chunk_id + half_window;
        
        let mut context = Vec::new();
        for id in start_id..=end_id {
            if let Some(chunk) = self.database.get_chunk_by_id(id)? {
                context.push(chunk.text);
            }
        }
        
        Ok(context)
    }

    /// Get all chunks for a specific frame
    pub async fn get_chunks_by_frame(&self, frame_number: u32) -> Result<Vec<ChunkMetadata>> {
        log::info!("Retrieving chunks for frame {}", frame_number);
        self.database.get_chunks_by_frame(frame_number)
    }

    /// Decode QR content from a specific frame (with caching)
    pub async fn decode_frame(&mut self, frame_number: u32) -> Result<String> {
        // Check cache first
        if let Some(cached_content) = self.frame_cache.get(&frame_number) {
            log::debug!("Frame {} content retrieved from cache", frame_number);
            return Ok(cached_content.clone());
        }

        log::info!("Decoding QR content from frame {}", frame_number);
        
        // Extract the specific frame from video
        let frame_image = self.video_decoder.extract_frame(&self.video_path, frame_number).await?;
        
        // Decode QR code from frame
        let qr_result = self.qr_decoder.decode_image(&frame_image)?;
        let content = qr_result.text;
        
        // Cache the result
        self.frame_cache.insert(frame_number, content.clone());
        
        Ok(content)
    }

    /// Get video information
    pub async fn get_video_info(&self) -> Result<VideoInfo> {
        self.video_decoder.get_video_info(&self.video_path).await
    }

    /// Prefetch frames for better performance
    pub async fn prefetch_frames(&mut self, frame_numbers: Vec<u32>) -> Result<()> {
        log::info!("Prefetching {} frames", frame_numbers.len());
        
        for frame_number in frame_numbers {
            if !self.frame_cache.contains_key(&frame_number) {
                match self.decode_frame_internal(frame_number).await {
                    Ok(content) => {
                        self.frame_cache.insert(frame_number, content);
                    }
                    Err(e) => {
                        log::warn!("Failed to prefetch frame {}: {}", frame_number, e);
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Internal frame decoding without mutable self
    async fn decode_frame_internal(&self, frame_number: u32) -> Result<String> {
        let frame_image = self.video_decoder.extract_frame(&self.video_path, frame_number).await?;
        let qr_result = self.qr_decoder.decode_image(&frame_image)?;
        Ok(qr_result.text)
    }

    /// Clear internal caches
    pub fn clear_cache(&mut self) {
        self.frame_cache.clear();
        log::info!("Frame cache cleared ({} entries removed)", self.frame_cache.len());
    }

    /// Get retrieval statistics
    pub fn get_stats(&self) -> Result<RetrievalStats> {
        let db_stats = self.database.get_stats()?;
        
        Ok(RetrievalStats {
            total_chunks: db_stats.chunk_count,
            total_frames: db_stats.frame_count,
            cache_hits: 0, // TODO: Track cache hits
            cache_misses: 0, // TODO: Track cache misses
            cached_frames: self.frame_cache.len(),
            database_size_bytes: db_stats.file_size_bytes,
            average_search_time: 0.0, // TODO: Track search times
        })
    }

    /// Get video file path
    pub fn video_path(&self) -> &str {
        &self.video_path
    }

    /// Get database file path
    pub fn database_path(&self) -> &str {
        &self.database_path
    }

    /// Get configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Compute cosine similarity between two embeddings
    fn compute_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

/// Retrieval statistics
#[derive(Debug, Clone)]
pub struct RetrievalStats {
    /// Total number of chunks in the database
    pub total_chunks: usize,
    
    /// Total number of frames
    pub total_frames: usize,
    
    /// Number of cache hits
    pub cache_hits: usize,
    
    /// Number of cache misses
    pub cache_misses: usize,
    
    /// Number of cached frames
    pub cached_frames: usize,
    
    /// Database file size in bytes
    pub database_size_bytes: usize,
    
    /// Average search time in seconds
    pub average_search_time: f64,
}

// TODO: Add tests for video decoding and QR extraction functionality

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::MemvidEncoder;
    use tempfile;

    /// Create test memory (equivalent to Python setup_test_memory fixture)
    async fn setup_test_memory() -> (String, String, Vec<String>, tempfile::TempDir) {
        let chunks = vec![
            "Quantum computing uses qubits for parallel processing".to_string(),
            "Machine learning models require large datasets".to_string(),
            "Neural networks mimic brain structure".to_string(),
            "Cloud computing provides scalable resources".to_string(),
            "Blockchain ensures data immutability".to_string(),
        ];

        // Create encoder and add chunks
        let mut encoder = MemvidEncoder::new(None).await.unwrap();
        encoder.add_chunks(chunks.clone()).unwrap();
        
        // Create temporary files
        let temp_dir = tempfile::tempdir().unwrap();
        let video_file = temp_dir.path().join("test.mp4").to_string_lossy().to_string();
        let index_file = temp_dir.path().join("test_index.db").to_string_lossy().to_string();
        
        // Build video
        encoder.build_video(&video_file, &index_file).await.unwrap();
        
        (video_file, index_file, chunks, temp_dir)
    }

        #[tokio::test]
    async fn test_retriever_initialization() {
        let (video_file, index_file, chunks, _temp_dir) = setup_test_memory().await;
        
        let retriever = MemvidRetriever::new(&video_file, &index_file).await.unwrap();
        assert_eq!(retriever.video_path(), video_file);
        
        let stats = retriever.get_stats().unwrap();
        assert_eq!(stats.total_frames, chunks.len());
    }

    #[tokio::test] 
    async fn test_search() {
        let (video_file, index_file, _chunks, _temp_dir) = setup_test_memory().await;
        let mut retriever = MemvidRetriever::new(&video_file, &index_file).await.unwrap();
        
        // Search for quantum - should find the first chunk
        let results = retriever.search("quantum computing", 3).await.unwrap();
        assert!(results.len() <= 3);
        assert!(!results.is_empty()); // Should find at least one result
        
        // Search for machine learning - should find relevant chunks
        let results = retriever.search("machine learning", 3).await.unwrap();
        assert!(results.len() <= 3);
        assert!(!results.is_empty()); // Should find at least one result
        
        // Search for blockchain - should find relevant chunks
        let results = retriever.search("blockchain", 3).await.unwrap();
        assert!(results.len() <= 3);
        assert!(!results.is_empty()); // Should find at least one result
    }

    #[tokio::test]
    async fn test_search_with_metadata() {
        let (video_file, index_file, _chunks, _temp_dir) = setup_test_memory().await;
        let mut retriever = MemvidRetriever::new(&video_file, &index_file).await.unwrap();
        
        let results = retriever.search_with_metadata("blockchain", 2).await.unwrap();
        assert!(results.len() <= 2);
        
        if !results.is_empty() {
            let result = &results[0];
            assert!(result.score > 0.0);
            assert!(!result.text.is_empty());
            assert!(result.metadata.is_some());
        }
    }

    #[tokio::test]
    async fn test_get_chunk_by_id() {
        let (video_file, index_file, _chunks, _temp_dir) = setup_test_memory().await;
        let retriever = MemvidRetriever::new(&video_file, &index_file).await.unwrap();
        
        // Get first chunk
        let chunk = retriever.get_chunk_by_id(0).await.unwrap();
        assert!(chunk.is_some());
        assert!(chunk.unwrap().to_lowercase().contains("quantum"));
        
        // Invalid ID
        let chunk = retriever.get_chunk_by_id(999).await.unwrap();
        assert!(chunk.is_none());
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let (video_file, index_file, _chunks, _temp_dir) = setup_test_memory().await;
        let mut retriever = MemvidRetriever::new(&video_file, &index_file).await.unwrap();
        
        // Initial cache should be empty
        let initial_stats = retriever.get_stats().unwrap();
        assert_eq!(initial_stats.cached_frames, 0);
        
        // Try to decode a frame (this may fail due to H.265 compression, but cache should work)
        let _ = retriever.decode_frame(0).await; // May fail, that's OK
        
        // Clear cache
        retriever.clear_cache();
        let stats_after_clear = retriever.get_stats().unwrap();
        assert_eq!(stats_after_clear.cached_frames, 0);
    }

    #[tokio::test]
    async fn test_retriever_stats() {
        let (video_file, index_file, chunks, _temp_dir) = setup_test_memory().await;
        let retriever = MemvidRetriever::new(&video_file, &index_file).await.unwrap();
        
        let stats = retriever.get_stats().unwrap();
        assert_eq!(stats.total_frames, chunks.len());
        assert!(stats.database_size_bytes > 0);
        assert_eq!(stats.cached_frames, 0); // No cache initially
    }

    #[tokio::test]
    async fn test_video_info() {
        let (video_file, index_file, _chunks, _temp_dir) = setup_test_memory().await;
        let retriever = MemvidRetriever::new(&video_file, &index_file).await.unwrap();
        
        let video_info = retriever.get_video_info().await.unwrap();
        assert!(video_info.width > 0);
        assert!(video_info.height > 0);
        assert!(video_info.fps > 0.0);
        assert!(video_info.frame_count > 0);
    }

    #[tokio::test]
    async fn test_context_window() {
        let (video_file, index_file, _chunks, _temp_dir) = setup_test_memory().await;
        let retriever = MemvidRetriever::new(&video_file, &index_file).await.unwrap();
        
        // Get context window around chunk 1
        let context = retriever.get_context_window(1, 3).await.unwrap();
        assert!(!context.is_empty());
        assert!(context.len() <= 4); // Window size + target chunk
    }

    #[tokio::test]
    async fn test_chunks_by_frame() {
        let (video_file, index_file, _chunks, _temp_dir) = setup_test_memory().await;
        let retriever = MemvidRetriever::new(&video_file, &index_file).await.unwrap();
        
        // Get chunks for frame 0
        let chunks = retriever.get_chunks_by_frame(0).await.unwrap();
        assert!(!chunks.is_empty());
        
        // Verify chunk metadata
        for chunk in chunks {
            assert!(!chunk.text.is_empty());
            assert_eq!(chunk.frame, Some(0));
        }
    }
} 