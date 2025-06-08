//! MemvidRetriever - Main retrieval API
//!
//! This provides the high-level interface for searching and retrieving text from QR code videos.

use crate::config::Config;
use crate::error::{MemvidError, Result};
use crate::storage::Database;
use crate::video::{VideoDecoder, VideoInfo};
use crate::qr::QrDecoder;
use crate::text::ChunkMetadata;
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
        
        log::info!("MemvidRetriever initialized for {} with database {}", video_path, database_path);
        
        Ok(Self {
            config: Config::default(),
            video_path,
            database_path,
            database,
            video_decoder,
            qr_decoder,
            frame_cache: HashMap::new(),
        })
    }

    /// Search for relevant chunks using database text search
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<(f32, String)>> {
        log::info!("Searching for: '{}' (top {})", query, top_k);
        
        // Use database text search for now (will upgrade to semantic search later)
        let chunks = self.database.search_chunks(query, top_k)?;
        
        let mut results = Vec::new();
        for chunk in chunks {
            // Score based on query relevance (simple substring match for now)
            let score = if chunk.text.to_lowercase().contains(&query.to_lowercase()) {
                1.0
            } else {
                0.5
            };
            
            results.push((score, chunk.text));
        }
        
        log::info!("Found {} results for query '{}'", results.len(), query);
        Ok(results)
    }

    /// Search with full metadata
    pub async fn search_with_metadata(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        log::info!("Searching with metadata for: '{}' (top {})", query, top_k);
        
        let chunks = self.database.search_chunks(query, top_k)?;
        
        let mut results = Vec::new();
        for chunk in chunks {
            let score = if chunk.text.to_lowercase().contains(&query.to_lowercase()) {
                1.0
            } else {
                0.5
            };
            
            results.push(SearchResult {
                score,
                text: chunk.text.clone(),
                metadata: Some(chunk),
            });
        }
        
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