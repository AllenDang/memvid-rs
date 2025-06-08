//! Index management for embeddings and vector search
//!
//! This module provides comprehensive index management capabilities including
//! building, persistence, metadata management, and frame-to-chunk mapping.
//! 
//! Phase 3D enhancements include rich metadata management, context windows,
//! parallel processing, and robust error recovery.

use crate::error::{MemvidError, Result};
use crate::ml::embedding::Embedding;
use crate::ml::search::{SearchConfig, SearchResult, VectorSearchIndex};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

/// Rich metadata types for advanced chunk management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetadataValue {
    /// String value
    Text(String),
    /// Numeric value
    Number(f64),
    /// Boolean flag
    Boolean(bool),
    /// Array of values
    Array(Vec<MetadataValue>),
    /// Nested object
    Object(HashMap<String, MetadataValue>),
    /// Timestamp for temporal metadata
    Timestamp(chrono::DateTime<chrono::Utc>),
    /// Reference to another chunk or frame
    Reference { chunk_id: Option<usize>, frame_id: Option<usize> },
}

/// Enhanced chunk metadata with rich typing and temporal information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Unique chunk ID
    pub id: usize,
    /// Text content of the chunk
    pub text: String,
    /// Frame number in video
    pub frame_number: usize,
    /// Length of the chunk in characters
    pub length: usize,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last updated timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Rich metadata with typed values
    pub metadata: HashMap<String, MetadataValue>,
    /// Legacy JSON metadata for backward compatibility
    pub legacy_metadata: HashMap<String, serde_json::Value>,
    /// Chunk importance score (0.0-1.0)
    pub importance_score: f32,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Context window configuration for advanced search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextWindowConfig {
    /// Number of chunks before the target chunk
    pub before: usize,
    /// Number of chunks after the target chunk
    pub after: usize,
    /// Include chunks from adjacent frames
    pub include_adjacent_frames: bool,
    /// Maximum total context chunks to return
    pub max_total: Option<usize>,
    /// Minimum importance score for context chunks
    pub min_importance: Option<f32>,
}

impl Default for ContextWindowConfig {
    fn default() -> Self {
        Self {
            before: 2,
            after: 2,
            include_adjacent_frames: true,
            max_total: Some(10),
            min_importance: None,
        }
    }
}

/// Search result with enhanced context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedSearchResult {
    /// Primary search result
    pub result: SearchResult,
    /// Context chunks around the result
    pub context: Vec<ChunkMetadata>,
    /// Relevance explanation
    pub relevance_info: RelevanceInfo,
}

/// Information about search result relevance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceInfo {
    /// Primary relevance score
    pub score: f32,
    /// Matched terms or concepts
    pub matched_terms: Vec<String>,
    /// Chunk importance factor
    pub importance_factor: f32,
    /// Temporal relevance factor
    pub temporal_factor: f32,
}

/// Processing statistics for performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    /// Total processing time
    pub total_time: Duration,
    /// Number of chunks processed
    pub chunks_processed: usize,
    /// Number of successful operations
    pub successful_operations: usize,
    /// Number of failed operations
    pub failed_operations: usize,
    /// Average time per chunk
    pub avg_time_per_chunk: Duration,
    /// Memory usage during processing
    pub peak_memory_mb: u64,
}

/// Index statistics with enhanced metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Total number of chunks
    pub total_chunks: usize,
    /// Total number of frames
    pub total_frames: usize,
    /// Average chunks per frame
    pub avg_chunks_per_frame: f32,
    /// Index dimension
    pub dimension: usize,
    /// Index type
    pub index_type: String,
    /// Memory usage estimate in bytes
    pub memory_usage_bytes: u64,
    /// HNSW build status
    pub hnsw_built: bool,
    /// Average chunk importance score
    pub avg_importance_score: f32,
    /// Most common tags
    pub common_tags: Vec<(String, usize)>,
    /// Temporal range of chunks
    pub temporal_range: Option<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>,
}

/// Comprehensive index manager for embeddings and metadata
pub struct IndexManager {
    /// Vector search index
    vector_index: VectorSearchIndex,
    /// Chunk metadata storage
    chunks: HashMap<usize, ChunkMetadata>,
    /// Frame to chunk mapping
    frame_to_chunks: HashMap<usize, Vec<usize>>,
    /// Chunk to frame mapping
    chunk_to_frame: HashMap<usize, usize>,
    /// Index dimension
    dimension: usize,
    /// Next available chunk ID
    next_chunk_id: usize,
}

impl IndexManager {
    /// Create new index manager
    pub fn new(dimension: usize, config: Option<SearchConfig>) -> Result<Self> {
        let config = config.unwrap_or_default();
        let vector_index = VectorSearchIndex::new(dimension, config.clone())?;

        Ok(Self {
            vector_index,
            chunks: HashMap::new(),
            frame_to_chunks: HashMap::new(),
            chunk_to_frame: HashMap::new(),
            dimension,
            next_chunk_id: 0,
        })
    }

    /// Add chunks with embeddings to the index
    pub fn add_chunks(
        &mut self,
        chunks: &[String],
        embeddings: &[Embedding],
        frame_numbers: &[usize],
    ) -> Result<Vec<usize>> {
        if chunks.len() != embeddings.len() || chunks.len() != frame_numbers.len() {
            return Err(MemvidError::MachineLearning(
                "Chunks, embeddings, and frame numbers must have the same length".to_string(),
            ));
        }

        let mut chunk_ids = Vec::new();

        for ((chunk_text, embedding), &frame_number) in chunks.iter().zip(embeddings.iter()).zip(frame_numbers.iter()) {
            let chunk_id = self.next_chunk_id;
            self.next_chunk_id += 1;

            // Create chunk metadata
            let now = chrono::Utc::now();
            let chunk_metadata = ChunkMetadata {
                id: chunk_id,
                text: chunk_text.clone(),
                frame_number,
                length: chunk_text.len(),
                created_at: now,
                updated_at: now,
                metadata: HashMap::new(),
                legacy_metadata: HashMap::new(),
                importance_score: 0.5,
                tags: Vec::new(),
            };

            // Add to vector index
            let mut vector_metadata = HashMap::new();
            vector_metadata.insert("frame".to_string(), serde_json::Value::Number(frame_number.into()));
            vector_metadata.insert("text".to_string(), serde_json::Value::String(chunk_text.clone()));
            vector_metadata.insert("length".to_string(), serde_json::Value::Number(chunk_text.len().into()));

            self.vector_index.add_vector(chunk_id, embedding, Some(vector_metadata))?;

            // Store metadata
            self.chunks.insert(chunk_id, chunk_metadata);

            // Update frame mappings
            self.frame_to_chunks
                .entry(frame_number)
                .or_default()
                .push(chunk_id);
            self.chunk_to_frame.insert(chunk_id, frame_number);

            chunk_ids.push(chunk_id);
        }

        log::info!("Added {} chunks to index", chunks.len());
        Ok(chunk_ids)
    }

    /// Add single chunk with embedding
    pub fn add_chunk(
        &mut self,
        text: &str,
        embedding: &Embedding,
        frame_number: usize,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<usize> {
        let chunk_id = self.next_chunk_id;
        self.next_chunk_id += 1;

        // Create chunk metadata
        let now = chrono::Utc::now();
        let chunk_metadata = ChunkMetadata {
            id: chunk_id,
            text: text.to_string(),
            frame_number,
            length: text.len(),
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
            legacy_metadata: metadata.unwrap_or_default(),
            importance_score: 0.5,
            tags: Vec::new(),
        };

        // Add to vector index
        let mut vector_metadata = HashMap::new();
        vector_metadata.insert("frame".to_string(), serde_json::Value::Number(frame_number.into()));
        vector_metadata.insert("text".to_string(), serde_json::Value::String(text.to_string()));
        vector_metadata.insert("length".to_string(), serde_json::Value::Number(text.len().into()));

        // Add custom legacy metadata to vector metadata
        for (key, value) in &chunk_metadata.legacy_metadata {
            vector_metadata.insert(key.clone(), value.clone());
        }

        self.vector_index.add_vector(chunk_id, embedding, Some(vector_metadata))?;

        // Store metadata
        self.chunks.insert(chunk_id, chunk_metadata);

        // Update frame mappings
        self.frame_to_chunks
            .entry(frame_number)
            .or_default()
            .push(chunk_id);
        self.chunk_to_frame.insert(chunk_id, frame_number);

        Ok(chunk_id)
    }

    /// Search for similar chunks using semantic search
    pub fn search(&self, query_embedding: &Embedding, top_k: usize) -> Result<Vec<SearchResult>> {
        self.vector_index.search_approximate(query_embedding, top_k)
    }

    /// Search with exact algorithm (slower but more accurate)
    pub fn search_exact(&self, query_embedding: &Embedding, top_k: usize) -> Result<Vec<SearchResult>> {
        self.vector_index.search_exact(query_embedding, top_k)
    }

    /// Get chunks by frame number
    pub fn get_chunks_by_frame(&self, frame_number: usize) -> Vec<&ChunkMetadata> {
        if let Some(chunk_ids) = self.frame_to_chunks.get(&frame_number) {
            chunk_ids
                .iter()
                .filter_map(|&id| self.chunks.get(&id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get chunk by ID
    pub fn get_chunk_by_id(&self, chunk_id: usize) -> Option<&ChunkMetadata> {
        self.chunks.get(&chunk_id)
    }

    /// Get frame number for chunk
    pub fn get_frame_for_chunk(&self, chunk_id: usize) -> Option<usize> {
        self.chunk_to_frame.get(&chunk_id).copied()
    }

    /// Get all chunks for multiple frames
    pub fn get_chunks_by_frames(&self, frame_numbers: &[usize]) -> HashMap<usize, Vec<&ChunkMetadata>> {
        let mut result = HashMap::new();
        for &frame_number in frame_numbers {
            result.insert(frame_number, self.get_chunks_by_frame(frame_number));
        }
        result
    }

    /// Get context window around a chunk (surrounding chunks)
    pub fn get_context_window(&self, chunk_id: usize, window_size: usize) -> Vec<&ChunkMetadata> {
        if let Some(&frame_number) = self.chunk_to_frame.get(&chunk_id) {
            let start_frame = frame_number.saturating_sub(window_size);
            let end_frame = frame_number + window_size;

            let mut context_chunks = Vec::new();
            for frame in start_frame..=end_frame {
                context_chunks.extend(self.get_chunks_by_frame(frame));
            }

            // Sort by chunk ID to maintain order
            context_chunks.sort_by_key(|chunk| chunk.id);
            context_chunks
        } else {
            Vec::new()
        }
    }

    /// Build the index for optimal search performance
    pub fn build(&mut self) -> Result<()> {
        self.vector_index.build()?;
        log::info!("Index built successfully with {} chunks", self.chunks.len());
        Ok(())
    }

    /// Save index to disk
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;

        // Save vector index
        self.vector_index.save(path)?;

        // Save chunk metadata
        let chunks_data = serde_json::to_string(&self.chunks)
            .map_err(|e| MemvidError::MachineLearning(format!("Failed to serialize chunks: {}", e)))?;
        std::fs::write(path.join("chunks.json"), chunks_data)?;

        // Save frame mappings
        let frame_mappings = serde_json::json!({
            "frame_to_chunks": self.frame_to_chunks,
            "chunk_to_frame": self.chunk_to_frame,
            "next_chunk_id": self.next_chunk_id,
            "dimension": self.dimension
        });
        let mappings_data = serde_json::to_string(&frame_mappings)
            .map_err(|e| MemvidError::MachineLearning(format!("Failed to serialize mappings: {}", e)))?;
        std::fs::write(path.join("mappings.json"), mappings_data)?;

        log::info!("Saved index to {:?}", path);
        Ok(())
    }

    /// Load index from disk
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        // Load mappings first to get dimension
        let mappings_data = std::fs::read_to_string(path.join("mappings.json"))?;
        let mappings: serde_json::Value = serde_json::from_str(&mappings_data)
            .map_err(|e| MemvidError::MachineLearning(format!("Failed to deserialize mappings: {}", e)))?;

        let dimension = mappings["dimension"]
            .as_u64()
            .ok_or_else(|| MemvidError::MachineLearning("Missing dimension in mappings".to_string()))? as usize;

        // Load vector index
        let vector_index = VectorSearchIndex::load(path, dimension)?;

        // Load chunk metadata
        let chunks_data = std::fs::read_to_string(path.join("chunks.json"))?;
        let chunks: HashMap<usize, ChunkMetadata> = serde_json::from_str(&chunks_data)
            .map_err(|e| MemvidError::MachineLearning(format!("Failed to deserialize chunks: {}", e)))?;

        // Load frame mappings
        let frame_to_chunks: HashMap<usize, Vec<usize>> = serde_json::from_value(mappings["frame_to_chunks"].clone())
            .map_err(|e| MemvidError::MachineLearning(format!("Failed to deserialize frame_to_chunks: {}", e)))?;

        let chunk_to_frame: HashMap<usize, usize> = serde_json::from_value(mappings["chunk_to_frame"].clone())
            .map_err(|e| MemvidError::MachineLearning(format!("Failed to deserialize chunk_to_frame: {}", e)))?;

        let next_chunk_id = mappings["next_chunk_id"]
            .as_u64()
            .ok_or_else(|| MemvidError::MachineLearning("Missing next_chunk_id in mappings".to_string()))? as usize;



        let manager = Self {
            vector_index,
            chunks,
            frame_to_chunks,
            chunk_to_frame,
            dimension,
            next_chunk_id,
        };

        log::info!("Loaded index from {:?} with {} chunks", path, manager.chunks.len());
        Ok(manager)
    }

    /// Get index statistics
    pub fn get_stats(&self) -> IndexStats {
        let total_chunks = self.chunks.len();
        let total_frames = self.frame_to_chunks.len();
        let avg_chunks_per_frame = if total_frames > 0 {
            total_chunks as f32 / total_frames as f32
        } else {
            0.0
        };

        // Estimate memory usage
        let vector_memory = total_chunks * self.dimension * std::mem::size_of::<f32>();
        let metadata_memory = self.chunks.values().map(|chunk| chunk.text.len() + std::mem::size_of::<ChunkMetadata>())
            .sum::<usize>();
        let mapping_memory = (self.frame_to_chunks.len() + self.chunk_to_frame.len()) * std::mem::size_of::<usize>() * 2;

        IndexStats {
            total_chunks,
            total_frames,
            avg_chunks_per_frame,
            dimension: self.dimension,
            index_type: "HNSW".to_string(),
            memory_usage_bytes: (vector_memory + metadata_memory + mapping_memory) as u64,
            hnsw_built: true,
            avg_importance_score: 0.5,
            common_tags: Vec::new(),
            temporal_range: None,
        }
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.vector_index.clear();
        self.chunks.clear();
        self.frame_to_chunks.clear();
        self.chunk_to_frame.clear();
        self.next_chunk_id = 0;
    }

    /// Get all frame numbers
    pub fn get_frame_numbers(&self) -> Vec<usize> {
        let mut frames: Vec<usize> = self.frame_to_chunks.keys().copied().collect();
        frames.sort();
        frames
    }

    /// Get chunk count
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Get frame count
    pub fn frame_count(&self) -> usize {
        self.frame_to_chunks.len()
    }

    /// Update chunk metadata (legacy JSON format)
    pub fn update_chunk_metadata(&mut self, chunk_id: usize, metadata: HashMap<String, serde_json::Value>) -> Result<()> {
        if let Some(chunk) = self.chunks.get_mut(&chunk_id) {
            chunk.legacy_metadata = metadata;
            chunk.updated_at = chrono::Utc::now();
            Ok(())
        } else {
            Err(MemvidError::MachineLearning(format!("Chunk {} not found", chunk_id)))
        }
    }

    /// Update chunk rich metadata
    pub fn update_rich_metadata(&mut self, chunk_id: usize, metadata: HashMap<String, MetadataValue>) -> Result<()> {
        if let Some(chunk) = self.chunks.get_mut(&chunk_id) {
            chunk.metadata = metadata;
            chunk.updated_at = chrono::Utc::now();
            Ok(())
        } else {
            Err(MemvidError::MachineLearning(format!("Chunk {} not found", chunk_id)))
        }
    }

    /// Get vector index reference for advanced operations
    pub fn vector_index(&self) -> &VectorSearchIndex {
        &self.vector_index
    }

    /// Get mutable vector index reference for advanced operations
    pub fn vector_index_mut(&mut self) -> &mut VectorSearchIndex {
        &mut self.vector_index
    }

    /// Enhanced context window with rich configuration
    pub fn get_enhanced_context_window(&self, chunk_id: usize, config: &ContextWindowConfig) -> Vec<ChunkMetadata> {
        if let Some(chunk) = self.chunks.get(&chunk_id) {
            let mut context_chunks = Vec::new();
            
            // Get chunks from current frame first
            let current_frame_chunks = self.get_chunks_by_frame(chunk.frame_number);
            let chunk_position = current_frame_chunks.iter().position(|c| c.id == chunk_id).unwrap_or(0);
            
            // Add chunks before the target
            let start_idx = chunk_position.saturating_sub(config.before);
            let end_idx = (chunk_position + config.after + 1).min(current_frame_chunks.len());
            
            for chunk_ref in &current_frame_chunks[start_idx..end_idx] {
                if let Some(importance) = config.min_importance {
                    if chunk_ref.importance_score >= importance {
                        context_chunks.push((*chunk_ref).clone());
                    }
                } else {
                    context_chunks.push((*chunk_ref).clone());
                }
            }
            
            // Add chunks from adjacent frames if configured
            if config.include_adjacent_frames {
                // Previous frame
                if chunk.frame_number > 0 {
                    let prev_chunks = self.get_chunks_by_frame(chunk.frame_number - 1);
                    for chunk_ref in prev_chunks.iter().rev().take(config.before) {
                        if let Some(importance) = config.min_importance {
                            if chunk_ref.importance_score >= importance {
                                context_chunks.insert(0, (*chunk_ref).clone());
                            }
                        } else {
                            context_chunks.insert(0, (*chunk_ref).clone());
                        }
                    }
                }
                
                // Next frame
                let next_chunks = self.get_chunks_by_frame(chunk.frame_number + 1);
                for chunk_ref in next_chunks.iter().take(config.after) {
                    if let Some(importance) = config.min_importance {
                        if chunk_ref.importance_score >= importance {
                            context_chunks.push((*chunk_ref).clone());
                        }
                    } else {
                        context_chunks.push((*chunk_ref).clone());
                    }
                }
            }
            
            // Apply max_total limit
            if let Some(max_total) = config.max_total {
                context_chunks.truncate(max_total);
            }
            
            // Sort by frame number and chunk position
            context_chunks.sort_by(|a, b| {
                a.frame_number.cmp(&b.frame_number)
                    .then_with(|| a.id.cmp(&b.id))
            });
            
            context_chunks
        } else {
            Vec::new()
        }
    }

    /// Parallel chunk processing with error recovery
    pub fn add_chunks_parallel(
        &mut self,
        chunks: &[String],
        embeddings: &[Embedding],
        frame_numbers: &[usize],
        importance_scores: Option<&[f32]>,
        tags: Option<&[Vec<String>]>,
    ) -> Result<(Vec<usize>, ProcessingStats)> {
        let start_time = Instant::now();
        
        if chunks.len() != embeddings.len() || chunks.len() != frame_numbers.len() {
            return Err(MemvidError::MachineLearning(
                "Chunks, embeddings, and frame numbers must have the same length".to_string(),
            ));
        }

        // Validate importance scores and tags if provided
        if let Some(scores) = importance_scores {
            if scores.len() != chunks.len() {
                return Err(MemvidError::MachineLearning(
                    "Importance scores length must match chunks length".to_string(),
                ));
            }
        }
        
        if let Some(tag_list) = tags {
            if tag_list.len() != chunks.len() {
                return Err(MemvidError::MachineLearning(
                    "Tags length must match chunks length".to_string(),
                ));
            }
        }

        let mut successful_operations = 0;
        let mut failed_operations = 0;
        let mut chunk_ids = Vec::new();

        // Process chunks in parallel batches for memory efficiency
        let batch_size = 100; // Process 100 chunks at a time
        let now = chrono::Utc::now();
        
        for (batch_idx, chunk_batch) in chunks.chunks(batch_size).enumerate() {
            let batch_start = batch_idx * batch_size;
            let batch_end = (batch_start + chunk_batch.len()).min(chunks.len());
            
            // Create batch of chunk metadata
            let batch_metadata: Vec<Result<ChunkMetadata>> = (batch_start..batch_end)
                .into_par_iter()
                .map(|i| {
                    let chunk_id = self.next_chunk_id + i;
                    let importance = importance_scores.map(|s| s[i]).unwrap_or(0.5);
                    let chunk_tags = tags.map(|t| t[i].clone()).unwrap_or_default();
                    
                    // Validate importance score
                    if !(0.0..=1.0).contains(&importance) {
                        return Err(MemvidError::MachineLearning(
                            format!("Importance score {} out of range [0.0, 1.0]", importance)
                        ));
                    }
                    
                    Ok(ChunkMetadata {
                        id: chunk_id,
                        text: chunks[i].clone(),
                        frame_number: frame_numbers[i],
                        length: chunks[i].len(),
                        created_at: now,
                        updated_at: now,
                        metadata: HashMap::new(),
                        legacy_metadata: HashMap::new(),
                        importance_score: importance,
                        tags: chunk_tags,
                    })
                })
                .collect();

            // Process batch results
            for (i, metadata_result) in batch_metadata.into_iter().enumerate() {
                let chunk_idx = batch_start + i;
                match metadata_result {
                    Ok(chunk_metadata) => {
                        let chunk_id = chunk_metadata.id;
                        
                        // Add to vector index with retry logic
                        let mut attempts = 0;
                        let max_attempts = 3;
                        let mut last_error = None;
                        
                        while attempts < max_attempts {
                            let mut vector_metadata = HashMap::new();
                                                         vector_metadata.insert("frame".to_string(), serde_json::json!(chunk_metadata.frame_number));
                             vector_metadata.insert("text".to_string(), serde_json::Value::String(chunk_metadata.text.clone()));
                             vector_metadata.insert("length".to_string(), serde_json::json!(chunk_metadata.length));
                             vector_metadata.insert("importance".to_string(), serde_json::json!(chunk_metadata.importance_score));
                            
                            match self.vector_index.add_vector(chunk_id, &embeddings[chunk_idx], Some(vector_metadata)) {
                                Ok(()) => {
                                    // Store metadata
                                    self.chunks.insert(chunk_id, chunk_metadata.clone());
                                    
                                    // Update frame mappings
                                    self.frame_to_chunks
                                        .entry(chunk_metadata.frame_number)
                                        .or_default()
                                        .push(chunk_id);
                                    self.chunk_to_frame.insert(chunk_id, chunk_metadata.frame_number);
                                    
                                    chunk_ids.push(chunk_id);
                                    successful_operations += 1;
                                    break;
                                }
                                Err(e) => {
                                    attempts += 1;
                                    last_error = Some(e);
                                    if attempts < max_attempts {
                                        std::thread::sleep(Duration::from_millis(100 * attempts as u64));
                                    }
                                }
                            }
                        }
                        
                        if attempts >= max_attempts {
                            log::error!("Failed to add chunk {} after {} attempts: {:?}", chunk_id, max_attempts, last_error);
                            failed_operations += 1;
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to create metadata for chunk {}: {}", chunk_idx, e);
                        failed_operations += 1;
                    }
                }
            }
            
            // Update next_chunk_id for next batch
            self.next_chunk_id += chunk_batch.len();
        }

        let total_time = start_time.elapsed();
        let avg_time_per_chunk = if !chunks.is_empty() {
            total_time / chunks.len() as u32
        } else {
            Duration::from_millis(0)
        };

        let stats = ProcessingStats {
            total_time,
            chunks_processed: chunks.len(),
            successful_operations,
            failed_operations,
            avg_time_per_chunk,
            peak_memory_mb: 0, // TODO: Implement memory monitoring
        };

        log::info!("Parallel processing completed: {} successful, {} failed, {:?} total time", 
                   successful_operations, failed_operations, total_time);

        Ok((chunk_ids, stats))
    }

    /// Enhanced search with rich context and relevance information
    pub fn search_enhanced(
        &self,
        query_embedding: &Embedding,
        top_k: usize,
        context_config: Option<&ContextWindowConfig>,
        filter_tags: Option<&[String]>,
        min_importance: Option<f32>,
    ) -> Result<Vec<EnhancedSearchResult>> {
        // Perform base search
        let search_results = self.vector_index.search_approximate(query_embedding, top_k * 2)?; // Get more to allow filtering
        
        let mut enhanced_results = Vec::new();
        let default_config = ContextWindowConfig::default();
        let context_config = context_config.unwrap_or(&default_config);
        
        for result in search_results.iter().take(top_k) {
            if let Some(chunk) = self.chunks.get(&result.id) {
                // Apply filters
                if let Some(tags) = filter_tags {
                    if !chunk.tags.iter().any(|tag| tags.contains(tag)) {
                        continue;
                    }
                }
                
                if let Some(min_imp) = min_importance {
                    if chunk.importance_score < min_imp {
                        continue;
                    }
                }
                
                // Get context window
                let context = self.get_enhanced_context_window(chunk.id, context_config);
                
                // Calculate relevance info
                let relevance_info = RelevanceInfo {
                    score: 1.0 - result.distance, // Convert distance to similarity
                    matched_terms: Vec::new(), // TODO: Implement term extraction
                    importance_factor: chunk.importance_score,
                    temporal_factor: 1.0, // TODO: Implement temporal relevance
                };
                
                enhanced_results.push(EnhancedSearchResult {
                    result: result.clone(),
                    context,
                    relevance_info,
                });
                
                if enhanced_results.len() >= top_k {
                    break;
                }
            }
        }
        
        Ok(enhanced_results)
    }

    /// Get chunks by tag with parallel processing
    pub fn get_chunks_by_tags(&self, tags: &[String], require_all: bool) -> Vec<ChunkMetadata> {
        self.chunks
            .par_iter()
            .filter_map(|(_, chunk)| {
                let matches = if require_all {
                    tags.iter().all(|tag| chunk.tags.contains(tag))
                } else {
                    tags.iter().any(|tag| chunk.tags.contains(tag))
                };
                
                if matches {
                    Some(chunk.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Bulk update chunk importance scores with validation
    pub fn update_importance_scores(&mut self, updates: HashMap<usize, f32>) -> Result<usize> {
        let mut updated_count = 0;
        let now = chrono::Utc::now();
        
        for (chunk_id, new_score) in updates {
            if !(0.0..=1.0).contains(&new_score) {
                return Err(MemvidError::MachineLearning(
                    format!("Importance score {} out of range [0.0, 1.0]", new_score)
                ));
            }
            
            if let Some(chunk) = self.chunks.get_mut(&chunk_id) {
                chunk.importance_score = new_score;
                chunk.updated_at = now;
                updated_count += 1;
            }
        }
        
        Ok(updated_count)
    }

    /// Advanced statistics with rich metadata analysis
    pub fn get_enhanced_stats(&self) -> IndexStats {
        let total_chunks = self.chunks.len();
        let total_frames = self.frame_to_chunks.len();
        let avg_chunks_per_frame = if total_frames > 0 {
            total_chunks as f32 / total_frames as f32
        } else {
            0.0
        };

        // Calculate average importance score
        let avg_importance_score = if total_chunks > 0 {
            self.chunks.values().map(|c| c.importance_score).sum::<f32>() / total_chunks as f32
        } else {
            0.0
        };

        // Get common tags
        let mut tag_counts: HashMap<String, usize> = HashMap::new();
        for chunk in self.chunks.values() {
            for tag in &chunk.tags {
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
            }
        }
        
        let mut common_tags: Vec<(String, usize)> = tag_counts.into_iter().collect();
        common_tags.sort_by(|a, b| b.1.cmp(&a.1));
        common_tags.truncate(10); // Top 10 tags

        // Calculate temporal range
        let temporal_range = if !self.chunks.is_empty() {
            let mut min_time = None;
            let mut max_time = None;
            
            for chunk in self.chunks.values() {
                if min_time.is_none() || chunk.created_at < min_time.unwrap() {
                    min_time = Some(chunk.created_at);
                }
                if max_time.is_none() || chunk.created_at > max_time.unwrap() {
                    max_time = Some(chunk.created_at);
                }
            }
            
            if let (Some(min), Some(max)) = (min_time, max_time) {
                Some((min, max))
            } else {
                None
            }
        } else {
            None
        };

        // Estimate memory usage
        let vector_memory = total_chunks * self.dimension * std::mem::size_of::<f32>();
        let metadata_memory = self.chunks.values().map(|chunk| chunk.text.len() + std::mem::size_of::<ChunkMetadata>())
            .sum::<usize>();
        let mapping_memory = (self.frame_to_chunks.len() + self.chunk_to_frame.len()) * std::mem::size_of::<usize>() * 2;

        IndexStats {
            total_chunks,
            total_frames,
            avg_chunks_per_frame,
            dimension: self.dimension,
            index_type: "Enhanced HNSW".to_string(),
            memory_usage_bytes: (vector_memory + metadata_memory + mapping_memory) as u64,
            hnsw_built: true,
            avg_importance_score,
            common_tags,
            temporal_range,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_index_manager_creation() {
        let manager = IndexManager::new(384, None).unwrap();
        assert_eq!(manager.dimension, 384);
        assert_eq!(manager.chunk_count(), 0);
        assert_eq!(manager.frame_count(), 0);
    }

    #[test]
    fn test_add_chunks() {
        let mut manager = IndexManager::new(3, None).unwrap();

        let chunks = vec!["Hello world".to_string(), "Test chunk".to_string()];
        let embeddings = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let frame_numbers = vec![0, 1];

        let chunk_ids = manager.add_chunks(&chunks, &embeddings, &frame_numbers).unwrap();
        assert_eq!(chunk_ids.len(), 2);
        assert_eq!(manager.chunk_count(), 2);
        assert_eq!(manager.frame_count(), 2);
    }

    #[test]
    fn test_frame_chunk_mapping() {
        let mut manager = IndexManager::new(2, None).unwrap();

        // Add multiple chunks to same frame
        let chunk_id1 = manager.add_chunk("Chunk 1", &vec![1.0, 0.0], 0, None).unwrap();
        let chunk_id2 = manager.add_chunk("Chunk 2", &vec![0.0, 1.0], 0, None).unwrap();
        let chunk_id3 = manager.add_chunk("Chunk 3", &vec![1.0, 1.0], 1, None).unwrap();

        // Test frame to chunks mapping
        let frame_0_chunks = manager.get_chunks_by_frame(0);
        assert_eq!(frame_0_chunks.len(), 2);

        let frame_1_chunks = manager.get_chunks_by_frame(1);
        assert_eq!(frame_1_chunks.len(), 1);

        // Test chunk to frame mapping
        assert_eq!(manager.get_frame_for_chunk(chunk_id1), Some(0));
        assert_eq!(manager.get_frame_for_chunk(chunk_id2), Some(0));
        assert_eq!(manager.get_frame_for_chunk(chunk_id3), Some(1));
    }

    #[test]
    fn test_context_window() {
        let mut manager = IndexManager::new(2, None).unwrap();

        // Add chunks across multiple frames
        for i in 0..5 {
            manager.add_chunk(&format!("Chunk {}", i), &vec![i as f32, 0.0], i, None).unwrap();
        }

        // Test context window around frame 2
        let context = manager.get_context_window(2, 1); // chunk_id 2, window size 1
        assert_eq!(context.len(), 3); // Frames 1, 2, 3
    }

    #[test]
    fn test_save_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");

        // Create and populate index
        let mut manager = IndexManager::new(2, None).unwrap();
        manager.add_chunk("Test chunk", &vec![1.0, 0.0], 0, None).unwrap();
        manager.build().unwrap();

        // Save index
        manager.save(&index_path).unwrap();

        // Load index
        let loaded_manager = IndexManager::load(&index_path).unwrap();
        assert_eq!(loaded_manager.chunk_count(), 1);
        assert_eq!(loaded_manager.dimension, 2);
    }

    #[test]
    fn test_search() {
        let mut manager = IndexManager::new(3, None).unwrap();

        // Add test chunks
        manager.add_chunk("Hello world", &vec![1.0, 0.0, 0.0], 0, None).unwrap();
        manager.add_chunk("Test chunk", &vec![0.0, 1.0, 0.0], 1, None).unwrap();
        manager.add_chunk("Another test", &vec![0.0, 0.0, 1.0], 2, None).unwrap();

        manager.build().unwrap();

        // Search for similar to first chunk
        let query = vec![0.9, 0.1, 0.0];
        let results = manager.search(&query, 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0); // Should be most similar to first chunk
    }



    #[tokio::test]
    async fn test_phase_3d_enhanced_context_window() {
        let mut manager = IndexManager::new(3, None).unwrap();

        // Add chunks across multiple frames
        let chunks = vec![
            "Frame 0 chunk 1".to_string(),
            "Frame 0 chunk 2".to_string(),
            "Frame 1 chunk 1".to_string(),
            "Frame 1 chunk 2".to_string(),
            "Frame 2 chunk 1".to_string(),
        ];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.9, 0.1],
            vec![0.0, 0.0, 1.0],
        ];
        let frame_numbers = vec![0, 0, 1, 1, 2];

        manager.add_chunks(&chunks, &embeddings, &frame_numbers).unwrap();

        // Test enhanced context window with configuration
        let config = ContextWindowConfig {
            before: 1,
            after: 1,
            include_adjacent_frames: true,
            max_total: Some(5),
            min_importance: None,
        };

        let context = manager.get_enhanced_context_window(2, &config); // Frame 1 chunk 1
        assert!(!context.is_empty());
        assert!(context.iter().any(|c| c.frame_number == 0)); // Previous frame
        assert!(context.iter().any(|c| c.frame_number == 1)); // Current frame
        assert!(context.iter().any(|c| c.frame_number == 2)); // Next frame
    }

    #[tokio::test]
    async fn test_phase_3d_parallel_processing() {
        let mut manager = IndexManager::new(3, None).unwrap();

        let chunks = vec![
            "Parallel chunk 1".to_string(),
            "Parallel chunk 2".to_string(),
            "Parallel chunk 3".to_string(),
            "Parallel chunk 4".to_string(),
        ];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.0],
        ];
        let frame_numbers = vec![0, 0, 1, 1];
        let importance_scores = vec![0.8, 0.6, 0.9, 0.7];
        let tags = vec![
            vec!["test".to_string(), "parallel".to_string()],
            vec!["test".to_string()],
            vec!["parallel".to_string()],
            vec!["test".to_string(), "parallel".to_string(), "advanced".to_string()],
        ];

        let (chunk_ids, stats) = manager.add_chunks_parallel(
            &chunks,
            &embeddings,
            &frame_numbers,
            Some(&importance_scores),
            Some(&tags),
        ).unwrap();

        assert_eq!(chunk_ids.len(), 4);
        assert_eq!(stats.successful_operations, 4);
        assert_eq!(stats.failed_operations, 0);
        assert_eq!(stats.chunks_processed, 4);
        assert!(stats.total_time.as_millis() < u128::MAX);
    }

    #[tokio::test]
    async fn test_phase_3d_enhanced_search() {
        let mut manager = IndexManager::new(3, None).unwrap();

        let chunks = vec![
            "Important document".to_string(),
            "Regular content".to_string(),
            "High priority info".to_string(),
        ];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let frame_numbers = vec![0, 1, 2];

        manager.add_chunks(&chunks, &embeddings, &frame_numbers).unwrap();

        // Update importance scores
        let mut updates = HashMap::new();
        updates.insert(0, 0.9); // High importance
        updates.insert(1, 0.3); // Low importance
        updates.insert(2, 0.8); // High importance
        let updated = manager.update_importance_scores(updates).unwrap();
        assert_eq!(updated, 3);

        // Test enhanced search with filtering
        let query = vec![1.0, 0.0, 0.0];
        let results = manager.search_enhanced(
            &query,
            2,
            None,
            None,
            Some(0.5), // Min importance filter
        ).unwrap();

        // Should only return chunks with importance >= 0.5 (but may return fewer due to search results)
        assert!(results.len() <= 2); // At most chunks 0 and 2
        assert!(results.len() >= 1); // At least one result
        for result in &results {
            assert!(result.relevance_info.importance_factor >= 0.5);
        }
    }

    #[tokio::test]
    async fn test_phase_3d_tag_filtering() {
        let mut manager = IndexManager::new(3, None).unwrap();

        let chunks = vec![
            "Machine learning content".to_string(),
            "Video processing info".to_string(),
            "General text".to_string(),
        ];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let frame_numbers = vec![0, 1, 2];
        let tags = vec![
            vec!["ml".to_string(), "ai".to_string()],
            vec!["video".to_string(), "processing".to_string()],
            vec!["general".to_string()],
        ];

        manager.add_chunks_parallel(
            &chunks,
            &embeddings,
            &frame_numbers,
            None,
            Some(&tags),
        ).unwrap();

        // Test tag filtering - require all tags
        let ml_chunks = manager.get_chunks_by_tags(&["ml".to_string()], false);
        assert_eq!(ml_chunks.len(), 1);
        assert!(ml_chunks[0].tags.contains(&"ml".to_string()));

        // Test tag filtering - any tag match
        let video_or_general = manager.get_chunks_by_tags(
            &["video".to_string(), "general".to_string()], 
            false
        );
        assert_eq!(video_or_general.len(), 2);
    }

    #[tokio::test]
    async fn test_phase_3d_enhanced_statistics() {
        let mut manager = IndexManager::new(3, None).unwrap();

        let chunks = vec![
            "Statistical analysis chunk".to_string(),
            "Data processing chunk".to_string(),
        ];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let frame_numbers = vec![0, 1];
        let importance_scores = vec![0.8, 0.6];
        let tags = vec![
            vec!["stats".to_string(), "analysis".to_string()],
            vec!["data".to_string(), "processing".to_string()],
        ];

        manager.add_chunks_parallel(
            &chunks,
            &embeddings,
            &frame_numbers,
            Some(&importance_scores),
            Some(&tags),
        ).unwrap();

        let stats = manager.get_enhanced_stats();
        assert_eq!(stats.total_chunks, 2);
        assert_eq!(stats.total_frames, 2);
        assert!((stats.avg_importance_score - 0.7).abs() < 0.01); // (0.8 + 0.6) / 2, allow for floating point precision
        assert!(!stats.common_tags.is_empty());
        assert!(stats.temporal_range.is_some());
        assert_eq!(stats.index_type, "Enhanced HNSW");
    }

    #[tokio::test]
    async fn test_phase_3d_metadata_management() {
        let mut manager = IndexManager::new(3, None).unwrap();

        let chunks = vec!["Metadata test chunk".to_string()];
        let embeddings = vec![vec![1.0, 0.0, 0.0]];
        let frame_numbers = vec![0];

        let chunk_ids = manager.add_chunks(&chunks, &embeddings, &frame_numbers).unwrap();
        let chunk_id = chunk_ids[0];

        // Test rich metadata update
        let mut rich_metadata = HashMap::new();
        rich_metadata.insert("priority".to_string(), MetadataValue::Text("high".to_string()));
        rich_metadata.insert("score".to_string(), MetadataValue::Number(0.95));
        rich_metadata.insert("processed".to_string(), MetadataValue::Boolean(true));
        rich_metadata.insert("created".to_string(), MetadataValue::Timestamp(chrono::Utc::now()));
        rich_metadata.insert("related".to_string(), MetadataValue::Reference { 
            chunk_id: Some(123), 
            frame_id: Some(456) 
        });

        manager.update_rich_metadata(chunk_id, rich_metadata).unwrap();

        let chunk = manager.get_chunk_by_id(chunk_id).unwrap();
        assert_eq!(chunk.metadata.len(), 5);
        assert!(matches!(chunk.metadata.get("priority"), Some(MetadataValue::Text(s)) if s == "high"));
        assert!(matches!(chunk.metadata.get("score"), Some(MetadataValue::Number(n)) if *n == 0.95));
        assert!(matches!(chunk.metadata.get("processed"), Some(MetadataValue::Boolean(true))));
    }

    #[tokio::test]
    async fn test_phase_3d_error_recovery() {
        let mut manager = IndexManager::new(3, None).unwrap();

        // Test with invalid importance scores (should fail gracefully)
        let chunks = vec!["Test chunk".to_string()];
        let embeddings = vec![vec![1.0, 0.0, 0.0]];
        let frame_numbers = vec![0];
        let invalid_scores = vec![1.5]; // Out of range [0.0, 1.0]

        let result = manager.add_chunks_parallel(
            &chunks,
            &embeddings,
            &frame_numbers,
            Some(&invalid_scores),
            None,
        );

        // The parallel processing should either fail or handle gracefully with failures
        match result {
            Ok((_, stats)) => {
                // If it succeeds, there should be failures reported
                assert!(stats.failed_operations > 0);
            }
            Err(_) => {
                // Error is also acceptable
            }
        }

        // Test bulk importance update with invalid values
        let mut invalid_updates = HashMap::new();
        invalid_updates.insert(999, -0.5); // Invalid score

        let update_result = manager.update_importance_scores(invalid_updates);
        assert!(update_result.is_err());
    }
} 