//! MemvidEncoder - Main encoding API
//!
//! This provides the high-level interface for encoding text documents into QR code videos.

use crate::config::Config;
use crate::error::{MemvidError, Result};
use crate::storage::EncodingStats;
use crate::text::{TextChunker, ChunkingStrategy, ChunkMetadata, PdfProcessor};
use crate::qr::QrEncoder;
use crate::video::encoder::VideoEncoder;
use crate::ml::{EmbeddingModel, EmbeddingConfig};
use std::path::Path;
use regex;

/// Main encoder for creating QR code videos from text documents
pub struct MemvidEncoder {
    config: Config,
    chunks: Vec<ChunkMetadata>,
    qr_encoder: QrEncoder,
    text_chunker: TextChunker,
    video_encoder: VideoEncoder,
    embedding_model: EmbeddingModel,
}

impl MemvidEncoder {
    /// Create a new encoder with optional configuration
    pub async fn new(config: Option<Config>) -> Result<Self> {
        let config = config.unwrap_or_default();
        
        let qr_encoder = QrEncoder::new(config.qr.clone());
        let text_chunker = TextChunker::new(
            config.chunking.clone(),
            ChunkingStrategy::Sentence,
        )?;
        let video_encoder = VideoEncoder::new(config.video.clone());
        
        // Try to initialize embedding model for semantic search support
        let embedding_model = match EmbeddingModel::new(EmbeddingConfig::default()).await {
            Ok(model) => {
                log::info!("🧠 Embedding model initialized - semantic embeddings will be generated during encoding");
                model
            }
            Err(e) => {
                log::error!("❌ Failed to initialize embedding model: {}. Encoding will not proceed.", e);
                return Err(MemvidError::Generic(e.to_string()));
            }
        };
        
        Ok(Self {
            config,
            chunks: Vec::new(),
            qr_encoder,
            text_chunker,
            video_encoder,
            embedding_model,
        })
    }

    /// Add text content with custom chunk size and overlap
    pub async fn add_text(&mut self, text: &str, chunk_size: usize, overlap: usize) -> Result<()> {
        let mut custom_config = self.config.chunking.clone();
        custom_config.chunk_size = chunk_size;
        custom_config.overlap = overlap;
        
        let custom_chunker = TextChunker::new(custom_config, ChunkingStrategy::Sentence)?;
        let mut new_chunks = custom_chunker.chunk_text(text, None)?;
        
        // Update chunk IDs to be sequential
        let start_id = self.chunks.len();
        for (i, chunk) in new_chunks.iter_mut().enumerate() {
            chunk.id = start_id + i;
        }
        
        let chunk_count = new_chunks.len();
        self.chunks.extend(new_chunks);
        log::info!("Added {} chunks from text. Total: {}", 
                  chunk_count, self.chunks.len());
        
        Ok(())
    }

    /// Add PDF document content
    pub async fn add_pdf<P: AsRef<Path>>(&mut self, pdf_path: P) -> Result<()> {
        let path = pdf_path.as_ref();
        
        if !path.exists() {
            return Err(MemvidError::Pdf(format!("PDF file not found: {}", path.display())));
        }

        let text = PdfProcessor::extract_text(path)?;
        let source = Some(path.to_string_lossy().to_string());
        
        let mut new_chunks = self.text_chunker.chunk_text(&text, source)?;
        
        // Update chunk IDs to be sequential
        let start_id = self.chunks.len();
        for (i, chunk) in new_chunks.iter_mut().enumerate() {
            chunk.id = start_id + i;
        }
        
        let chunk_count = new_chunks.len();
        self.chunks.extend(new_chunks);
        log::info!("Added {} chunks from PDF {}. Total: {}", 
                  chunk_count, path.display(), self.chunks.len());
        
        Ok(())
    }

    /// Add plain text file content
    pub async fn add_text_file<P: AsRef<Path>>(&mut self, file_path: P) -> Result<()> {
        let path = file_path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(MemvidError::Io)?;
        
        let source = Some(path.to_string_lossy().to_string());
        let mut new_chunks = self.text_chunker.chunk_text(&text, source)?;
        
        // Update chunk IDs to be sequential
        let start_id = self.chunks.len();
        for (i, chunk) in new_chunks.iter_mut().enumerate() {
            chunk.id = start_id + i;
        }
        
        let chunk_count = new_chunks.len();
        self.chunks.extend(new_chunks);
        log::info!("Added {} chunks from text file {}. Total: {}", 
                  chunk_count, path.display(), self.chunks.len());
        
        Ok(())
    }

    /// Add markdown file content
    pub async fn add_markdown_file<P: AsRef<Path>>(&mut self, file_path: P) -> Result<()> {
        let content = std::fs::read_to_string(file_path)
            .map_err(MemvidError::Io)?;
        
        // Extract text content from markdown
        let text = self.extract_text_from_markdown(&content);
        
        // Use the configured chunking settings from config
        let chunk_size = self.config.chunking.chunk_size;
        let overlap = self.config.chunking.overlap;
        
        self.add_text(&text, chunk_size, overlap).await
    }

    /// Build video from chunks, generating embeddings if enabled
    pub async fn build_video(&mut self, output_file: &str, index_file: &str) -> Result<EncodingStats> {
        if self.chunks.is_empty() {
            return Err(MemvidError::Generic("No chunks added for encoding".to_string()));
        }

        let start_time = std::time::Instant::now();

        log::info!("Starting video encoding for {} chunks", self.chunks.len());

        // Generate embeddings if embedding model is available
        log::info!("🧠 Generating semantic embeddings for {} chunks...", self.chunks.len());
        
        let chunk_texts: Vec<String> = self.chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = self.embedding_model.encode_batch(&chunk_texts)?;
        
        // Assign embeddings to chunks
        for (chunk, embedding) in self.chunks.iter_mut().zip(embeddings.iter()) {
            chunk.embedding = Some(embedding.clone());
        }
        
        log::info!("✅ Generated embeddings for {} chunks", self.chunks.len());

        // Generate QR code frames
        log::info!("Generating QR code frames...");
        let mut qr_images = Vec::new();
        
        for (frame_num, chunk) in self.chunks.iter_mut().enumerate() {
            let qr_frame = self.qr_encoder.encode_text(&chunk.text)?;
            qr_images.push(qr_frame.image);
            
            // Update chunk with frame number
            chunk.frame = Some(frame_num as u32);
            
            if frame_num % 100 == 0 {
                log::info!("Generated {} QR frames", frame_num + 1);
            }
        }

        log::info!("Generated {} QR frames, encoding video...", qr_images.len());

        // Encode video using VideoEncoder
        self.video_encoder.encode_frames(&qr_images, output_file).await?;

        // Get file size
        let video_file_size = std::fs::metadata(output_file)
            .map(|m| m.len())
            .unwrap_or(0);

        // Save index to SQLite database (now with embeddings!)
        self.save_index(index_file).await?;

        let processing_time = start_time.elapsed().as_secs_f64();
        
        log::info!(
            "Video encoding completed: {} chunks → {} frames → {} MB in {:.2}s",
            self.chunks.len(),
            qr_images.len(),
            video_file_size as f64 / 1_048_576.0,
            processing_time
        );

        Ok(EncodingStats {
            total_chunks: self.chunks.len(),
            total_frames: qr_images.len(),
            processing_time,
            video_file_size,
        })
    }

    /// Build video with progress callback
    pub async fn build_video_with_progress<F>(
        &mut self, 
        output_file: &str, 
        index_file: &str,
        progress_callback: F
    ) -> Result<EncodingStats>
    where
        F: Fn(f32),
    {
        if self.chunks.is_empty() {
            return Err(MemvidError::Generic("No chunks added for encoding".to_string()));
        }

        let start_time = std::time::Instant::now();
        let total_chunks = self.chunks.len();

        progress_callback(0.0);

        // Generate embeddings if embedding model is available
        log::info!("🧠 Generating semantic embeddings for {} chunks...", self.chunks.len());
        
        let chunk_texts: Vec<String> = self.chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = self.embedding_model.encode_batch(&chunk_texts)?;
        
        // Assign embeddings to chunks
        for (chunk, embedding) in self.chunks.iter_mut().zip(embeddings.iter()) {
            chunk.embedding = Some(embedding.clone());
        }
        
        log::info!("✅ Generated embeddings for {} chunks", self.chunks.len());
        progress_callback(25.0); // Embeddings are ~25% of work

        // Generate QR code frames with progress
        log::info!("Generating QR code frames...");
        let mut qr_images = Vec::new();
        
        for (frame_num, chunk) in self.chunks.iter_mut().enumerate() {
            let qr_frame = self.qr_encoder.encode_text(&chunk.text)?;
            qr_images.push(qr_frame.image);
            
            chunk.frame = Some(frame_num as u32);
            
            // Update progress (QR generation is ~40% of total work, from 25% to 65%)
            let qr_progress = 25.0 + (frame_num + 1) as f32 / total_chunks as f32 * 40.0;
            progress_callback(qr_progress);
        }

        progress_callback(65.0);

        // Encode video
        log::info!("Encoding video...");
        self.video_encoder.encode_frames(&qr_images, output_file).await?;
        
        progress_callback(90.0);

        // Save index to SQLite database (now with embeddings!)
        self.save_index(index_file).await?;
        
        progress_callback(100.0);

        let processing_time = start_time.elapsed().as_secs_f64();
        let video_file_size = std::fs::metadata(output_file)
            .map(|m| m.len())
            .unwrap_or(0);

        log::info!(
            "Video encoding completed: {} chunks → {} frames → {} MB in {:.2}s",
            self.chunks.len(),
            qr_images.len(),
            video_file_size as f64 / 1_048_576.0,
            processing_time
        );

        Ok(EncodingStats {
            total_chunks: self.chunks.len(),
            total_frames: qr_images.len(),
            processing_time,
            video_file_size,
        })
    }

    /// Save chunk index to SQLite database
    async fn save_index(&self, database_file: &str) -> Result<()> {
        // Create parent directory if needed
        if let Some(parent) = Path::new(database_file).parent() {
            std::fs::create_dir_all(parent).map_err(MemvidError::Io)?;
        }
        
        let mut database = crate::storage::Database::new(database_file)?;
        database.insert_chunks(&self.chunks)?;
        
        let stats = database.get_stats()?;
        log::info!("Saved {} chunks to SQLite database {} ({} bytes)", 
                  stats.chunk_count, database_file, stats.file_size_bytes);
        Ok(())
    }

    /// Clear all chunks
    pub fn clear(&mut self) {
        self.chunks.clear();
        log::info!("Cleared all chunks");
    }

    /// Get encoding statistics
    pub fn get_stats(&self) -> EncodingStats {
        EncodingStats {
            total_chunks: self.chunks.len(),
            total_frames: self.chunks.iter().filter(|c| c.frame.is_some()).count(),
            processing_time: 0.0,
            video_file_size: 0,
        }
    }

    /// Get current chunk count
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Get configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Extract text from markdown content
    fn extract_text_from_markdown(&self, content: &str) -> String {
        // Basic markdown parsing - remove markdown syntax and keep text content
        let mut text = content.to_string();
        
        // Remove code blocks (``` or `)
        text = regex::Regex::new(r"```[\s\S]*?```").unwrap().replace_all(&text, " ").to_string();
        text = regex::Regex::new(r"`[^`]*`").unwrap().replace_all(&text, " ").to_string();
        
        // Remove headers (# ## ### etc)
        text = regex::Regex::new(r"^#{1,6}\s+").unwrap().replace_all(&text, "").to_string();
        
        // Remove links [text](url) -> text
        text = regex::Regex::new(r"\[([^\]]*)\]\([^\)]*\)").unwrap().replace_all(&text, "$1").to_string();
        
        // Remove images ![alt](url)
        text = regex::Regex::new(r"!\[[^\]]*\]\([^\)]*\)").unwrap().replace_all(&text, " ").to_string();
        
        // Remove emphasis **bold** *italic* -> text
        text = regex::Regex::new(r"\*\*([^\*]*)\*\*").unwrap().replace_all(&text, "$1").to_string();
        text = regex::Regex::new(r"\*([^\*]*)\*").unwrap().replace_all(&text, "$1").to_string();
        text = regex::Regex::new(r"__([^_]*)__").unwrap().replace_all(&text, "$1").to_string();
        text = regex::Regex::new(r"_([^_]*)_").unwrap().replace_all(&text, "$1").to_string();
        
        // Remove horizontal rules
        text = regex::Regex::new(r"^---+$").unwrap().replace_all(&text, " ").to_string();
        text = regex::Regex::new(r"^\*\*\*+$").unwrap().replace_all(&text, " ").to_string();
        
        // Remove HTML tags
        text = regex::Regex::new(r"<[^>]*>").unwrap().replace_all(&text, " ").to_string();
        
        // Clean up whitespace
        text = regex::Regex::new(r"\s+").unwrap().replace_all(&text, " ").to_string();
        text = text.trim().to_string();
        
        text
    }

    /// Add pre-chunked text content directly
    pub fn add_chunks(&mut self, chunks: Vec<String>) -> Result<()> {
        let start_id = self.chunks.len();
        
        for (i, text) in chunks.iter().enumerate() {
            let chunk = ChunkMetadata {
                id: start_id + i,
                text: text.clone(),
                source: None,
                page: None,
                offset: 0,
                length: text.len(),
                frame: None,
                embedding: None,
            };
            self.chunks.push(chunk);
        }
        
        log::info!("Added {} chunks directly. Total: {}", chunks.len(), self.chunks.len());
        Ok(())
    }

    /// Check if embeddings are enabled (always true)
    pub fn has_embeddings(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[tokio::test]
    async fn test_encoder_creation() {
        let encoder = MemvidEncoder::new(None).await.unwrap();
        assert_eq!(encoder.chunk_count(), 0);
    }

    #[tokio::test]
    async fn test_add_text() {
        let mut encoder = MemvidEncoder::new(None).await.unwrap();
        // Use longer text to meet min_chunk_size requirement (default: 100 chars)
        let test_text = "This is a longer test text that should definitely meet the minimum chunk size requirement for proper chunking. The text needs to be at least 100 characters long to create a valid chunk.";
        encoder.add_text(test_text, 1024, 32).await.unwrap();
        assert_eq!(encoder.chunk_count(), 1);
    }

    #[tokio::test]
    async fn test_add_text_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        // Write longer content to meet min_chunk_size
        writeln!(temp_file, "This is test content that is long enough to meet the minimum chunk size requirement. We need at least 100 characters to create a valid chunk for processing.").unwrap();

        let mut encoder = MemvidEncoder::new(None).await.unwrap();
        encoder.add_text_file(temp_file.path()).await.unwrap();
        assert!(encoder.chunk_count() > 0);
    }

    #[tokio::test]
    async fn test_clear() {
        let mut encoder = MemvidEncoder::new(None).await.unwrap();
        // Use longer text to meet min_chunk_size requirement
        let test_text = "This is test content that is long enough to meet the minimum chunk size requirement for proper chunking and processing. The text needs to be sufficient.";
        encoder.add_text(test_text, 1024, 32).await.unwrap();
        assert!(encoder.chunk_count() > 0);
        
        encoder.clear();
        assert_eq!(encoder.chunk_count(), 0);
    }

    #[tokio::test]
    async fn test_add_chunks_direct() {
        let mut encoder = MemvidEncoder::new(None).await.unwrap();
        let chunks = vec![
            "chunk1".to_string(),
            "chunk2".to_string(), 
            "chunk3".to_string()
        ];
        
        encoder.add_chunks(chunks.clone()).unwrap();
        
        assert_eq!(encoder.chunk_count(), 3);
        assert_eq!(encoder.chunks[0].text, "chunk1");
        assert_eq!(encoder.chunks[1].text, "chunk2");
        assert_eq!(encoder.chunks[2].text, "chunk3");
    }

    #[tokio::test]
    async fn test_add_text_chunking() {
        let mut encoder = MemvidEncoder::new(None).await.unwrap();
        // Create text that will definitely create multiple chunks
        let test_text = "This is a long test document that should be split into multiple chunks. ".repeat(20); // About 1460 characters
        
        encoder.add_text(&test_text, 500, 50).await.unwrap(); // Smaller chunk size to force multiple chunks
        
        assert!(encoder.chunk_count() > 1);
        // Verify no empty chunks
        for chunk in &encoder.chunks {
            assert!(!chunk.text.is_empty());
        }
    }

    #[tokio::test]
    async fn test_build_video_integration() {
        let mut encoder = MemvidEncoder::new(None).await.unwrap();
        let chunks = vec![
            "Test chunk 1: Important information".to_string(),
            "Test chunk 2: More data here".to_string(),
            "Test chunk 3: Final piece of info".to_string()
        ];
        encoder.add_chunks(chunks).unwrap();
        
        // Create temporary directory for test files
        let temp_dir = tempfile::tempdir().unwrap();
        let video_file = temp_dir.path().join("test.mp4");
        let index_file = temp_dir.path().join("test_index.db");
        
        // Build video
        let stats = encoder.build_video(
            video_file.to_str().unwrap(),
            index_file.to_str().unwrap()
        ).await.unwrap();
        
        // Check files exist
        assert!(video_file.exists());
        assert!(index_file.exists());
        
        // Check stats
        assert_eq!(stats.total_chunks, 3);
        assert_eq!(stats.total_frames, 3);
        assert!(stats.video_file_size > 0);
        assert!(stats.processing_time > 0.0);
    }

    #[tokio::test]
    async fn test_encoder_stats_detailed() {
        let mut encoder = MemvidEncoder::new(None).await.unwrap();
        let chunks = vec![
            "short".to_string(),
            "medium length chunk".to_string(),
            "this is a longer chunk with more text".to_string()
        ];
        encoder.add_chunks(chunks.clone()).unwrap();
        
        let stats = encoder.get_stats();
        assert_eq!(stats.total_chunks, 3);
        
        let total_chars: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(stats.total_chunks, 3);
        
        let avg_chunk_size = total_chars as f64 / chunks.len() as f64;
        assert!(avg_chunk_size > 0.0);
    }
} 