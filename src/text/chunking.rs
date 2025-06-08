//! Text chunking algorithms and utilities
//!
//! This module provides various text chunking strategies for processing documents
//! into manageable segments for QR encoding.

use crate::error::{MemvidError, Result};
use crate::config::ChunkingConfig;
use regex::Regex;
use serde::{Deserialize, Serialize};

/// Metadata for a text chunk
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkMetadata {
    /// Unique chunk identifier
    pub id: usize,
    
    /// The actual text content
    pub text: String,
    
    /// Original document source
    pub source: Option<String>,
    
    /// Page number (for PDFs)
    pub page: Option<u32>,
    
    /// Character offset in original document
    pub offset: usize,
    
    /// Length of the chunk in characters
    pub length: usize,
    
    /// Frame number in the video (set during encoding)
    pub frame: Option<u32>,
    
    /// Embedding vector (set during ML processing)
    pub embedding: Option<Vec<f32>>,
}

/// Text chunking strategies
#[derive(Debug, Clone)]
pub enum ChunkingStrategy {
    /// Simple character-based chunking with overlap
    Character,
    
    /// Sentence-aware chunking (preserves sentence boundaries)
    Sentence,
    
    /// Paragraph-aware chunking
    Paragraph,
    
    /// Token-based chunking (requires tokenizer)
    Token,
}

/// Text chunker for processing documents into manageable chunks
pub struct TextChunker {
    config: ChunkingConfig,
    strategy: ChunkingStrategy,
    sentence_regex: Regex,
    paragraph_regex: Regex,
}

impl TextChunker {
    /// Create a new text chunker with the given configuration
    pub fn new(config: ChunkingConfig, strategy: ChunkingStrategy) -> Result<Self> {
        let sentence_regex = Regex::new(r"[.!?]+\s+")
            .map_err(|e| MemvidError::TextProcessing(format!("Failed to compile sentence regex: {}", e)))?;
        
        let paragraph_regex = Regex::new(r"\n\s*\n")
            .map_err(|e| MemvidError::TextProcessing(format!("Failed to compile paragraph regex: {}", e)))?;
        
        Ok(Self {
            config,
            strategy,
            sentence_regex,
            paragraph_regex,
        })
    }

    /// Create a chunker with default configuration
    pub fn with_default_config() -> Result<Self> {
        Self::new(ChunkingConfig::default(), ChunkingStrategy::Character)
    }

    /// Chunk text into overlapping segments
    pub fn chunk_text(&self, text: &str, source: Option<String>) -> Result<Vec<ChunkMetadata>> {
        let text = self.preprocess_text(text);
        
        match self.strategy {
            ChunkingStrategy::Character => self.chunk_by_characters(&text, source),
            ChunkingStrategy::Sentence => self.chunk_by_sentences(&text, source),
            ChunkingStrategy::Paragraph => self.chunk_by_paragraphs(&text, source),
            ChunkingStrategy::Token => {
                self.chunk_by_tokens(&text, source)
            }
        }
    }

    /// Preprocess text by normalizing whitespace and removing excessive newlines
    fn preprocess_text(&self, text: &str) -> String {
        // Normalize whitespace
        let normalized = text
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        
        // Remove excessive whitespace
        let whitespace_regex = Regex::new(r"\s+").unwrap();
        whitespace_regex.replace_all(&normalized, " ").to_string()
    }

    /// Character-based chunking with overlap
    fn chunk_by_characters(&self, text: &str, source: Option<String>) -> Result<Vec<ChunkMetadata>> {
        let mut chunks = Vec::new();
        let mut id = 0;
        
        if text.len() <= self.config.chunk_size {
            // Text is small enough to be a single chunk
            chunks.push(ChunkMetadata {
                id,
                text: text.to_string(),
                source: source.clone(),
                page: None,
                offset: 0,
                length: text.len(),
                frame: None,
                embedding: None,
            });
            return Ok(chunks);
        }

        let mut start = 0;
        while start < text.len() {
            let end = std::cmp::min(start + self.config.chunk_size, text.len());
            let chunk_text = text[start..end].to_string();
            
            // Skip chunks that are too small (except for the last chunk)
            if chunk_text.len() >= self.config.min_chunk_size || end == text.len() {
                chunks.push(ChunkMetadata {
                    id,
                    text: chunk_text.clone(),
                    source: source.clone(),
                    page: None,
                    offset: start,
                    length: chunk_text.len(),
                    frame: None,
                    embedding: None,
                });
                id += 1;
            }
            
            // Move start position with overlap
            start += self.config.chunk_size.saturating_sub(self.config.overlap);
            
            // Prevent infinite loop if overlap is too large
            if start <= chunks.last().map(|c| c.offset).unwrap_or(0) {
                start = chunks.last().map(|c| c.offset + c.length).unwrap_or(0);
            }
        }

        Ok(chunks)
    }

    /// Sentence-aware chunking that preserves sentence boundaries
    fn chunk_by_sentences(&self, text: &str, source: Option<String>) -> Result<Vec<ChunkMetadata>> {
        let sentences: Vec<&str> = self.sentence_regex.split(text).collect();
        let mut chunks = Vec::new();
        let mut id = 0;
        let mut current_chunk = String::new();
        let mut current_offset = 0;
        let mut sentence_offset = 0;

        for sentence in sentences {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }

            let sentence_with_space = if current_chunk.is_empty() {
                sentence.to_string()
            } else {
                format!(" {}", sentence)
            };

            // Check if adding this sentence would exceed chunk size
            if !current_chunk.is_empty() && 
               (current_chunk.len() + sentence_with_space.len()) > self.config.chunk_size {
                
                // Finalize current chunk
                if current_chunk.len() >= self.config.min_chunk_size {
                    chunks.push(ChunkMetadata {
                        id,
                        text: current_chunk.clone(),
                        source: source.clone(),
                        page: None,
                        offset: current_offset,
                        length: current_chunk.len(),
                        frame: None,
                        embedding: None,
                    });
                    id += 1;
                }

                // Start new chunk
                current_chunk = sentence.to_string();
                current_offset = sentence_offset;
            } else {
                current_chunk.push_str(&sentence_with_space);
            }

            sentence_offset += sentence.len() + 1; // +1 for space/punctuation
        }

        // Add final chunk if it exists
        if !current_chunk.is_empty() && current_chunk.len() >= self.config.min_chunk_size {
            let chunk_length = current_chunk.len();
            chunks.push(ChunkMetadata {
                id,
                text: current_chunk,
                source,
                page: None,
                offset: current_offset,
                length: chunk_length,
                frame: None,
                embedding: None,
            });
        }

        Ok(chunks)
    }

    /// Paragraph-aware chunking
    fn chunk_by_paragraphs(&self, text: &str, source: Option<String>) -> Result<Vec<ChunkMetadata>> {
        let paragraphs: Vec<&str> = self.paragraph_regex.split(text).collect();
        let mut chunks = Vec::new();
        let mut id = 0;
        let mut current_chunk = String::new();
        let mut current_offset = 0;
        let mut paragraph_offset = 0;

        for paragraph in paragraphs {
            let paragraph = paragraph.trim();
            if paragraph.is_empty() {
                continue;
            }

            let paragraph_with_newline = if current_chunk.is_empty() {
                paragraph.to_string()
            } else {
                format!("\n\n{}", paragraph)
            };

            // Check if adding this paragraph would exceed chunk size
            if !current_chunk.is_empty() && 
               (current_chunk.len() + paragraph_with_newline.len()) > self.config.chunk_size {
                
                // Finalize current chunk
                if current_chunk.len() >= self.config.min_chunk_size {
                    chunks.push(ChunkMetadata {
                        id,
                        text: current_chunk.clone(),
                        source: source.clone(),
                        page: None,
                        offset: current_offset,
                        length: current_chunk.len(),
                        frame: None,
                        embedding: None,
                    });
                    id += 1;
                }

                // Start new chunk
                current_chunk = paragraph.to_string();
                current_offset = paragraph_offset;
            } else {
                current_chunk.push_str(&paragraph_with_newline);
            }

            paragraph_offset += paragraph.len() + 2; // +2 for double newline
        }

        // Add final chunk if it exists
        if !current_chunk.is_empty() && current_chunk.len() >= self.config.min_chunk_size {
            let chunk_length = current_chunk.len();
            chunks.push(ChunkMetadata {
                id,
                text: current_chunk,
                source,
                page: None,
                offset: current_offset,
                length: chunk_length,
                frame: None,
                embedding: None,
            });
        }

        Ok(chunks)
    }

    /// Token-based chunking using simple whitespace tokenization
    fn chunk_by_tokens(&self, text: &str, source: Option<String>) -> Result<Vec<ChunkMetadata>> {
        // Simple token-based chunking using whitespace splits
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();
        let mut id = 0;
        let mut current_chunk = String::new();
        let mut current_offset = 0;
        let mut token_offset: usize = 0;

        for token in tokens {
            let token_with_space = if current_chunk.is_empty() {
                token.to_string()
            } else {
                format!(" {}", token)
            };

            // Check if adding this token would exceed chunk size
            if !current_chunk.is_empty() && 
               (current_chunk.len() + token_with_space.len()) > self.config.chunk_size {
                
                // Finalize current chunk
                if current_chunk.len() >= self.config.min_chunk_size {
                    chunks.push(ChunkMetadata {
                        id,
                        text: current_chunk.clone(),
                        source: source.clone(),
                        page: None,
                        offset: current_offset,
                        length: current_chunk.len(),
                        frame: None,
                        embedding: None,
                    });
                    id += 1;
                }

                // Start new chunk with overlap
                let overlap_tokens = if self.config.overlap > 0 {
                    let current_tokens: Vec<&str> = current_chunk.split_whitespace().collect();
                    let overlap_count = std::cmp::min(
                        self.config.overlap / 10, // Approximate tokens in overlap
                        current_tokens.len()
                    );
                    if overlap_count > 0 {
                        current_tokens[current_tokens.len() - overlap_count..].join(" ")
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };

                current_chunk = if overlap_tokens.is_empty() {
                    token.to_string()
                } else {
                    format!("{} {}", overlap_tokens, token)
                };
                current_offset = token_offset.saturating_sub(overlap_tokens.len());
            } else {
                current_chunk.push_str(&token_with_space);
            }

            token_offset += token.len() + 1; // +1 for space
        }

        // Add final chunk if it exists
        if !current_chunk.is_empty() && current_chunk.len() >= self.config.min_chunk_size {
            chunks.push(ChunkMetadata {
                id,
                text: current_chunk,
                source,
                page: None,
                offset: current_offset,
                length: token_offset - current_offset,
                frame: None,
                embedding: None,
            });
        }

        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_character_chunking() {
        let chunker = TextChunker::new(
            ChunkingConfig {
                chunk_size: 10,
                overlap: 2,
                min_chunk_size: 5,
                max_chunk_size: 50,
            },
            ChunkingStrategy::Character,
        ).unwrap();

        let text = "This is a test text for chunking functionality.";
        let chunks = chunker.chunk_text(text, None).unwrap();
        
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].id, 0);
        assert!(chunks[0].text.len() <= 10);
    }

    #[test]
    fn test_sentence_chunking() {
        let chunker = TextChunker::new(
            ChunkingConfig {
                chunk_size: 50,
                overlap: 10,
                min_chunk_size: 10,
                max_chunk_size: 100,
            },
            ChunkingStrategy::Sentence,
        ).unwrap();

        let text = "First sentence. Second sentence! Third sentence? Fourth sentence.";
        let chunks = chunker.chunk_text(text, Some("test.txt".to_string())).unwrap();
        
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].source, Some("test.txt".to_string()));
    }

    #[test]
    fn test_small_text() {
        let chunker = TextChunker::with_default_config().unwrap();
        let text = "Short text";
        let chunks = chunker.chunk_text(text, None).unwrap();
        
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, text);
    }

    #[test]
    fn test_chunk_metadata() {
        let chunk = ChunkMetadata {
            id: 1,
            text: "Test chunk".to_string(),
            source: Some("test.txt".to_string()),
            page: Some(1),
            offset: 0,
            length: 10,
            frame: None,
            embedding: None,
        };

        // Test serialization
        let json = serde_json::to_string(&chunk).unwrap();
        let deserialized: ChunkMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(chunk, deserialized);
    }
} 