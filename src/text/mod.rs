//! Text processing and chunking functionality for memvid-rs
//!
//! This module provides text chunking algorithms, document format support,
//! and metadata management for text content.

pub mod chunking;
pub mod pdf;

// Re-export main types and functions
pub use chunking::{ChunkMetadata, ChunkingStrategy, TextChunker};
pub use pdf::PdfProcessor; 