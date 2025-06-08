//! API layer for memvid-rs
//!
//! This module provides the main public API interfaces for encoding and retrieving
//! text data from QR code videos.

pub mod encoder;
pub mod retriever;

// Re-export main API types
pub use encoder::MemvidEncoder;
pub use retriever::{MemvidRetriever, SearchResult}; 