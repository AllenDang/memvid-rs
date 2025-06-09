//! API layer for memvid-rs
//!
//! This module provides the main public API interfaces for encoding and retrieving
//! text data from QR code videos.

pub mod encoder;
pub mod retriever;
pub mod chat;

// Re-export main API types
pub use encoder::MemvidEncoder;
pub use retriever::{MemvidRetriever, SearchResult};
pub use chat::{quick_chat, chat_with_memory, quick_chat_with_config, chat_with_memory_config}; 