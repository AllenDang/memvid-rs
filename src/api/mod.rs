//! API layer for memvid-rs
//!
//! This module provides the main public API interfaces for encoding and retrieving
//! text data from QR code videos.

pub mod chat;
pub mod encoder;
pub mod retriever;

// Re-export main API types
pub use chat::{chat_with_memory, chat_with_memory_config, quick_chat, quick_chat_with_config};
pub use encoder::MemvidEncoder;
pub use retriever::{MemvidRetriever, SearchResult};
