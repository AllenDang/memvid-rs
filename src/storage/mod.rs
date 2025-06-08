//! Storage functionality for memvid-rs
//!
//! This module provides database operations using embedded SQLite.

pub mod database;
pub mod schema;
pub mod migrations;

// Re-export main types
pub use database::Database;

/// Encoding statistics
#[derive(Debug, Clone)]
pub struct EncodingStats {
    /// Total number of chunks processed
    pub total_chunks: usize,
    
    /// Total number of frames created
    pub total_frames: usize,
    
    /// Total processing time in seconds
    pub processing_time: f64,
    
    /// Video file size in bytes
    pub video_file_size: u64,
} 