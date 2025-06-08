//! Video processing functionality for memvid-rs
//!
//! This module provides video encoding and decoding capabilities using static FFmpeg
//! with pure Rust fallbacks for maximum compatibility.

pub mod encoder;
pub mod decoder;

// Re-export main types and functions
pub use encoder::VideoEncoder;
pub use decoder::{VideoDecoder, VideoInfo}; 