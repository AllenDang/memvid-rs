//! Video processing functionality for memvid-rs
//!
//! This module provides video encoding and decoding capabilities using static FFmpeg
//! with pure Rust fallbacks for maximum compatibility.

pub mod decoder;
pub mod encoder;

// Re-export main types and functions
pub use decoder::{VideoDecoder, VideoInfo};
pub use encoder::VideoEncoder;
