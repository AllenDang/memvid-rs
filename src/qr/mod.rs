//! QR code generation and decoding functionality for memvid-rs
//!
//! This module provides pure Rust QR code encoding and decoding capabilities
//! with compression support for efficient data storage.

pub mod encoder;
pub mod decoder;

// Re-export main types and functions
pub use encoder::{QrEncoder, QrFrame};
pub use decoder::{QrDecoder, DecodeResult}; 