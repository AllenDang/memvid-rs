//! Error types for memvid-rs
//!
//! This module provides comprehensive error handling for all memvid operations,
//! including video processing, QR encoding/decoding, ML operations, and storage.

use thiserror::Error;

/// Main error type for memvid operations
#[derive(Error, Debug)]
pub enum MemvidError {
    /// Text processing errors
    #[error("Text processing error: {0}")]
    TextProcessing(String),

    /// QR code encoding/decoding errors
    #[error("QR code error: {0}")]
    QrCode(String),

    /// Video processing errors
    #[error("Video processing error: {0}")]
    Video(String),

    /// Machine learning model errors
    #[error("ML model error: {0}")]
    MachineLearning(String),

    /// Vector search errors
    #[error("Search error: {0}")]
    Search(String),

    /// Database/storage errors
    #[error("Storage error: {0}")]
    Storage(String),

    /// PDF processing errors
    #[error("PDF processing error: {0}")]
    Pdf(String),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// SQLite database errors
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// Candle ML framework errors
    #[error("Candle ML error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Image processing errors
    #[error("Image processing error: {0}")]
    Image(String),

    /// Generic errors
    #[error("Generic error: {0}")]
    Generic(String),
}

/// Result type alias for memvid operations
pub type Result<T> = std::result::Result<T, MemvidError>;

// Implement From traits for external error types
impl From<image::ImageError> for MemvidError {
    fn from(err: image::ImageError) -> Self {
        MemvidError::Image(err.to_string())
    }
}

impl From<qrcode::types::QrError> for MemvidError {
    fn from(err: qrcode::types::QrError) -> Self {
        MemvidError::QrCode(err.to_string())
    }
}

impl From<anyhow::Error> for MemvidError {
    fn from(err: anyhow::Error) -> Self {
        MemvidError::Generic(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let error = MemvidError::TextProcessing("test error".to_string());
        assert_eq!(error.to_string(), "Text processing error: test error");
    }

    #[test]
    fn test_error_chain() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let memvid_error = MemvidError::from(io_error);
        
        match memvid_error {
            MemvidError::Io(_) => (),
            _ => panic!("Expected Io error"),
        }
    }
} 