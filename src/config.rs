//! Configuration management for memvid-rs
//!
//! This module provides comprehensive configuration options for all memvid operations,
//! including video encoding, QR code generation, ML models, and search parameters.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main configuration structure for memvid operations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// Text chunking configuration
    pub chunking: ChunkingConfig,

    /// QR code generation configuration
    pub qr: QrConfig,

    /// Video encoding configuration
    pub video: VideoConfig,

    /// Machine learning configuration
    pub ml: MlConfig,

    /// Search configuration
    pub search: SearchConfig,

    /// Storage configuration
    pub storage: StorageConfig,
}

/// Text chunking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Default chunk size in characters
    pub chunk_size: usize,

    /// Overlap between chunks in characters
    pub overlap: usize,

    /// Minimum chunk size
    pub min_chunk_size: usize,

    /// Maximum chunk size
    pub max_chunk_size: usize,
}

/// QR code generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QrConfig {
    /// QR code version (1-40, None for auto)
    pub version: Option<i16>,

    /// Error correction level
    pub error_correction: ErrorCorrectionLevel,

    /// Box size for each QR module
    pub box_size: u32,

    /// Border size around QR code
    pub border: u32,

    /// Fill color (black modules)
    pub fill_color: String,

    /// Background color (white modules)
    pub back_color: String,

    /// Enable compression for large data
    pub enable_compression: bool,

    /// Compression threshold in bytes
    pub compression_threshold: usize,
}

/// QR error correction levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorCorrectionLevel {
    Low,      // ~7%
    Medium,   // ~15%
    Quartile, // ~25%
    High,     // ~30%
}

/// Video encoding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoConfig {
    /// Video codec
    pub codec: String,

    /// Frames per second
    pub fps: f64,

    /// Frame width in pixels
    pub frame_width: u32,

    /// Frame height in pixels
    pub frame_height: u32,

    /// Video quality/bitrate parameters
    pub quality_params: HashMap<String, String>,

    /// Enable hardware acceleration
    pub hardware_acceleration: bool,
}

/// Machine learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlConfig {
    /// Embedding model name or path
    pub model_name: String,

    /// Device preference (auto, cpu, cuda, metal)
    pub device: String,

    /// Maximum sequence length for tokenization
    pub max_sequence_length: usize,

    /// Batch size for embedding generation
    pub batch_size: usize,

    /// Model cache directory
    pub cache_dir: Option<String>,

    /// Enable model quantization for smaller memory footprint
    pub quantization: bool,
}

/// Search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Vector search engine (hnsw, flat, auto)
    pub engine: String,

    /// HNSW parameters
    pub hnsw: HnswConfig,

    /// Maximum results to return
    pub max_results: usize,

    /// Minimum similarity score threshold
    pub min_score_threshold: f32,

    /// Enable result re-ranking
    pub enable_reranking: bool,

    /// Cache size for decoded frames
    pub cache_size: usize,

    /// Maximum parallel workers for frame decoding
    pub max_workers: usize,
}

/// HNSW (Hierarchical Navigable Small World) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum number of connections for each node
    pub max_connections: usize,

    /// Size of the dynamic candidate list during construction
    pub ef_construction: usize,

    /// Size of the dynamic candidate list during search
    pub ef_search: usize,

    /// Random seed for reproducible results
    pub seed: Option<u64>,
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Database file path
    pub database_path: Option<String>,

    /// Enable WAL mode for SQLite
    pub enable_wal_mode: bool,

    /// SQLite cache size (in pages)
    pub cache_size: i64,

    /// Enable foreign key constraints
    pub foreign_keys: bool,

    /// Synchronous mode for SQLite
    pub synchronous: String,

    /// Index file format version
    pub index_format_version: u32,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024,
            overlap: 32,
            min_chunk_size: 100,
            max_chunk_size: 4096,
        }
    }
}

impl Default for QrConfig {
    fn default() -> Self {
        Self {
            version: None, // Auto-detect
            error_correction: ErrorCorrectionLevel::Medium,
            box_size: 10,
            border: 4,
            fill_color: "black".to_string(),
            back_color: "white".to_string(),
            enable_compression: true,
            compression_threshold: 100,
        }
    }
}

impl Default for VideoConfig {
    fn default() -> Self {
        let mut quality_params = HashMap::new();
        // H.265 parameters for QR code preservation (using compatible options)
        quality_params.insert("crf".to_string(), "28".to_string());
        quality_params.insert("preset".to_string(), "slower".to_string());
        quality_params.insert("tune".to_string(), "zerolatency".to_string()); // Use zerolatency instead of stillimage
        quality_params.insert("profile".to_string(), "main".to_string());
        quality_params.insert("pix_fmt".to_string(), "yuv420p".to_string());

        Self {
            codec: "libx265".to_string(), // H.265 exactly like Python
            fps: 30.0,                    // Python: video_fps: 30
            frame_width: 256,             // Python: frame_width: 256
            frame_height: 256,            // Python: frame_height: 256
            quality_params,
            hardware_acceleration: true,
        }
    }
}

impl Default for MlConfig {
    fn default() -> Self {
        Self {
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            device: "auto".to_string(),
            max_sequence_length: 512,
            batch_size: 32,
            cache_dir: None,
            quantization: false,
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            engine: "auto".to_string(),
            hnsw: HnswConfig::default(),
            max_results: 100,
            min_score_threshold: 0.0,
            enable_reranking: false,
            cache_size: 1000,
            max_workers: 4,
        }
    }
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            max_connections: 16,
            ef_construction: 200,
            ef_search: 50,
            seed: None,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            database_path: None,
            enable_wal_mode: true,
            cache_size: 10000, // 10,000 pages = ~40MB cache
            foreign_keys: true,
            synchronous: "NORMAL".to_string(),
            index_format_version: 1,
        }
    }
}

/// Convert error correction level to qrcode crate constant
impl From<ErrorCorrectionLevel> for qrcode::EcLevel {
    fn from(level: ErrorCorrectionLevel) -> Self {
        match level {
            ErrorCorrectionLevel::Low => qrcode::EcLevel::L,
            ErrorCorrectionLevel::Medium => qrcode::EcLevel::M,
            ErrorCorrectionLevel::Quartile => qrcode::EcLevel::Q,
            ErrorCorrectionLevel::High => qrcode::EcLevel::H,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.chunking.chunk_size, 1024);
        assert_eq!(config.qr.box_size, 10);
        assert_eq!(config.video.fps, 30.0);
        assert_eq!(
            config.ml.model_name,
            "sentence-transformers/all-MiniLM-L6-v2"
        );
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: Config = serde_json::from_str(&json).unwrap();

        assert_eq!(config.chunking.chunk_size, deserialized.chunking.chunk_size);
        assert_eq!(config.ml.model_name, deserialized.ml.model_name);
    }

    #[test]
    fn test_error_correction_conversion() {
        let level = ErrorCorrectionLevel::High;
        let qr_level: qrcode::EcLevel = level.into();
        assert_eq!(qr_level, qrcode::EcLevel::H);
    }
}
