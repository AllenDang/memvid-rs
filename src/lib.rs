//! # memvid-rs
//!
//! A high-performance, self-contained Rust implementation of memvid, encoding text documents 
//! as QR codes within video files for efficient storage and semantic retrieval.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use memvid_rs::{MemvidEncoder, MemvidRetriever, Config};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create an encoder with default settings
//!     let mut encoder = MemvidEncoder::new(None).await?;
//!     
//!     // Add content from various sources
//!     encoder.add_text("Your text content here", 1024, 32).await?;
//!     
//!     // Build the video memory
//!     let stats = encoder.build_video("memory.mp4", "index.json").await?;
//!     println!("Encoded {} chunks into video", stats.total_chunks);
//!     
//!     // Query your video memory
//!     let retriever = MemvidRetriever::new("memory.mp4", "index.json").await?;
//!     let results = retriever.search("your query", 5).await?;
//!     
//!     for (score, text) in results {
//!         println!("Score: {:.3} - {}", score, text);
//!     }
//!     
//!     Ok(())
//! }
//! ```

// Core modules
pub mod config;
pub mod error;
pub mod qr;
pub mod video;
pub mod text;
pub mod ml;
pub mod storage;
pub mod api;
pub mod utils;

// Re-export main API types
pub use api::{MemvidEncoder, MemvidRetriever, SearchResult};
pub use config::Config;
pub use error::{MemvidError, Result};

// Re-export commonly used types
pub use text::ChunkMetadata;
pub use storage::EncodingStats;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_imports() {
        // Ensure all major types can be imported
        let _config = Config::default();
    }
}
