//! Utility functions for memvid-rs
//!
//! This module provides common utility functions used throughout the project.

use crate::error::{MemvidError, Result};
use std::path::Path;

/// Get file extension from path
pub fn get_file_extension<P: AsRef<Path>>(path: P) -> Option<String> {
    path.as_ref()
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
}

/// Check if a file is a supported document format
pub fn is_supported_document<P: AsRef<Path>>(path: P) -> bool {
    match get_file_extension(path) {
        Some(ext) => matches!(ext.as_str(), "pdf" | "txt" | "md" | "markdown"),
        None => false,
    }
}

/// Format file size in human readable format
pub fn format_file_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    const THRESHOLD: f64 = 1024.0;

    if bytes == 0 {
        return "0 B".to_string();
    }

    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= THRESHOLD && unit_index < UNITS.len() - 1 {
        size /= THRESHOLD;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// Calculate progress percentage
pub fn calculate_progress(current: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        (current as f32 / total as f32) * 100.0
    }
}

/// Validate and normalize file path
pub fn normalize_path<P: AsRef<Path>>(path: P) -> Result<std::path::PathBuf> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(MemvidError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("File not found: {}", path.display()),
        )));
    }

    path.canonicalize().map_err(MemvidError::Io)
}

/// Create directory if it doesn't exist
pub fn ensure_directory<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();

    if !path.exists() {
        std::fs::create_dir_all(path).map_err(MemvidError::Io)?;
    }

    Ok(())
}

/// Get timestamp string for file naming
pub fn get_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    format!("{}", duration.as_secs())
}

/// Validate video file extension
pub fn is_video_file<P: AsRef<Path>>(path: P) -> bool {
    match get_file_extension(path) {
        Some(ext) => matches!(ext.as_str(), "mp4" | "avi" | "mov" | "mkv" | "webm"),
        None => false,
    }
}

/// Calculate optimal chunk size based on content length
pub fn calculate_optimal_chunk_size(content_length: usize, target_chunks: usize) -> usize {
    if target_chunks == 0 {
        return 1024; // Default chunk size
    }

    let calculated = content_length / target_chunks;

    // Clamp to reasonable bounds
    calculated.clamp(256, 4096)
}

/// Escape special characters for safe file naming
pub fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect::<String>()
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_file_extension() {
        assert_eq!(get_file_extension("test.pdf"), Some("pdf".to_string()));
        assert_eq!(get_file_extension("test.PDF"), Some("pdf".to_string()));
        assert_eq!(get_file_extension("test"), None);
        assert_eq!(get_file_extension("test.tar.gz"), Some("gz".to_string()));
    }

    #[test]
    fn test_supported_document() {
        assert!(is_supported_document("document.pdf"));
        assert!(is_supported_document("README.md"));
        assert!(is_supported_document("notes.txt"));
        assert!(!is_supported_document("image.jpg"));
        assert!(!is_supported_document("video.mp4"));
    }

    #[test]
    fn test_file_size_formatting() {
        assert_eq!(format_file_size(0), "0 B");
        assert_eq!(format_file_size(512), "512 B");
        assert_eq!(format_file_size(1024), "1.0 KB");
        assert_eq!(format_file_size(1536), "1.5 KB");
        assert_eq!(format_file_size(1048576), "1.0 MB");
        assert_eq!(format_file_size(1073741824), "1.0 GB");
    }

    #[test]
    fn test_progress_calculation() {
        assert_eq!(calculate_progress(0, 100), 0.0);
        assert_eq!(calculate_progress(50, 100), 50.0);
        assert_eq!(calculate_progress(100, 100), 100.0);
        assert_eq!(calculate_progress(0, 0), 0.0); // Edge case
    }

    #[test]
    fn test_normalize_path() {
        let temp_file = NamedTempFile::new().unwrap();
        let normalized = normalize_path(temp_file.path()).unwrap();
        assert!(normalized.is_absolute());
    }

    #[test]
    fn test_video_file_detection() {
        assert!(is_video_file("video.mp4"));
        assert!(is_video_file("movie.avi"));
        assert!(is_video_file("clip.MOV"));
        assert!(!is_video_file("document.pdf"));
        assert!(!is_video_file("image.jpg"));
    }

    #[test]
    fn test_chunk_size_calculation() {
        assert_eq!(calculate_optimal_chunk_size(10000, 10), 1000);
        assert_eq!(calculate_optimal_chunk_size(100, 10), 256); // Clamped to minimum
        assert_eq!(calculate_optimal_chunk_size(50000, 10), 4096); // Clamped to maximum
        assert_eq!(calculate_optimal_chunk_size(1000, 0), 1024); // Edge case
    }

    #[test]
    fn test_filename_sanitization() {
        assert_eq!(sanitize_filename("normal_file.txt"), "normal_file.txt");
        assert_eq!(
            sanitize_filename("file/with\\bad:chars*?.txt"),
            "file_with_bad_chars__.txt"
        );
        assert_eq!(
            sanitize_filename("file\nwith\tcontrol\rchars"),
            "file_with_control_chars"
        );
    }

    #[test]
    fn test_timestamp() {
        let timestamp = get_timestamp();
        assert!(!timestamp.is_empty());
        assert!(timestamp.parse::<u64>().is_ok());
    }
}
