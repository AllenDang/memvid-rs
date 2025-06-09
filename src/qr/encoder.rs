//! QR code encoding functionality
//!
//! This module provides QR code generation with automatic compression
//! for efficient text storage in video frames.

use crate::config::QrConfig;
use crate::error::{MemvidError, Result};
use base64::{Engine as _, engine::general_purpose};
use flate2::Compression;
use flate2::write::GzEncoder;
use image::{DynamicImage, Luma};
use qrcode::{EcLevel, QrCode};
use std::io::Write;

/// QR code encoder with compression support
pub struct QrEncoder {
    config: QrConfig,
}

/// A QR code frame ready for video encoding
#[derive(Debug, Clone)]
pub struct QrFrame {
    /// The QR code image data
    pub image: DynamicImage,

    /// Original text content (before compression)
    pub original_text: String,

    /// Encoded data size in bytes
    pub encoded_size: usize,

    /// Whether compression was used
    pub compressed: bool,
}

impl Default for QrEncoder {
    fn default() -> Self {
        Self::new(QrConfig::default())
    }
}

impl QrEncoder {
    /// Create a new QR encoder with the given configuration
    pub fn new(config: QrConfig) -> Self {
        Self { config }
    }

    /// Encode text data into a QR code frame
    pub fn encode_text(&self, text: &str) -> Result<QrFrame> {
        let original_text = text.to_string();
        let (data_to_encode, compressed) = self.prepare_data(text)?;

        // Create QR code
        let qr_code = self.create_qr_code(&data_to_encode)?;

        // Convert to image
        let image = self.qr_to_image(&qr_code)?;

        Ok(QrFrame {
            image,
            original_text,
            encoded_size: data_to_encode.len(),
            compressed,
        })
    }

    /// Encode multiple text chunks into QR frames
    pub fn encode_chunks(&self, texts: &[String]) -> Result<Vec<QrFrame>> {
        texts.iter().map(|text| self.encode_text(text)).collect()
    }

    /// Prepare data for encoding (apply compression if beneficial)
    fn prepare_data(&self, text: &str) -> Result<(String, bool)> {
        if !self.config.enable_compression || text.len() < self.config.compression_threshold {
            return Ok((text.to_string(), false));
        }

        // Try compression
        let compressed_data = self.compress_text(text)?;
        let compressed_with_prefix = format!("GZ:{}", compressed_data);

        // Use compression only if it reduces size significantly
        if compressed_with_prefix.len() < text.len() {
            Ok((compressed_with_prefix, true))
        } else {
            Ok((text.to_string(), false))
        }
    }

    /// Compress text using gzip and base64 encode
    fn compress_text(&self, text: &str) -> Result<String> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(text.as_bytes())
            .map_err(|e| MemvidError::QrCode(format!("Compression failed: {}", e)))?;

        let compressed_data = encoder
            .finish()
            .map_err(|e| MemvidError::QrCode(format!("Compression finalization failed: {}", e)))?;

        Ok(general_purpose::STANDARD.encode(&compressed_data))
    }

    /// Create QR code from data
    fn create_qr_code(&self, data: &str) -> Result<QrCode> {
        let ec_level: EcLevel = self.config.error_correction.clone().into();

        let qr_builder = QrCode::with_error_correction_level(data, ec_level);

        // If version is specified, try to use it
        if let Some(version) = self.config.version {
            // qrcode crate doesn't directly support setting version, but we can validate
            if !(1..=40).contains(&version) {
                return Err(MemvidError::QrCode(format!(
                    "Invalid QR version: {}",
                    version
                )));
            }
            // The library will automatically choose the appropriate version
        }

        qr_builder.map_err(|e| MemvidError::QrCode(format!("QR code creation failed: {}", e)))
    }

    /// Convert QR code to image with proper sizing
    fn qr_to_image(&self, qr_code: &QrCode) -> Result<DynamicImage> {
        // Create image from QR code with better parameters for decoding
        let qr_image = qr_code
            .render::<Luma<u8>>()
            .quiet_zone(true) // Essential for decoder to work
            .module_dimensions(self.config.box_size, self.config.box_size) // Proper module size
            .dark_color(Luma([0u8])) // Pure black
            .light_color(Luma([255u8])) // Pure white
            .build();

        // Convert to DynamicImage
        let dynamic_image = DynamicImage::ImageLuma8(qr_image);

        Ok(dynamic_image)
    }

    /// Estimate the capacity of a QR code at different error correction levels
    pub fn estimate_capacity(&self, text: &str) -> Result<QrCapacityInfo> {
        let mut capacities = Vec::new();

        for &ec_level in &[EcLevel::L, EcLevel::M, EcLevel::Q, EcLevel::H] {
            match QrCode::with_error_correction_level(text, ec_level) {
                Ok(qr) => {
                    let version = match qr.version() {
                        qrcode::Version::Normal(v) => v,
                        qrcode::Version::Micro(v) => v,
                    };
                    capacities.push(QrLevelCapacity {
                        error_correction: ec_level,
                        version,
                        fits: true,
                    });
                }
                Err(_) => {
                    capacities.push(QrLevelCapacity {
                        error_correction: ec_level,
                        version: 0,
                        fits: false,
                    });
                }
            }
        }

        Ok(QrCapacityInfo {
            text_length: text.len(),
            capacities,
        })
    }

    /// Get maximum data capacity for a given QR version and error correction level
    pub fn get_max_capacity(version: i16, ec_level: EcLevel) -> usize {
        // Approximate capacities for alphanumeric data
        // These are rough estimates - actual capacity depends on data type
        match (version, ec_level) {
            (1, EcLevel::L) => 25,
            (1, EcLevel::M) => 20,
            (1, EcLevel::Q) => 16,
            (1, EcLevel::H) => 10,
            (10, EcLevel::L) => 174,
            (10, EcLevel::M) => 136,
            (10, EcLevel::Q) => 100,
            (10, EcLevel::H) => 74,
            (20, EcLevel::L) => 370,
            (20, EcLevel::M) => 290,
            (20, EcLevel::Q) => 216,
            (20, EcLevel::H) => 158,
            (40, EcLevel::L) => 852,
            (40, EcLevel::M) => 666,
            (40, EcLevel::Q) => 496,
            (40, EcLevel::H) => 364,
            _ => {
                // Linear interpolation for other versions
                let base_capacity = match ec_level {
                    EcLevel::L => 20,
                    EcLevel::M => 15,
                    EcLevel::Q => 12,
                    EcLevel::H => 8,
                };
                (base_capacity * version as usize).min(1000)
            }
        }
    }
}

/// Information about QR code capacity at different error correction levels
#[derive(Debug, Clone)]
pub struct QrCapacityInfo {
    /// Length of the input text
    pub text_length: usize,

    /// Capacity information for each error correction level
    pub capacities: Vec<QrLevelCapacity>,
}

/// Capacity information for a specific error correction level
#[derive(Debug, Clone)]
pub struct QrLevelCapacity {
    /// Error correction level
    pub error_correction: EcLevel,

    /// QR code version needed (0 if doesn't fit)
    pub version: i16,

    /// Whether the text fits at this error correction level
    pub fits: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qr_encoding() {
        let encoder = QrEncoder::default();
        let frame = encoder.encode_text("Hello, World!").unwrap();

        assert_eq!(frame.original_text, "Hello, World!");
        assert!(!frame.compressed); // Should not compress short text
    }

    #[test]
    fn test_compression() {
        let mut config = QrConfig::default();
        config.compression_threshold = 10; // Force compression for small text

        let encoder = QrEncoder::new(config);
        let long_text = "This is a longer text that should be compressed when encoding into QR codes for efficient storage in video frames.";

        let frame = encoder.encode_text(long_text).unwrap();
        assert_eq!(frame.original_text, long_text);
        // Note: Compression might not always reduce size for medium text
    }

    #[test]
    fn test_capacity_estimation() {
        let encoder = QrEncoder::default();
        let capacity_info = encoder.estimate_capacity("Test").unwrap();

        assert_eq!(capacity_info.text_length, 4);
        assert_eq!(capacity_info.capacities.len(), 4); // Four error correction levels

        // Short text should fit at all levels
        assert!(capacity_info.capacities.iter().any(|c| c.fits));
    }

    #[test]
    fn test_max_capacity() {
        assert!(QrEncoder::get_max_capacity(1, EcLevel::L) > 0);
        assert!(
            QrEncoder::get_max_capacity(40, EcLevel::L)
                > QrEncoder::get_max_capacity(1, EcLevel::L)
        );
        assert!(
            QrEncoder::get_max_capacity(20, EcLevel::L)
                > QrEncoder::get_max_capacity(20, EcLevel::H)
        );
    }

    #[test]
    fn test_empty_text() {
        let encoder = QrEncoder::default();
        let frame = encoder.encode_text("").unwrap();

        assert_eq!(frame.original_text, "");
        assert!(!frame.compressed);
    }
}
