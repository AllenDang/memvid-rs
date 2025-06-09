//! QR code decoding functionality
//!
//! This module provides QR code decoding with automatic decompression
//! for extracting text content from video frames.

use crate::error::{MemvidError, Result};
use base64::{Engine as _, engine::general_purpose};
use flate2::read::GzDecoder;
use image::DynamicImage;
use std::io::Read;

/// QR code decoder with decompression support
pub struct QrDecoder;

/// Result of QR code decoding
#[derive(Debug, Clone)]
pub struct DecodeResult {
    /// The decoded text content
    pub text: String,

    /// Whether the data was compressed
    pub was_compressed: bool,

    /// Size of the encoded data before decompression
    pub encoded_size: usize,
}

impl QrDecoder {
    /// Create a new QR decoder
    pub fn new() -> Self {
        Self
    }

    /// Decode QR code from image data
    pub fn decode_image(&self, image: &DynamicImage) -> Result<DecodeResult> {
        // Convert to grayscale for better QR detection
        let gray_image = image.to_luma8();

        // Convert to rqrr format
        let mut img = rqrr::PreparedImage::prepare(gray_image);

        // Find and decode QR codes
        let grids = img.detect_grids();

        if grids.is_empty() {
            return Err(MemvidError::QrCode("No QR code found in image".to_string()));
        }

        // Decode the first QR code found
        let grid = &grids[0];
        let (_meta, content) = grid.decode()?;

        let encoded_data = content;

        let encoded_size = encoded_data.len();

        // Check if data is compressed and decompress if needed
        let (text, was_compressed) = self.process_decoded_data(&encoded_data)?;

        Ok(DecodeResult {
            text,
            was_compressed,
            encoded_size,
        })
    }

    /// Decode QR code from raw image bytes
    pub fn decode_bytes(&self, image_bytes: &[u8]) -> Result<DecodeResult> {
        let image = image::load_from_memory(image_bytes)
            .map_err(|e| MemvidError::Image(format!("Failed to load image: {}", e)))?;

        self.decode_image(&image)
    }

    /// Decode multiple QR codes from images
    pub fn decode_batch(&self, images: &[DynamicImage]) -> Vec<Result<DecodeResult>> {
        images.iter().map(|img| self.decode_image(img)).collect()
    }

    /// Process decoded data (handle decompression if needed)
    fn process_decoded_data(&self, data: &str) -> Result<(String, bool)> {
        // Check if data has compression prefix
        if let Some(compressed_data) = data.strip_prefix("GZ:") {
            // Remove "GZ:" prefix
            let decompressed = self.decompress_data(compressed_data)?;
            Ok((decompressed, true))
        } else {
            Ok((data.to_string(), false))
        }
    }

    /// Decompress base64-encoded gzip data
    fn decompress_data(&self, base64_data: &str) -> Result<String> {
        // Decode base64
        let compressed_bytes = general_purpose::STANDARD
            .decode(base64_data)
            .map_err(|e| MemvidError::QrCode(format!("Base64 decode failed: {}", e)))?;

        // Decompress gzip
        let mut decoder = GzDecoder::new(&compressed_bytes[..]);
        let mut decompressed = String::new();
        decoder
            .read_to_string(&mut decompressed)
            .map_err(|e| MemvidError::QrCode(format!("Decompression failed: {}", e)))?;

        Ok(decompressed)
    }

    /// Try to decode QR code with multiple preprocessing strategies
    pub fn decode_with_preprocessing(&self, image: &DynamicImage) -> Result<DecodeResult> {
        // Try direct decoding first
        if let Ok(result) = self.decode_image(image) {
            return Ok(result);
        }

        // Try with contrast enhancement
        if let Ok(result) = self.decode_with_contrast_enhancement(image) {
            return Ok(result);
        }

        // Try with different scaling
        if let Ok(result) = self.decode_with_scaling(image) {
            return Ok(result);
        }

        Err(MemvidError::QrCode(
            "Failed to decode QR code with all strategies".to_string(),
        ))
    }

    /// Decode with contrast enhancement
    fn decode_with_contrast_enhancement(&self, image: &DynamicImage) -> Result<DecodeResult> {
        use imageproc::contrast::*;

        let gray_image = image.to_luma8();
        let enhanced = stretch_contrast(&gray_image, 0, 255, 0, 255);
        let enhanced_dynamic = DynamicImage::ImageLuma8(enhanced);

        self.decode_image(&enhanced_dynamic)
    }

    /// Decode with different scaling factors
    fn decode_with_scaling(&self, image: &DynamicImage) -> Result<DecodeResult> {
        let scale_factors = [0.5, 1.5, 2.0, 0.75, 1.25];

        for &scale in &scale_factors {
            let (new_width, new_height) = (
                (image.width() as f32 * scale) as u32,
                (image.height() as f32 * scale) as u32,
            );

            if new_width > 0 && new_height > 0 {
                let resized = image.resize_exact(
                    new_width,
                    new_height,
                    image::imageops::FilterType::Lanczos3,
                );
                if let Ok(result) = self.decode_image(&resized) {
                    return Ok(result);
                }
            }
        }

        Err(MemvidError::QrCode(
            "Failed to decode with scaling".to_string(),
        ))
    }

    /// Validate that decoded text is reasonable
    pub fn validate_decoded_text(&self, text: &str) -> bool {
        // Basic validation: not empty, reasonable length, valid UTF-8
        !text.is_empty() &&
        text.len() < 100_000 && // Reasonable size limit
        text.chars().all(|c| c.is_ascii() || c.is_alphabetic() || c.is_numeric() || c.is_whitespace())
    }
}

impl Default for QrDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// Convert rqrr error to MemvidError
impl From<rqrr::DeQRError> for MemvidError {
    fn from(error: rqrr::DeQRError) -> Self {
        MemvidError::QrCode(format!("QR decode error: {:?}", error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QrConfig;
    use crate::qr::encoder::QrEncoder;

    fn create_test_qr_image(text: &str) -> DynamicImage {
        let mut config = QrConfig::default();
        config.box_size = 10; // Make QR code larger for better decoding
        config.border = 4; // Ensure sufficient quiet zone

        let encoder = QrEncoder::new(config);
        let frame = encoder.encode_text(text).unwrap();

        // Resize the image to be larger for better decoding reliability
        let resized = frame
            .image
            .resize(200, 200, image::imageops::FilterType::Nearest);
        resized
    }

    #[test]
    fn test_simple_decode() {
        let decoder = QrDecoder::new();
        let test_text = "Hello, World!";
        let qr_image = create_test_qr_image(test_text);

        let result = decoder.decode_image(&qr_image).unwrap();
        assert_eq!(result.text, test_text);
        assert!(!result.was_compressed);
    }

    #[test]
    fn test_compressed_decode() {
        let mut config = QrConfig::default();
        config.compression_threshold = 5; // Force compression
        config.enable_compression = true;

        let encoder = QrEncoder::new(config);
        let test_text =
            "This is a longer text that should be compressed and then decompressed correctly.";
        let frame = encoder.encode_text(test_text).unwrap();

        let decoder = QrDecoder::new();
        let result = decoder.decode_image(&frame.image).unwrap();

        assert_eq!(result.text, test_text);
        // Note: compression might not always be used depending on efficiency
    }

    #[test]
    fn test_batch_decode() {
        let decoder = QrDecoder::new();
        let texts = vec!["Text 1", "Text 2", "Text 3"];
        let images: Vec<DynamicImage> = texts
            .iter()
            .map(|text| create_test_qr_image(text))
            .collect();

        let results = decoder.decode_batch(&images);
        assert_eq!(results.len(), 3);

        for (i, result) in results.iter().enumerate() {
            match result {
                Ok(decode_result) => assert_eq!(decode_result.text, texts[i]),
                Err(e) => panic!("Decode failed: {}", e),
            }
        }
    }

    #[test]
    fn test_invalid_image() {
        let decoder = QrDecoder::new();

        // Create a blank image with no QR code
        let blank_image = DynamicImage::new_luma8(100, 100);
        let result = decoder.decode_image(&blank_image);

        assert!(result.is_err());
    }

    #[test]
    fn test_text_validation() {
        let decoder = QrDecoder::new();

        assert!(decoder.validate_decoded_text("Valid text"));
        assert!(decoder.validate_decoded_text("Text with numbers 123"));
        assert!(!decoder.validate_decoded_text("")); // Empty

        // Very long text should fail
        let long_text = "a".repeat(200_000);
        assert!(!decoder.validate_decoded_text(&long_text));
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let encoder = QrEncoder::default();
        let decoder = QrDecoder::new();

        let original_texts = vec![
            "Simple text",
            "Text with special characters: !@#$%^&*()",
            "Multi-line\ntext\nwith\nbreaks",
            "Unicode text: 🚀🎯✨",
        ];

        for original in original_texts {
            let frame = encoder.encode_text(original).unwrap();
            let result = decoder.decode_image(&frame.image).unwrap();
            assert_eq!(result.text, original);
        }
    }
}
