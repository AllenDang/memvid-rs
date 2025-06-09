//! QR code generation and decoding functionality for memvid-rs
//!
//! This module provides pure Rust QR code encoding and decoding capabilities
//! with compression support for efficient data storage.

pub mod decoder;
pub mod encoder;

// Re-export main types and functions
pub use decoder::{DecodeResult, QrDecoder};
pub use encoder::{QrEncoder, QrFrame};

#[cfg(test)]
mod python_equivalent_tests {
    use super::*;
    use crate::config::QrConfig;

    #[test]
    fn test_python_equivalent_qr_encode_decode() {
        // Equivalent to Python test_qr_encode_decode
        let encoder = QrEncoder::new(QrConfig::default());
        let decoder = QrDecoder::new();

        let test_data = "Test data for QR encoding and decoding round trip";

        // Encode
        let qr_frame = encoder.encode_text(test_data).unwrap();

        // Decode
        let decode_result = decoder.decode_image(&qr_frame.image).unwrap();

        // Verify round trip
        assert_eq!(decode_result.text, test_data);
    }

    #[test]
    fn test_python_equivalent_qr_round_trip_various_sizes() {
        // Test with various data sizes like Python version
        let encoder = QrEncoder::new(QrConfig::default());
        let decoder = QrDecoder::new();

        let test_cases = vec![
            "Short",
            "Medium length test data that spans multiple words",
            "This is a much longer test string that will test the QR code capabilities with more substantial content and should verify that larger amounts of data can be encoded and decoded correctly.",
        ];

        for test_data in test_cases {
            // Encode
            let qr_frame = encoder.encode_text(test_data).unwrap();

            // Decode
            let decode_result = decoder.decode_image(&qr_frame.image).unwrap();

            // Verify round trip
            assert_eq!(decode_result.text, test_data);
        }
    }

    #[test]
    fn test_python_equivalent_qr_unicode_data() {
        // Test with Unicode data
        let encoder = QrEncoder::new(QrConfig::default());
        let decoder = QrDecoder::new();

        let unicode_data = "Hello 世界! 🚀 Testing Unicode: åäö, éèê, ñ, ü";

        // Encode
        let qr_frame = encoder.encode_text(unicode_data).unwrap();

        // Decode
        let decode_result = decoder.decode_image(&qr_frame.image).unwrap();

        // Verify Unicode round trip
        assert_eq!(decode_result.text, unicode_data);
    }

    #[test]
    fn test_python_equivalent_qr_json_data() {
        // Test with JSON data like Python might
        let encoder = QrEncoder::new(QrConfig::default());
        let decoder = QrDecoder::new();

        let json_data =
            r#"{"id": 123, "text": "Sample chunk", "metadata": {"source": "test", "page": 1}}"#;

        // Encode
        let qr_frame = encoder.encode_text(json_data).unwrap();

        // Decode
        let decode_result = decoder.decode_image(&qr_frame.image).unwrap();

        // Verify JSON round trip
        assert_eq!(decode_result.text, json_data);

        // Verify it's valid JSON
        let _: serde_json::Value = serde_json::from_str(&decode_result.text).unwrap();
    }
}
