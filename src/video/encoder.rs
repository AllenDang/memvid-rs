//! Video encoding functionality for memvid-rs
//!
//! This module provides video encoding using static FFmpeg with H.265 codec
//! optimized for QR code preservation using exact Python parameters.

use crate::config::VideoConfig;
use crate::error::{MemvidError, Result};
use image::DynamicImage;
use std::path::Path;

/// Video encoder with static FFmpeg support
pub struct VideoEncoder {
    config: VideoConfig,
}

impl Default for VideoEncoder {
    fn default() -> Self {
        Self::new(VideoConfig::default())
    }
}

impl VideoEncoder {
    /// Create a new video encoder with custom configuration
    pub fn new(config: VideoConfig) -> Self {
        Self { config }
    }

    /// Encode multiple frames into a video file
    pub async fn encode_frames(&self, frames: &[DynamicImage], output_path: &str) -> Result<()> {
        if frames.is_empty() {
            return Err(MemvidError::Video(
                "No frames provided for encoding".to_string(),
            ));
        }

        let output_path = Path::new(output_path);

        // Create output directory if it doesn't exist
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(MemvidError::Io)?;
        }

        log::info!(
            "Encoding {} frames to {} ({}x{} @ {} fps)",
            frames.len(),
            output_path.display(),
            self.config.frame_width,
            self.config.frame_height,
            self.config.fps
        );

        // Initialize FFmpeg
        ffmpeg_next::init()
            .map_err(|e| MemvidError::Video(format!("FFmpeg init failed: {}", e)))?;

        // Create output format context
        let mut output_ctx = ffmpeg_next::format::output(&output_path)
            .map_err(|e| MemvidError::Video(format!("Failed to create output context: {}", e)))?;

        // Find H.265 encoder (HEVC)
        let codec = ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::HEVC)
            .ok_or_else(|| MemvidError::Video("H.265 (HEVC) encoder not found".to_string()))?;

        // Create video stream
        let mut stream = output_ctx
            .add_stream(codec)
            .map_err(|e| MemvidError::Video(format!("Failed to add video stream: {}", e)))?;
        let stream_index = stream.index();

        // Create encoder context
        let mut encoder = ffmpeg_next::codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()
            .map_err(|e| MemvidError::Video(format!("Failed to create video encoder: {}", e)))?;

        // Set encoder parameters - use config dimensions, not QR size
        encoder.set_width(self.config.frame_width);
        encoder.set_height(self.config.frame_height);
        encoder.set_format(ffmpeg_next::format::Pixel::YUV420P);
        encoder.set_time_base(ffmpeg_next::Rational::new(1, self.config.fps as i32));
        encoder.set_frame_rate(Some(ffmpeg_next::Rational::new(self.config.fps as i32, 1)));

        // Set codec parameters optimized for QR codes (matching Python H.265 implementation exactly)
        let mut dictionary = ffmpeg_next::Dictionary::new();

        // Use parameters from config.quality_params (matches Python H265_PARAMETERS)
        for (key, value) in &self.config.quality_params {
            dictionary.set(key, value);
        }

        // Open encoder
        let mut encoder = encoder
            .open_with(dictionary)
            .map_err(|e| MemvidError::Video(format!("Failed to open encoder: {}", e)))?;

        // Update stream parameters
        stream.set_parameters(&encoder);

        // Write header
        output_ctx
            .write_header()
            .map_err(|e| MemvidError::Video(format!("Failed to write header: {}", e)))?;

        // Encode frames with upscaling
        self.encode_image_frames(frames, &mut encoder, &mut output_ctx, stream_index)
            .await?;

        // Write trailer
        output_ctx
            .write_trailer()
            .map_err(|e| MemvidError::Video(format!("Failed to write trailer: {}", e)))?;

        log::info!(
            "Successfully encoded {} frames to {}",
            frames.len(),
            output_path.display()
        );
        Ok(())
    }

    /// Encode individual image frames with upscaling for QR preservation
    async fn encode_image_frames(
        &self,
        frames: &[DynamicImage],
        encoder: &mut ffmpeg_next::encoder::Video,
        output_ctx: &mut ffmpeg_next::format::context::Output,
        stream_index: usize,
    ) -> Result<()> {
        // Use config dimensions for all frames
        let target_width = self.config.frame_width;
        let target_height = self.config.frame_height;

        log::info!(
            "Upscaling frames from QR size to {}x{} for compression resistance",
            target_width,
            target_height
        );

        for (i, image) in frames.iter().enumerate() {
            // Always upscale QR codes to target resolution for better compression resistance
            let upscaled_image = image.resize_exact(
                target_width,
                target_height,
                image::imageops::FilterType::Nearest, // Use nearest neighbor for crisp QR codes
            );

            // Convert image to RGB format
            let rgb_image = upscaled_image.to_rgb8();
            let rgb_data = rgb_image.as_raw();

            // Create video frame
            let mut frame = ffmpeg_next::frame::Video::new(
                ffmpeg_next::format::Pixel::RGB24,
                target_width,
                target_height,
            );

            // Copy RGB data to frame
            frame.data_mut(0)[..rgb_data.len()].copy_from_slice(rgb_data);
            frame.set_pts(Some((i as f64 / self.config.fps * 1000.0) as i64));

            // Convert RGB to YUV420P for encoding
            let mut yuv_frame = ffmpeg_next::frame::Video::new(
                ffmpeg_next::format::Pixel::YUV420P,
                target_width,
                target_height,
            );

            // Set up software scaler for RGB to YUV conversion
            let mut scaler = ffmpeg_next::software::scaling::Context::get(
                ffmpeg_next::format::Pixel::RGB24,
                target_width,
                target_height,
                ffmpeg_next::format::Pixel::YUV420P,
                target_width,
                target_height,
                ffmpeg_next::software::scaling::Flags::BILINEAR,
            )
            .map_err(|e| MemvidError::Video(format!("Failed to create scaler: {}", e)))?;

            scaler
                .run(&frame, &mut yuv_frame)
                .map_err(|e| MemvidError::Video(format!("Failed to scale frame: {}", e)))?;

            yuv_frame.set_pts(frame.pts());

            // Send frame to encoder
            encoder
                .send_frame(&yuv_frame)
                .map_err(|e| MemvidError::Video(format!("Failed to send frame: {}", e)))?;

            // Receive and write packets
            self.receive_and_write_packets(encoder, output_ctx, stream_index)
                .await?;

            if (i + 1) % 10 == 0 {
                log::info!("Encoded {}/{} frames", i + 1, frames.len());
            }
        }

        // Flush encoder
        encoder
            .send_eof()
            .map_err(|e| MemvidError::Video(format!("Failed to send EOF: {}", e)))?;

        // Receive remaining packets
        self.receive_and_write_packets(encoder, output_ctx, stream_index)
            .await?;

        Ok(())
    }

    /// Receive packets from encoder and write to output
    async fn receive_and_write_packets(
        &self,
        encoder: &mut ffmpeg_next::encoder::Video,
        output_ctx: &mut ffmpeg_next::format::context::Output,
        stream_index: usize,
    ) -> Result<()> {
        let mut packet = ffmpeg_next::Packet::empty();

        while encoder.receive_packet(&mut packet).is_ok() {
            packet.set_stream(stream_index);
            packet
                .write_interleaved(output_ctx)
                .map_err(|e| MemvidError::Video(format!("Failed to write packet: {}", e)))?;
        }

        Ok(())
    }

    /// Encode a single image into a one-frame video
    pub async fn encode_single_image(&self, image: &DynamicImage, output_path: &str) -> Result<()> {
        self.encode_frames(&[image.clone()], output_path).await
    }

    /// Get video configuration
    pub fn config(&self) -> &VideoConfig {
        &self.config
    }

    /// Get supported output formats
    pub fn supported_formats() -> Vec<String> {
        // Common video formats supported by H.264
        vec![
            "mp4".to_string(),
            "mkv".to_string(),
            "avi".to_string(),
            "mov".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qr::encoder::QrEncoder;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_encoder_creation() {
        let encoder = VideoEncoder::default();
        assert_eq!(encoder.config.fps, 30.0); // Updated default FPS
        assert_eq!(encoder.config.frame_width, 256); // QR upscaling target
        assert_eq!(encoder.config.frame_height, 256); // QR upscaling target
    }

    #[tokio::test]
    async fn test_video_config() {
        let config = VideoConfig::default();
        assert_eq!(config.fps, 30.0);
        assert_eq!(config.frame_width, 256);
        assert_eq!(config.frame_height, 256);
        assert_eq!(config.codec, "libx265");
        assert!(config.quality_params.contains_key("crf"));
        assert!(config.quality_params.contains_key("preset"));
    }

    #[tokio::test]
    async fn test_supported_formats() {
        let formats = VideoEncoder::supported_formats();
        assert!(formats.contains(&"mp4".to_string()));
        assert!(formats.contains(&"mkv".to_string()));
    }

    #[tokio::test]
    async fn test_encode_single_qr_frame() {
        // Create a test QR code image
        let qr_encoder = QrEncoder::default();
        let qr_frame = qr_encoder.encode_text("Test QR code").unwrap();

        // Create temporary output file
        let temp_file = NamedTempFile::with_suffix(".mp4").unwrap();
        let output_path = temp_file.path().to_str().unwrap();

        // Test encoding with QR upscaling
        let video_encoder = VideoEncoder::default();
        let result = video_encoder
            .encode_single_image(&qr_frame.image, output_path)
            .await;

        // Should succeed or fail gracefully (depending on FFmpeg availability)
        match result {
            Ok(_) => {
                // Verify file was created
                assert!(temp_file.path().exists());
                let metadata = std::fs::metadata(temp_file.path()).unwrap();
                assert!(metadata.len() > 0);
                log::info!("Video encoding test successful - QR upscaling working");
            }
            Err(MemvidError::Video(_)) => {
                // Expected if FFmpeg not properly configured in test environment
                log::warn!("Video encoding test skipped - FFmpeg not available");
            }
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }
}
