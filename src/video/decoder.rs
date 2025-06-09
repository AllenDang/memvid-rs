//! Video decoding functionality
//!
//! This module provides video decoding using static FFmpeg for extracting frames from memvid videos.

use crate::error::{MemvidError, Result};
use image::DynamicImage;
use std::path::Path;

/// Video decoder with static FFmpeg support for frame extraction
pub struct VideoDecoder {}

impl VideoDecoder {
    /// Create a new video decoder
    pub fn new() -> Result<Self> {
        // Initialize FFmpeg
        ffmpeg_next::init()
            .map_err(|e| MemvidError::Video(format!("FFmpeg init failed: {}", e)))?;

        Ok(Self {})
    }

    /// Extract all frames from video file
    pub async fn extract_frames(&self, video_path: &str) -> Result<Vec<DynamicImage>> {
        let path = Path::new(video_path);
        if !path.exists() {
            return Err(MemvidError::Video(format!(
                "Video file not found: {}",
                video_path
            )));
        }

        log::info!("Extracting frames from video: {}", video_path);

        // Open input video
        let mut input_ctx = ffmpeg_next::format::input(&path)
            .map_err(|e| MemvidError::Video(format!("Failed to open video file: {}", e)))?;

        // Find video stream
        let video_stream_index = input_ctx
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or_else(|| MemvidError::Video("No video stream found".to_string()))?
            .index();

        // Get video stream
        let video_stream = input_ctx
            .stream(video_stream_index)
            .ok_or_else(|| MemvidError::Video("Failed to get video stream".to_string()))?;

        // Create decoder context
        let context_decoder =
            ffmpeg_next::codec::context::Context::from_parameters(video_stream.parameters())
                .map_err(|e| {
                    MemvidError::Video(format!("Failed to create decoder context: {}", e))
                })?;

        let mut decoder = context_decoder
            .decoder()
            .video()
            .map_err(|e| MemvidError::Video(format!("Failed to create video decoder: {}", e)))?;

        // Get frame dimensions
        let width = decoder.width();
        let height = decoder.height();

        log::info!("Video dimensions: {}x{}", width, height);

        // Set up frame conversion
        let mut scaler = ffmpeg_next::software::scaling::Context::get(
            decoder.format(),
            width,
            height,
            ffmpeg_next::format::Pixel::RGB24,
            width,
            height,
            ffmpeg_next::software::scaling::Flags::BILINEAR,
        )
        .map_err(|e| MemvidError::Video(format!("Failed to create scaler: {}", e)))?;

        let mut frames = Vec::new();
        let mut frame_count = 0;

        // Process packets
        for (stream, packet) in input_ctx.packets() {
            if stream.index() == video_stream_index {
                decoder
                    .send_packet(&packet)
                    .map_err(|e| MemvidError::Video(format!("Failed to send packet: {}", e)))?;

                self.receive_frames(&mut decoder, &mut scaler, &mut frames, &mut frame_count)
                    .await?;
            }
        }

        // Flush decoder
        decoder
            .send_eof()
            .map_err(|e| MemvidError::Video(format!("Failed to send EOF: {}", e)))?;

        self.receive_frames(&mut decoder, &mut scaler, &mut frames, &mut frame_count)
            .await?;

        log::info!("Extracted {} frames from video", frames.len());
        Ok(frames)
    }

    /// Extract specific frame by number (0-indexed) - EFFICIENT VERSION
    pub async fn extract_frame(&self, video_path: &str, frame_number: u32) -> Result<DynamicImage> {
        let path = Path::new(video_path);
        if !path.exists() {
            return Err(MemvidError::Video(format!(
                "Video file not found: {}",
                video_path
            )));
        }

        log::debug!("Extracting frame {} from video: {}", frame_number, video_path);

        // Open input video
        let mut input_ctx = ffmpeg_next::format::input(&path)
            .map_err(|e| MemvidError::Video(format!("Failed to open video file: {}", e)))?;

        // Find video stream
        let video_stream_index = input_ctx
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or_else(|| MemvidError::Video("No video stream found".to_string()))?
            .index();

        // Get video stream and extract needed parameters before borrowing mutably
        let (time_base, avg_frame_rate, stream_parameters) = {
            let video_stream = input_ctx
                .stream(video_stream_index)
                .ok_or_else(|| MemvidError::Video("Failed to get video stream".to_string()))?;
            (video_stream.time_base(), video_stream.avg_frame_rate(), video_stream.parameters())
        };

        // Calculate timestamp for the target frame
        let frame_duration = if avg_frame_rate.denominator() > 0 && avg_frame_rate.numerator() > 0 {
            time_base.denominator() as f64 / (time_base.numerator() as f64 * avg_frame_rate.numerator() as f64 / avg_frame_rate.denominator() as f64)
        } else {
            // Fallback: assume 30 FPS
            time_base.denominator() as f64 / (time_base.numerator() as f64 * 30.0)
        };
        
        let target_timestamp = (frame_number as f64 * frame_duration) as i64;

        // Seek to the target frame
        if frame_number > 0 {
            input_ctx.seek(target_timestamp, ..target_timestamp + (frame_duration as i64))
                .map_err(|e| MemvidError::Video(format!("Failed to seek to frame {}: {}", frame_number, e)))?;
        }

        // Create decoder context
        let context_decoder =
            ffmpeg_next::codec::context::Context::from_parameters(stream_parameters)
                .map_err(|e| {
                    MemvidError::Video(format!("Failed to create decoder context: {}", e))
                })?;

        let mut decoder = context_decoder
            .decoder()
            .video()
            .map_err(|e| MemvidError::Video(format!("Failed to create video decoder: {}", e)))?;

        // Get frame dimensions
        let width = decoder.width();
        let height = decoder.height();

        // Set up frame conversion
        let mut scaler = ffmpeg_next::software::scaling::Context::get(
            decoder.format(),
            width,
            height,
            ffmpeg_next::format::Pixel::RGB24,
            width,
            height,
            ffmpeg_next::software::scaling::Flags::BILINEAR,
        )
        .map_err(|e| MemvidError::Video(format!("Failed to create scaler: {}", e)))?;

        let mut current_frame = 0u32;

        // Process packets until we find our target frame
        for (stream, packet) in input_ctx.packets() {
            if stream.index() == video_stream_index {
                decoder
                    .send_packet(&packet)
                    .map_err(|e| MemvidError::Video(format!("Failed to send packet: {}", e)))?;

                let mut decoded_frame = ffmpeg_next::frame::Video::empty();
                while decoder.receive_frame(&mut decoded_frame).is_ok() {
                    if current_frame == frame_number {
                        // Found our target frame, convert and return it
                        let mut rgb_frame =
                            ffmpeg_next::frame::Video::new(ffmpeg_next::format::Pixel::RGB24, width, height);

                        scaler
                            .run(&decoded_frame, &mut rgb_frame)
                            .map_err(|e| MemvidError::Video(format!("Failed to scale frame: {}", e)))?;

                        let image = self.frame_to_image(&rgb_frame)?;
                        
                        log::debug!("Successfully extracted frame {} from video", frame_number);
                        return Ok(image);
                    }
                    current_frame += 1;
                    
                    // Stop processing if we've gone past our target
                    if current_frame > frame_number {
                        break;
                    }
                }
                
                if current_frame > frame_number {
                    break;
                }
            }
        }

        Err(MemvidError::Video(format!(
            "Frame number {} not found (video has fewer frames or seeking failed)",
            frame_number
        )))
    }

    /// Extract specific frame by number (0-indexed) - DEPRECATED SLOW VERSION
    pub async fn extract_frame_slow(&self, video_path: &str, frame_number: u32) -> Result<DynamicImage> {
        let frames = self.extract_frames(video_path).await?;

        if frame_number as usize >= frames.len() {
            return Err(MemvidError::Video(format!(
                "Frame number {} out of range (video has {} frames)",
                frame_number,
                frames.len()
            )));
        }

        Ok(frames[frame_number as usize].clone())
    }

    /// Get video information without extracting frames
    pub async fn get_video_info(&self, video_path: &str) -> Result<VideoInfo> {
        let path = Path::new(video_path);
        if !path.exists() {
            return Err(MemvidError::Video(format!(
                "Video file not found: {}",
                video_path
            )));
        }

        // Open input video
        let input_ctx = ffmpeg_next::format::input(&path)
            .map_err(|e| MemvidError::Video(format!("Failed to open video file: {}", e)))?;

        // Find video stream
        let video_stream = input_ctx
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or_else(|| MemvidError::Video("No video stream found".to_string()))?;

        // Get decoder context for metadata
        let context_decoder =
            ffmpeg_next::codec::context::Context::from_parameters(video_stream.parameters())
                .map_err(|e| {
                    MemvidError::Video(format!("Failed to create decoder context: {}", e))
                })?;

        let decoder = context_decoder
            .decoder()
            .video()
            .map_err(|e| MemvidError::Video(format!("Failed to create video decoder: {}", e)))?;

        // Calculate duration and frame count
        let duration_seconds = if video_stream.duration() > 0 {
            let time_base: f64 = video_stream.time_base().into();
            video_stream.duration() as f64 * time_base
        } else {
            input_ctx.duration() as f64 / ffmpeg_next::ffi::AV_TIME_BASE as f64
        };

        let fps: f64 = video_stream.avg_frame_rate().into();
        let frame_count = if fps > 0.0 {
            (duration_seconds * fps) as u32
        } else {
            0
        };

        Ok(VideoInfo {
            width: decoder.width(),
            height: decoder.height(),
            fps,
            duration_seconds,
            frame_count,
            format: format!("{:?}", decoder.format()),
            codec: "H.264".to_string(), // Default for memvid videos
        })
    }

    /// Helper to receive and convert frames
    async fn receive_frames(
        &self,
        decoder: &mut ffmpeg_next::decoder::Video,
        scaler: &mut ffmpeg_next::software::scaling::Context,
        frames: &mut Vec<DynamicImage>,
        frame_count: &mut u32,
    ) -> Result<()> {
        let mut decoded_frame = ffmpeg_next::frame::Video::empty();

        while decoder.receive_frame(&mut decoded_frame).is_ok() {
            let width = decoded_frame.width();
            let height = decoded_frame.height();

            // Create RGB frame
            let mut rgb_frame =
                ffmpeg_next::frame::Video::new(ffmpeg_next::format::Pixel::RGB24, width, height);

            // Scale/convert to RGB
            scaler
                .run(&decoded_frame, &mut rgb_frame)
                .map_err(|e| MemvidError::Video(format!("Failed to scale frame: {}", e)))?;

            // Convert to image
            let image = self.frame_to_image(&rgb_frame)?;
            frames.push(image);

            *frame_count += 1;

            if *frame_count % 10 == 0 {
                log::info!("Processed {} frames", *frame_count);
            }
        }

        Ok(())
    }

    /// Convert FFmpeg frame to DynamicImage
    fn frame_to_image(&self, frame: &ffmpeg_next::frame::Video) -> Result<DynamicImage> {
        let width = frame.width();
        let height = frame.height();

        // Get RGB data from frame
        let data = frame.data(0);
        let linesize = frame.stride(0);

        // Create image buffer
        let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);

        for y in 0..height {
            let row_start = (y * linesize as u32) as usize;
            let row_end = row_start + (width * 3) as usize;
            rgb_data.extend_from_slice(&data[row_start..row_end]);
        }

        // Create RGB image
        let rgb_image = image::RgbImage::from_raw(width, height, rgb_data).ok_or_else(|| {
            MemvidError::Video("Failed to create RGB image from frame data".to_string())
        })?;

        Ok(DynamicImage::ImageRgb8(rgb_image))
    }

    /// Extract frames within a specific range - EFFICIENT VERSION
    pub async fn extract_frames_range(
        &self,
        video_path: &str,
        start_frame: u32,
        end_frame: u32,
    ) -> Result<Vec<DynamicImage>> {
        if start_frame > end_frame {
            return Err(MemvidError::Video(format!(
                "Invalid frame range: start {} > end {}",
                start_frame, end_frame
            )));
        }

        let path = Path::new(video_path);
        if !path.exists() {
            return Err(MemvidError::Video(format!(
                "Video file not found: {}",
                video_path
            )));
        }

        log::debug!("Extracting frames {}-{} from video: {}", start_frame, end_frame, video_path);

        // Open input video
        let mut input_ctx = ffmpeg_next::format::input(&path)
            .map_err(|e| MemvidError::Video(format!("Failed to open video file: {}", e)))?;

        // Find video stream
        let video_stream_index = input_ctx
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or_else(|| MemvidError::Video("No video stream found".to_string()))?
            .index();

        // Get video stream and extract needed parameters before borrowing mutably
        let (time_base, avg_frame_rate, stream_parameters) = {
            let video_stream = input_ctx
                .stream(video_stream_index)
                .ok_or_else(|| MemvidError::Video("Failed to get video stream".to_string()))?;
            (video_stream.time_base(), video_stream.avg_frame_rate(), video_stream.parameters())
        };

        // Calculate timestamp for the start frame
        let frame_duration = if avg_frame_rate.denominator() > 0 && avg_frame_rate.numerator() > 0 {
            time_base.denominator() as f64 / (time_base.numerator() as f64 * avg_frame_rate.numerator() as f64 / avg_frame_rate.denominator() as f64)
        } else {
            // Fallback: assume 30 FPS
            time_base.denominator() as f64 / (time_base.numerator() as f64 * 30.0)
        };
        
        let start_timestamp = (start_frame as f64 * frame_duration) as i64;

        // Seek to the start frame
        if start_frame > 0 {
            input_ctx.seek(start_timestamp, ..start_timestamp + (frame_duration as i64))
                .map_err(|e| MemvidError::Video(format!("Failed to seek to frame {}: {}", start_frame, e)))?;
        }

        // Create decoder context
        let context_decoder =
            ffmpeg_next::codec::context::Context::from_parameters(stream_parameters)
                .map_err(|e| {
                    MemvidError::Video(format!("Failed to create decoder context: {}", e))
                })?;

        let mut decoder = context_decoder
            .decoder()
            .video()
            .map_err(|e| MemvidError::Video(format!("Failed to create video decoder: {}", e)))?;

        // Get frame dimensions
        let width = decoder.width();
        let height = decoder.height();

        // Set up frame conversion
        let mut scaler = ffmpeg_next::software::scaling::Context::get(
            decoder.format(),
            width,
            height,
            ffmpeg_next::format::Pixel::RGB24,
            width,
            height,
            ffmpeg_next::software::scaling::Flags::BILINEAR,
        )
        .map_err(|e| MemvidError::Video(format!("Failed to create scaler: {}", e)))?;

        let mut current_frame = 0u32;
        let mut extracted_frames = Vec::new();

        // Process packets and collect frames in the range
        for (stream, packet) in input_ctx.packets() {
            if stream.index() == video_stream_index {
                decoder
                    .send_packet(&packet)
                    .map_err(|e| MemvidError::Video(format!("Failed to send packet: {}", e)))?;

                let mut decoded_frame = ffmpeg_next::frame::Video::empty();
                while decoder.receive_frame(&mut decoded_frame).is_ok() {
                    if current_frame >= start_frame && current_frame <= end_frame {
                        // Frame is in our target range, convert and store it
                        let mut rgb_frame =
                            ffmpeg_next::frame::Video::new(ffmpeg_next::format::Pixel::RGB24, width, height);

                        scaler
                            .run(&decoded_frame, &mut rgb_frame)
                            .map_err(|e| MemvidError::Video(format!("Failed to scale frame: {}", e)))?;

                        let image = self.frame_to_image(&rgb_frame)?;
                        extracted_frames.push(image);
                    }
                    
                    current_frame += 1;
                    
                    // Stop processing if we've gone past our target range
                    if current_frame > end_frame {
                        log::debug!("Successfully extracted {} frames ({}-{})", extracted_frames.len(), start_frame, end_frame);
                        return Ok(extracted_frames);
                    }
                }
                
                // If we've collected all frames in range, stop processing
                if current_frame > end_frame {
                    break;
                }
            }
        }

        log::debug!("Successfully extracted {} frames ({}-{})", extracted_frames.len(), start_frame, end_frame);
        Ok(extracted_frames)
    }

    /// Extract frames within a specific range - DEPRECATED SLOW VERSION 
    pub async fn extract_frames_range_slow(
        &self,
        video_path: &str,
        start_frame: u32,
        end_frame: u32,
    ) -> Result<Vec<DynamicImage>> {
        let all_frames = self.extract_frames(video_path).await?;

        let start_idx = start_frame as usize;
        let end_idx = (end_frame + 1) as usize;

        if start_idx >= all_frames.len() {
            return Err(MemvidError::Video(format!(
                "Start frame {} out of range (video has {} frames)",
                start_frame,
                all_frames.len()
            )));
        }

        let end_idx = end_idx.min(all_frames.len());
        Ok(all_frames[start_idx..end_idx].to_vec())
    }
}

impl Default for VideoDecoder {
    fn default() -> Self {
        Self::new().unwrap_or(Self {})
    }
}

/// Video metadata information
#[derive(Debug, Clone)]
pub struct VideoInfo {
    /// Video width in pixels
    pub width: u32,
    /// Video height in pixels  
    pub height: u32,
    /// Frames per second
    pub fps: f64,
    /// Duration in seconds
    pub duration_seconds: f64,
    /// Total number of frames
    pub frame_count: u32,
    /// Pixel format
    pub format: String,
    /// Video codec
    pub codec: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_decoder_creation() {
        let decoder = VideoDecoder::new();
        assert!(decoder.is_ok());
    }

    #[tokio::test]
    async fn test_nonexistent_video() {
        let decoder = VideoDecoder::new().unwrap();
        let result = decoder.extract_frames("nonexistent.mp4").await;
        assert!(result.is_err());
    }
}
