//! Device detection and optimization for ML inference
//!
//! This module handles automatic detection of the best available compute device
//! (CUDA GPU, Metal GPU, or CPU) and provides device management for ML operations.

use crate::error::{MemvidError, Result};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::ptr;
use std::sync::Once;

/// Device types supported for ML inference
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DeviceType {
    /// CPU inference
    Cpu,
    /// CUDA GPU inference
    Cuda(usize),
    /// Metal GPU inference (macOS)
    Metal,
}

/// Device information and capabilities
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device type
    pub device_type: DeviceType,
    /// Candle device instance
    pub device: Device,
    /// Device name/description
    pub name: String,
    /// Estimated compute capability (relative score)
    pub compute_score: f32,
    /// Available memory in bytes (estimate)
    pub memory_bytes: Option<u64>,
}

/// Global device manager instance
static mut DEVICE_MANAGER: Option<DeviceManager> = None;
static DEVICE_MANAGER_INIT: Once = Once::new();

/// Device manager for automatic device selection and optimization
pub struct DeviceManager {
    /// Current optimal device
    current_device: DeviceInfo,
    /// All available devices
    available_devices: Vec<DeviceInfo>,
}

impl DeviceManager {
    /// Initialize device manager with automatic device detection
    pub fn initialize() -> Result<&'static DeviceManager> {
        unsafe {
            DEVICE_MANAGER_INIT.call_once(|| match Self::new() {
                Ok(manager) => {
                    log::info!(
                        "Initialized device manager with optimal device: {}",
                        manager.current_device.name
                    );
                    DEVICE_MANAGER = Some(manager);
                }
                Err(e) => {
                    log::error!("Failed to initialize device manager: {}", e);
                }
            });

            ptr::addr_of!(DEVICE_MANAGER)
                .as_ref()
                .unwrap()
                .as_ref()
                .ok_or_else(|| {
                    MemvidError::MachineLearning("Device manager initialization failed".to_string())
                })
        }
    }

    /// Get global device manager instance
    pub fn global() -> Result<&'static DeviceManager> {
        unsafe {
            ptr::addr_of!(DEVICE_MANAGER)
                .as_ref()
                .unwrap()
                .as_ref()
                .ok_or_else(|| {
                    MemvidError::MachineLearning("Device manager not initialized".to_string())
                })
        }
    }

    /// Create new device manager with automatic device detection
    fn new() -> Result<Self> {
        let mut available_devices = Vec::new();

        // Detect CPU
        let cpu_device = DeviceInfo {
            device_type: DeviceType::Cpu,
            device: Device::Cpu,
            name: "CPU".to_string(),
            compute_score: 1.0, // Base score
            memory_bytes: Self::estimate_system_memory(),
        };
        available_devices.push(cpu_device);

        // Detect CUDA devices
        #[cfg(feature = "cuda")]
        {
            for device_id in 0..8 {
                // Check up to 8 CUDA devices
                if let Ok(device) = Device::cuda_if_available(device_id) {
                    let device_info = DeviceInfo {
                        device_type: DeviceType::Cuda(device_id),
                        device,
                        name: format!("CUDA GPU {}", device_id),
                        compute_score: 10.0 + device_id as f32, // Higher score for GPUs
                        memory_bytes: Self::estimate_gpu_memory(device_id),
                    };
                    available_devices.push(device_info);
                    log::info!("Detected CUDA device {}", device_id);
                }
            }
        }

        // Detect Metal device (macOS)
        #[cfg(feature = "metal")]
        {
            if let Ok(device) = Device::new_metal(0) {
                let device_info = DeviceInfo {
                    device_type: DeviceType::Metal,
                    device,
                    name: "Metal GPU".to_string(),
                    compute_score: 15.0, // High score for Metal
                    memory_bytes: Self::estimate_metal_memory(),
                };
                available_devices.push(device_info);
                log::info!("Detected Metal GPU");
            }
        }

        // Select optimal device (highest compute score)
        let current_device = available_devices
            .iter()
            .max_by(|a, b| a.compute_score.partial_cmp(&b.compute_score).unwrap())
            .cloned()
            .ok_or_else(|| MemvidError::MachineLearning("No devices available".to_string()))?;

        log::info!("Selected optimal device: {}", current_device.name);

        Ok(Self {
            current_device,
            available_devices,
        })
    }

    /// Get current optimal device
    pub fn current_device(&self) -> &DeviceInfo {
        &self.current_device
    }

    /// Get all available devices
    pub fn available_devices(&self) -> &[DeviceInfo] {
        &self.available_devices
    }

    /// Get device by type
    pub fn get_device(&self, device_type: &DeviceType) -> Option<&DeviceInfo> {
        self.available_devices
            .iter()
            .find(|d| d.device_type == *device_type)
    }

    /// Switch to a specific device type
    pub fn switch_device(&mut self, device_type: DeviceType) -> Result<()> {
        if let Some(device_info) = self
            .available_devices
            .iter()
            .find(|d| d.device_type == device_type)
            .cloned()
        {
            self.current_device = device_info;
            log::info!("Switched to device: {}", self.current_device.name);
            Ok(())
        } else {
            Err(MemvidError::MachineLearning(format!(
                "Device type {:?} not available",
                device_type
            )))
        }
    }

    /// Get optimal batch size for current device
    pub fn optimal_batch_size(&self, base_batch_size: usize) -> usize {
        match self.current_device.device_type {
            DeviceType::Cpu => base_batch_size.min(32), // Conservative for CPU
            DeviceType::Cuda(_) => base_batch_size * 2, // Can handle larger batches
            DeviceType::Metal => base_batch_size.max(16), // Good performance on Metal
        }
    }

    /// Check if device supports half precision
    pub fn supports_half_precision(&self) -> bool {
        matches!(
            self.current_device.device_type,
            DeviceType::Cuda(_) | DeviceType::Metal
        )
    }

    /// Estimate system memory
    fn estimate_system_memory() -> Option<u64> {
        // Simple heuristic - could be improved with platform-specific APIs
        Some(8 * 1024 * 1024 * 1024) // Assume 8GB as conservative estimate
    }

    /// Estimate GPU memory for CUDA device
    #[cfg(feature = "cuda")]
    fn estimate_gpu_memory(_device_id: usize) -> Option<u64> {
        // Would use CUDA APIs in production
        Some(4 * 1024 * 1024 * 1024) // Assume 4GB as conservative estimate
    }

    /// Estimate Metal GPU memory
    #[cfg(feature = "metal")]
    fn estimate_metal_memory() -> Option<u64> {
        // Would use Metal APIs in production
        Some(8 * 1024 * 1024 * 1024) // Assume 8GB unified memory
    }

    #[cfg(not(feature = "metal"))]
    fn estimate_metal_memory() -> Option<u64> {
        None
    }
}

/// Initialize device system
pub fn initialize() -> Result<()> {
    DeviceManager::initialize()?;
    Ok(())
}

/// Get current optimal device
pub fn current_device() -> Result<&'static DeviceInfo> {
    Ok(DeviceManager::global()?.current_device())
}

/// Get all available devices
pub fn available_devices() -> Result<&'static [DeviceInfo]> {
    Ok(DeviceManager::global()?.available_devices())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_manager_initialization() {
        let manager = DeviceManager::initialize().unwrap();
        assert!(!manager.available_devices().is_empty());

        // Should always have at least CPU
        assert!(
            manager
                .available_devices()
                .iter()
                .any(|d| matches!(d.device_type, DeviceType::Cpu))
        );
    }

    #[test]
    fn test_device_selection() {
        let manager = DeviceManager::initialize().unwrap();
        let current = manager.current_device();

        // Should select a valid device
        assert!(!current.name.is_empty());
        assert!(current.compute_score > 0.0);
    }

    #[test]
    fn test_batch_size_optimization() {
        let manager = DeviceManager::initialize().unwrap();
        let base_size = 16;
        let optimal = manager.optimal_batch_size(base_size);

        assert!(optimal > 0);
        assert!(optimal <= base_size * 4); // Reasonable upper bound
    }
}
