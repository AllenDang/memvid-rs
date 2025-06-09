//! Model management for memvid-rs ML system
//!
//! This module handles downloading, caching, and managing ML models
//! using pure Rust implementations.

use crate::error::{MemvidError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Types of models supported
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModelType {
    /// Sentence transformer for embeddings
    SentenceTransformer,
    /// BERT-based models
    Bert,
    /// Custom models
    Custom(String),
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model name/identifier
    pub name: String,
    /// Model type
    pub model_type: ModelType,
    /// Local path to model files
    pub local_path: Option<PathBuf>,
    /// HuggingFace model hub identifier
    pub hub_id: Option<String>,
    /// Model configuration
    pub config: ModelConfig,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model dimension
    pub dimension: usize,
    /// Maximum sequence length
    pub max_length: usize,
    /// Whether model is cached locally
    pub cached: bool,
    /// Additional parameters
    pub params: HashMap<String, String>,
}

/// Model manager for downloading and caching models
pub struct ModelManager {
    /// Cache directory for models
    cache_dir: PathBuf,
    /// Available models
    models: HashMap<String, ModelInfo>,
}

impl ModelManager {
    /// Create new model manager
    pub fn new(cache_dir: Option<PathBuf>) -> Result<Self> {
        let cache_dir = cache_dir.unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home)
                .join(".cache")
                .join("memvid-rs")
                .join("models")
        });

        // Create cache directory if it doesn't exist
        std::fs::create_dir_all(&cache_dir)?;

        let mut manager = Self {
            cache_dir,
            models: HashMap::new(),
        };

        // Register default models
        manager.register_default_models()?;

        Ok(manager)
    }

    /// Register default models
    fn register_default_models(&mut self) -> Result<()> {
        // Register all-MiniLM-L6-v2 model
        let mini_lm = ModelInfo {
            name: "all-MiniLM-L6-v2".to_string(),
            model_type: ModelType::SentenceTransformer,
            local_path: None,
            hub_id: Some("sentence-transformers/all-MiniLM-L6-v2".to_string()),
            config: ModelConfig {
                dimension: 384,
                max_length: 384,
                cached: false,
                params: HashMap::new(),
            },
        };

        // Register with both short and full names for compatibility
        self.models
            .insert("all-MiniLM-L6-v2".to_string(), mini_lm.clone());
        self.models.insert(
            "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            mini_lm,
        );

        // Register other common models
        let bert_base = ModelInfo {
            name: "bert-base-uncased".to_string(),
            model_type: ModelType::Bert,
            local_path: None,
            hub_id: Some("bert-base-uncased".to_string()),
            config: ModelConfig {
                dimension: 768,
                max_length: 512,
                cached: false,
                params: HashMap::new(),
            },
        };

        self.models
            .insert("bert-base-uncased".to_string(), bert_base);

        Ok(())
    }

    /// Get model info by name
    pub fn get_model(&self, name: &str) -> Option<&ModelInfo> {
        self.models.get(name)
    }

    /// List available models
    pub fn list_models(&self) -> Vec<&ModelInfo> {
        self.models.values().collect()
    }

    /// Check if model is cached locally
    pub fn is_cached(&self, name: &str) -> bool {
        if let Some(model) = self.models.get(name) {
            model.config.cached && model.local_path.is_some()
        } else {
            false
        }
    }

    /// Get cache directory
    pub fn cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }

    /// Download and cache model from HuggingFace Hub
    pub async fn download_model(&mut self, name: &str) -> Result<PathBuf> {
        let model = self
            .models
            .get_mut(name)
            .ok_or_else(|| MemvidError::MachineLearning(format!("Model '{}' not found", name)))?;

        if let Some(local_path) = &model.local_path {
            if local_path.exists() && Self::validate_model_files_static(local_path)? {
                log::info!("Model '{}' already cached at {:?}", name, local_path);
                return Ok(local_path.clone());
            }
        }

        let model_dir = self.cache_dir.join(name);
        std::fs::create_dir_all(&model_dir)?;

        if let Some(hub_id) = &model.hub_id {
            log::info!(
                "Downloading model '{}' from HuggingFace Hub: {}",
                name,
                hub_id
            );

            // Download essential model files
            let files_to_download = vec![
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors",
                "vocab.txt", // For BERT-based models
            ];

            let mut downloaded_any = false;
            for file_name in files_to_download {
                match Self::download_file_static(hub_id, file_name, &model_dir) {
                    Ok(_) => {
                        downloaded_any = true;
                        log::debug!("Downloaded {}/{}", hub_id, file_name);
                    }
                    Err(e) => {
                        // Some files are optional, only warn
                        log::warn!("Failed to download {}/{}: {}", hub_id, file_name, e);
                    }
                }
            }

            if downloaded_any {
                log::info!("Successfully downloaded model files for '{}'", name);
                model.local_path = Some(model_dir.clone());
                model.config.cached = true;
            } else {
                log::error!("Failed to download any files for model '{}'", name);
                return Err(MemvidError::MachineLearning(format!(
                    "Failed to download model '{}'",
                    name
                )));
            }
        } else {
            // Create placeholder for models without hub_id
            log::warn!(
                "No HuggingFace Hub ID for model '{}', creating placeholder",
                name
            );
            model.local_path = Some(model_dir.clone());
            model.config.cached = true;
        }

        Ok(model_dir)
    }

    /// Download a single file from HuggingFace Hub (static method to avoid borrowing issues)
    fn download_file_static(repo_id: &str, filename: &str, target_dir: &Path) -> Result<()> {
        use hf_hub::api::sync::Api;

        let api = Api::new()
            .map_err(|e| MemvidError::MachineLearning(format!("Failed to create HF API: {}", e)))?;

        let repo = api.model(repo_id.to_string());
        let target_path = target_dir.join(filename);

        // Skip if file already exists and is valid
        if target_path.exists() && target_path.metadata()?.len() > 0 {
            return Ok(());
        }

        match repo.get(filename) {
            Ok(downloaded_path) => {
                // Copy from downloaded path to target path
                std::fs::copy(&downloaded_path, &target_path).map_err(|e| {
                    MemvidError::MachineLearning(format!("Failed to copy file: {}", e))
                })?;
                log::debug!("Downloaded and copied {} to {:?}", filename, target_path);
                Ok(())
            }
            Err(e) => Err(MemvidError::MachineLearning(format!(
                "Failed to download {}: {}",
                filename, e
            ))),
        }
    }

    /// Validate that essential model files exist (static method)
    fn validate_model_files_static(model_dir: &Path) -> Result<bool> {
        let essential_files = vec!["config.json"];

        for file_name in essential_files {
            let file_path = model_dir.join(file_name);
            if !file_path.exists() || file_path.metadata()?.len() == 0 {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Add custom model
    pub fn add_model(&mut self, model_info: ModelInfo) {
        self.models.insert(model_info.name.clone(), model_info);
    }

    /// Remove model from cache
    pub fn remove_model(&mut self, name: &str) -> Result<()> {
        if let Some(model) = self.models.get_mut(name) {
            if let Some(local_path) = &model.local_path {
                if local_path.exists() {
                    std::fs::remove_dir_all(local_path)?;
                }
            }
            model.local_path = None;
            model.config.cached = false;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_model_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new(Some(temp_dir.path().to_path_buf())).unwrap();

        assert!(manager.cache_dir().exists());
        assert!(manager.get_model("all-MiniLM-L6-v2").is_some());
        assert!(manager.get_model("bert-base-uncased").is_some());
    }

    #[tokio::test]
    async fn test_model_listing() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ModelManager::new(Some(temp_dir.path().to_path_buf())).unwrap();

        let models = manager.list_models();
        assert!(models.len() >= 2); // at least our default models

        let model_names: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
        assert!(model_names.contains(&"all-MiniLM-L6-v2"));
        assert!(model_names.contains(&"bert-base-uncased"));
    }

    #[tokio::test]
    async fn test_model_caching() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = ModelManager::new(Some(temp_dir.path().to_path_buf())).unwrap();

        assert!(!manager.is_cached("all-MiniLM-L6-v2"));

        let model_path = manager.download_model("all-MiniLM-L6-v2").await.unwrap();
        assert!(model_path.exists());
        assert!(manager.is_cached("all-MiniLM-L6-v2"));
    }
}
