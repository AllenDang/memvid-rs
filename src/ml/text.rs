//! Text preprocessing and tokenization for ML models
//!
//! This module provides text preprocessing capabilities including tokenization,
//! normalization, and preparation for ML model inference.

use crate::error::{MemvidError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;
use unicode_normalization::UnicodeNormalization;

/// Text preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextConfig {
    /// Maximum sequence length
    pub max_length: usize,
    /// Whether to truncate long sequences
    pub truncate: bool,
    /// Whether to add special tokens (CLS, SEP)
    pub add_special_tokens: bool,
    /// Whether to normalize unicode
    pub normalize_unicode: bool,
    /// Whether to lowercase text
    pub lowercase: bool,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            max_length: 384,
            truncate: true,
            add_special_tokens: true,
            normalize_unicode: true,
            lowercase: false, // SentenceTransformers typically preserve case
        }
    }
}

/// Tokenized text ready for model inference
#[derive(Debug, Clone)]
pub struct TokenizedText {
    /// Token IDs
    pub input_ids: Vec<u32>,
    /// Attention mask (1 for real tokens, 0 for padding)
    pub attention_mask: Vec<u32>,
    /// Token type IDs (for BERT-style models)
    pub token_type_ids: Vec<u32>,
    /// Original text length before processing
    pub original_length: usize,
}

/// Text preprocessor and tokenizer
pub struct TextProcessor {
    /// Tokenizer instance
    tokenizer: Option<Tokenizer>,
    /// Configuration
    config: TextConfig,
}

impl TextProcessor {
    /// Create new text processor
    pub fn new(config: TextConfig) -> Self {
        Self {
            tokenizer: None,
            config,
        }
    }

    /// Load tokenizer from model directory
    pub fn load_tokenizer<P: AsRef<Path>>(&mut self, model_dir: P) -> Result<()> {
        let tokenizer_path = model_dir.as_ref().join("tokenizer.json");

        if tokenizer_path.exists() {
            match Tokenizer::from_file(&tokenizer_path) {
                Ok(tokenizer) => {
                    self.tokenizer = Some(tokenizer);
                    log::info!("Loaded tokenizer from {:?}", tokenizer_path);
                    Ok(())
                }
                Err(e) => {
                    log::warn!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e);
                    Err(MemvidError::MachineLearning(format!(
                        "Failed to load tokenizer: {}",
                        e
                    )))
                }
            }
        } else {
            log::warn!("Tokenizer file not found at {:?}", tokenizer_path);
            Err(MemvidError::MachineLearning(
                "Tokenizer file not found".to_string(),
            ))
        }
    }

    /// Preprocess text (normalize, clean, etc.)
    pub fn preprocess_text(&self, text: &str) -> String {
        let mut processed = text.to_string();

        // Unicode normalization
        if self.config.normalize_unicode {
            processed = processed.nfc().collect::<String>();
        }

        // Lowercase if configured
        if self.config.lowercase {
            processed = processed.to_lowercase();
        }

        // Basic cleaning
        processed = processed.trim().to_string();

        // Remove excessive whitespace
        processed = processed
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ");

        processed
    }

    /// Tokenize text for model inference
    pub fn tokenize(&self, text: &str) -> Result<TokenizedText> {
        let preprocessed = self.preprocess_text(text);
        let original_length = text.len();

        if let Some(ref tokenizer) = self.tokenizer {
            // Use real tokenizer
            let encoding = tokenizer
                .encode(preprocessed.clone(), self.config.add_special_tokens)
                .map_err(|e| MemvidError::MachineLearning(format!("Tokenization failed: {}", e)))?;

            let input_ids = encoding.get_ids().to_vec();
            let attention_mask = encoding.get_attention_mask().to_vec();
            let token_type_ids = encoding.get_type_ids().to_vec();

            // Truncate or pad to max_length
            let (input_ids, attention_mask, token_type_ids) =
                self.pad_or_truncate(input_ids, attention_mask, token_type_ids);

            Ok(TokenizedText {
                input_ids,
                attention_mask,
                token_type_ids,
                original_length,
            })
        } else {
            // Fallback to simple word-based tokenization
            log::warn!("No tokenizer loaded, using fallback tokenization");
            self.fallback_tokenize(&preprocessed, original_length)
        }
    }

    /// Tokenize multiple texts in batch
    pub fn tokenize_batch(&self, texts: &[String]) -> Result<Vec<TokenizedText>> {
        let mut results = Vec::new();

        if let Some(ref tokenizer) = self.tokenizer {
            // Batch tokenization for efficiency
            let preprocessed: Vec<String> = texts
                .iter()
                .map(|text| self.preprocess_text(text))
                .collect();

            let encodings = tokenizer
                .encode_batch(preprocessed.clone(), self.config.add_special_tokens)
                .map_err(|e| {
                    MemvidError::MachineLearning(format!("Batch tokenization failed: {}", e))
                })?;

            for (encoding, original_text) in encodings.iter().zip(texts.iter()) {
                let input_ids = encoding.get_ids().to_vec();
                let attention_mask = encoding.get_attention_mask().to_vec();
                let token_type_ids = encoding.get_type_ids().to_vec();

                let (input_ids, attention_mask, token_type_ids) =
                    self.pad_or_truncate(input_ids, attention_mask, token_type_ids);

                results.push(TokenizedText {
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    original_length: original_text.len(),
                });
            }
        } else {
            // Fallback to individual tokenization
            for text in texts {
                results.push(self.tokenize(text)?);
            }
        }

        Ok(results)
    }

    /// Pad or truncate sequences to max_length
    fn pad_or_truncate(
        &self,
        mut input_ids: Vec<u32>,
        mut attention_mask: Vec<u32>,
        mut token_type_ids: Vec<u32>,
    ) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let max_len = self.config.max_length;

        if input_ids.len() > max_len && self.config.truncate {
            // Truncate
            input_ids.truncate(max_len);
            attention_mask.truncate(max_len);
            token_type_ids.truncate(max_len);
        } else if input_ids.len() < max_len {
            // Pad with zeros (or appropriate padding tokens)
            let pad_len = max_len - input_ids.len();
            input_ids.extend(vec![0; pad_len]); // 0 is typically PAD token
            attention_mask.extend(vec![0; pad_len]); // 0 for padding
            token_type_ids.extend(vec![0; pad_len]); // 0 for padding
        }

        (input_ids, attention_mask, token_type_ids)
    }

    /// Fallback tokenization when no real tokenizer is available
    fn fallback_tokenize(&self, text: &str, original_length: usize) -> Result<TokenizedText> {
        // Simple word-based tokenization for fallback
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut input_ids = Vec::new();

        // Add CLS token if configured
        if self.config.add_special_tokens {
            input_ids.push(101); // [CLS] token ID
        }

        // Convert words to simple hash-based IDs (for testing)
        for word in words.iter().take(self.config.max_length - 2) {
            // Leave space for special tokens
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            use std::hash::{Hash, Hasher};
            word.hash(&mut hasher);
            let token_id = (hasher.finish() % 30000 + 1000) as u32; // Keep in reasonable range
            input_ids.push(token_id);
        }

        // Add SEP token if configured
        if self.config.add_special_tokens {
            input_ids.push(102); // [SEP] token ID
        }

        // Create attention mask and token type IDs
        let seq_len = input_ids.len();
        let attention_mask = vec![1u32; seq_len];
        let token_type_ids = vec![0u32; seq_len];

        // Pad to max_length
        let (input_ids, attention_mask, token_type_ids) =
            self.pad_or_truncate(input_ids, attention_mask, token_type_ids);

        log::debug!(
            "Fallback tokenization: {} words -> {} tokens",
            words.len(),
            seq_len
        );

        Ok(TokenizedText {
            input_ids,
            attention_mask,
            token_type_ids,
            original_length,
        })
    }

    /// Get tokenizer vocabulary size
    pub fn vocab_size(&self) -> Option<usize> {
        self.tokenizer.as_ref().map(|t| t.get_vocab_size(false))
    }

    /// Get configuration
    pub fn config(&self) -> &TextConfig {
        &self.config
    }

    /// Check if real tokenizer is loaded
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_config_default() {
        let config = TextConfig::default();
        assert_eq!(config.max_length, 384);
        assert!(config.truncate);
        assert!(config.add_special_tokens);
    }

    #[test]
    fn test_text_preprocessing() {
        let config = TextConfig {
            normalize_unicode: true,
            lowercase: true,
            ..Default::default()
        };
        let processor = TextProcessor::new(config);

        let text = "  Hello    WORLD!  ";
        let processed = processor.preprocess_text(text);
        assert_eq!(processed, "hello world!");
    }

    #[test]
    fn test_fallback_tokenization() {
        let config = TextConfig::default();
        let max_length = config.max_length;
        let processor = TextProcessor::new(config);

        let text = "Hello world test";
        let tokenized = processor.tokenize(text).unwrap();

        assert!(!tokenized.input_ids.is_empty());
        assert_eq!(tokenized.input_ids.len(), max_length);
        assert_eq!(tokenized.attention_mask.len(), max_length);
        assert_eq!(tokenized.original_length, text.len());
    }

    #[test]
    fn test_batch_tokenization_fallback() {
        let config = TextConfig::default();
        let max_length = config.max_length;
        let processor = TextProcessor::new(config);

        let texts = vec![
            "First sentence".to_string(),
            "Second sentence".to_string(),
            "Third sentence".to_string(),
        ];

        let tokenized = processor.tokenize_batch(&texts).unwrap();
        assert_eq!(tokenized.len(), 3);

        for tokens in &tokenized {
            assert_eq!(tokens.input_ids.len(), max_length);
            assert_eq!(tokens.attention_mask.len(), max_length);
        }
    }

    #[test]
    fn test_padding_truncation() {
        let config = TextConfig {
            max_length: 10,
            truncate: true,
            ..Default::default()
        };
        let processor = TextProcessor::new(config);

        // Test truncation
        let long_text = "This is a very long sentence that should be truncated";
        let tokenized = processor.tokenize(long_text).unwrap();
        assert_eq!(tokenized.input_ids.len(), 10);

        // Test padding
        let short_text = "Short";
        let tokenized = processor.tokenize(short_text).unwrap();
        assert_eq!(tokenized.input_ids.len(), 10);

        // Check that padding tokens are 0
        let padding_start = tokenized.attention_mask.iter().position(|&x| x == 0);
        assert!(padding_start.is_some());
    }
}
