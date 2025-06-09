#!/usr/bin/env rust
//! Ollama/Local LLM Chat Example
//!
//! This example demonstrates how to use memvid-rs with Ollama or other OpenAI-compatible APIs.
//!
//! Setup for Ollama:
//! 1. Install Ollama: https://ollama.ai
//! 2. Run: ollama serve
//! 3. Pull a model: ollama pull llama2
//! 4. Run this example: cargo run --example ollama_chat
//!
//! For other OpenAI-compatible APIs, adjust the base_url accordingly.

use memvid_rs::{Result, chat_with_memory_config, quick_chat_with_config};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("Memvid Ollama/Local LLM Chat Example");
    println!("{}", "=".repeat(50));

    let video_file = "data/memory.mp4";
    let index_file = "data/memory_index.db";

    // Check if memory exists
    if !Path::new(video_file).exists() {
        println!(
            "\nError: Run 'cargo run -- encode data/bitcoin.pdf -o {} -i {}' first!",
            video_file, index_file
        );
        return Ok(());
    }

    // Configuration for Ollama (default local setup)
    let base_url = std::env::var("OLLAMA_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama2".to_string());
    let api_key = ""; // Ollama doesn't require API key

    println!("\nConfiguration:");
    println!("  Base URL: {}", base_url);
    println!("  Model: {}", model);
    println!(
        "  API Key: {}",
        if api_key.is_empty() {
            "Not required"
        } else {
            "Set"
        }
    );

    // Test connectivity with a quick query
    println!("\n1. Testing connectivity with quick query:");
    println!("{}", "-".repeat(40));
    let test_query = "What topics are covered in this knowledge base?";
    println!("Query: {}", test_query);

    match quick_chat_with_config(
        video_file,
        index_file,
        test_query,
        api_key,
        Some(&base_url),
        Some(&model),
    )
    .await
    {
        Ok(response) => {
            println!("✅ Connection successful!");
            println!("Response: {}", response);
        }
        Err(e) => {
            println!("❌ Connection failed: {}", e);
            println!("\nTroubleshooting:");
            println!("1. Make sure Ollama is running: ollama serve");
            println!("2. Make sure the model is available: ollama pull {}", model);
            println!("3. Check the base URL: {}", base_url);
            println!(
                "4. For other APIs, set OLLAMA_BASE_URL and OLLAMA_MODEL environment variables"
            );
            return Ok(());
        }
    }

    println!("\n\n2. Starting interactive chat session:");
    println!("{}", "-".repeat(40));
    println!("Type 'help' for commands, 'exit' to quit\n");

    // Start interactive session with Ollama
    chat_with_memory_config(
        video_file,
        index_file,
        api_key,
        Some(&base_url),
        Some(&model),
    )
    .await?;

    Ok(())
}
