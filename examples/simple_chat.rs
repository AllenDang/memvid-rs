#!/usr/bin/env rust
//! Simplified interactive chat example
//! 
//! This example demonstrates how to use memvid-rs for:
//! 1. Quick one-off queries
//! 2. Interactive chat sessions
//! 
//! Run with: cargo run --example simple_chat

use memvid_rs::{quick_chat, chat_with_memory, Result};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    
    println!("Memvid Simple Chat Examples");
    println!("{}", "=".repeat(50));
    
    let video_file = "data/memory.mp4";
    let index_file = "data/memory_index.db";
    
    // Check if memory exists
    if !Path::new(video_file).exists() {
        println!("\nError: Run 'cargo run -- encode data/bitcoin.pdf -o {} -i {}' first!", video_file, index_file);
        return Ok(());
    }
    
    // Check for API key
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
        println!("\nNote: Set OPENAI_API_KEY environment variable for full LLM responses.");
        println!("Without it, you'll see context-only responses.\n");
        String::new()
    });
    
    println!("\n1. Quick one-off query:");
    println!("{}", "-".repeat(30));
    let response = quick_chat(video_file, index_file, "How many qubits did the quantum computer achieve?", &api_key).await?;
    println!("Response: {}", response);
    
    println!("\n\n2. Interactive chat session:");
    println!("{}", "-".repeat(30));
    println!("Starting interactive mode...\n");
    
    // Interactive chat session
    chat_with_memory(video_file, index_file, &api_key).await?;
    
    Ok(())
} 