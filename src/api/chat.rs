//! High-level chat API
//!
//! This module provides convenient functions for quick queries and interactive chat sessions.

use crate::api::MemvidRetriever;
use crate::error::{Result, MemvidError};
use std::io::{self, Write};
use async_openai::{Client, config::OpenAIConfig};
use async_openai::types::{CreateChatCompletionRequestArgs, ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage, ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessageContent};

/// Quick one-off query function
/// 
/// This function performs a single query against the video memory and returns either
/// an LLM-generated response (if api_key is provided) or raw context chunks.
///
/// # Arguments
/// * `video_file` - Path to the video memory file
/// * `index_file` - Path to the database index file  
/// * `query` - The question to ask
/// * `api_key` - Optional OpenAI API key for LLM responses
///
/// # Examples
/// ```no_run
/// use memvid_rs::quick_chat;
/// 
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let response = quick_chat(
///         "memory.mp4", 
///         "memory_index.db", 
///         "What is quantum computing?",
///         ""  // No API key - will return raw chunks
///     ).await?;
///     println!("Response: {}", response);
///     Ok(())
/// }
/// ```
pub async fn quick_chat(
    video_file: &str,
    index_file: &str, 
    query: &str,
    api_key: &str
) -> Result<String> {
    quick_chat_with_config(video_file, index_file, query, api_key, None, None).await
}

/// Quick chat with configurable OpenAI-compatible API settings
/// 
/// # Arguments
/// * `video_file` - Path to the video memory file
/// * `index_file` - Path to the database index file  
/// * `query` - The question to ask
/// * `api_key` - Optional OpenAI API key for LLM responses
/// * `base_url` - Optional base URL for OpenAI-compatible APIs (e.g., "http://localhost:11434/v1" for Ollama)
/// * `model` - Optional model name (defaults to "gpt-3.5-turbo")
///
/// # Examples
/// ```no_run
/// use memvid_rs::quick_chat_with_config;
/// 
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Use with Ollama
///     let response = quick_chat_with_config(
///         "memory.mp4", 
///         "memory_index.db", 
///         "What is quantum computing?",
///         "",  // No API key needed for local Ollama
///         Some("http://localhost:11434/v1"),
///         Some("llama2")
///     ).await?;
///     println!("Response: {}", response);
///     Ok(())
/// }
/// ```
pub async fn quick_chat_with_config(
    video_file: &str,
    index_file: &str, 
    query: &str,
    api_key: &str,
    base_url: Option<&str>,
    model: Option<&str>
) -> Result<String> {
    let mut retriever = MemvidRetriever::new(video_file, index_file).await?;
    
    // Get relevant chunks (matching Python's default of 5 chunks)
    let results = retriever.search(query, 5).await?;
    
    if results.is_empty() {
        return Ok("I couldn't find any relevant information in the knowledge base.".to_string());
    }
    
    // Check if the chunks are actually relevant (following Python logic)
    let avg_chunk_length: f32 = results.iter()
        .map(|(_, text)| text.len() as f32)
        .sum::<f32>() / results.len() as f32;
    
    if avg_chunk_length < 50.0 {
        return Ok("I couldn't find any relevant information about that topic in the knowledge base.".to_string());
    }
    
    // Build context from search results (join top chunks)
    let context = results.iter()
        .take(3) // Top 3 chunks like Python
        .map(|(_score, text)| format!("[Context]: {}", text))
        .collect::<Vec<_>>()
        .join("\n\n");
    
    // If no API key and no custom base URL, return context-only response (Python fallback behavior)
    // For local APIs like Ollama, empty API key is acceptable if base_url is provided
    if api_key.is_empty() && base_url.is_none() {
        return Ok(generate_context_only_response(&results));
    }
    
    // Generate LLM response using OpenAI API (matching Python implementation)
    match generate_openai_response(query, &context, api_key, base_url, model).await {
        Ok(response) => Ok(response),
        Err(e) => {
            log::warn!("LLM API error: {}. Falling back to context-only response.", e);
            Ok(generate_context_only_response(&results))
        }
    }
}

/// Generate context-only response (fallback when no LLM available)
/// Matches Python's _generate_context_only_response logic
fn generate_context_only_response(results: &[(f32, String)]) -> String {
    let mut response = "Based on the knowledge base, here's what I found:\n\n".to_string();
    
    for (i, (_score, chunk)) in results.iter().take(3).enumerate() {
        let chunk_preview = if chunk.len() > 200 {
            format!("{}...", &chunk[..200])
        } else {
            chunk.clone()
        };
        response.push_str(&format!("{}. {}\n\n", i + 1, chunk_preview));
    }
    
    response.trim().to_string()
}

/// Generate LLM response using async_openai (supporting OpenAI-compatible APIs)
async fn generate_openai_response(
    query: &str, 
    context: &str, 
    api_key: &str,
    base_url: Option<&str>,
    model: Option<&str>
) -> std::result::Result<String, Box<dyn std::error::Error>> {
    // Create OpenAI client with custom configuration
    let config = if let Some(base_url) = base_url {
        OpenAIConfig::new()
            .with_api_key(api_key)
            .with_api_base(base_url)
    } else {
        OpenAIConfig::new().with_api_key(api_key)
    };
    
    let client = Client::with_config(config);
    let model_name = model.unwrap_or("gpt-3.5-turbo");
    
    // System prompt matching Python's default
    let system_prompt = "You are a helpful AI assistant with access to a knowledge base stored in video format. \n\nWhen answering questions:\n1. Use the provided context from the knowledge base when relevant\n2. Be clear about what information comes from the knowledge base vs. your general knowledge\n3. If the context doesn't contain enough information, say so clearly\n4. Provide helpful, accurate, and concise responses\n\nThe context will be provided with each query based on semantic similarity to the user's question.";
    
    // Build enhanced message with context (matching Python's _build_messages)
    let enhanced_message = if context.trim().is_empty() {
        query.to_string()
    } else {
        format!("Context from knowledge base:\n{}\n\nUser question: {}", context, query)
    };
    
    // Build messages using async_openai types
    let messages = vec![
        ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
            content: ChatCompletionRequestSystemMessageContent::Text(system_prompt.to_string()),
            name: None,
        }),
        ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Text(enhanced_message),
            name: None,
        }),
    ];
    
    // Create chat completion request
    let request = CreateChatCompletionRequestArgs::default()
        .model(model_name)
        .messages(messages)
        .max_tokens(500u16)
        .temperature(0.7)
        .build()?;
    
    // Make the API call
    let response = client.chat().create(request).await?;
    
    // Extract the response content
    let content = response
        .choices
        .first()
        .and_then(|choice| choice.message.content.as_ref())
        .ok_or_else(|| {
            log::error!("No content in chat response: {:?}", response);
            MemvidError::Generic("No content in response".to_string())
        })?;
    
    Ok(content.clone())
}

/// Interactive chat session
///
/// This function starts an interactive chat session where users can ask multiple questions
/// and receive responses. It includes special commands like 'help', 'stats', etc.
/// Matches Python's interactive.py chat_with_memory functionality.
///
/// # Arguments
/// * `video_file` - Path to the video memory file
/// * `index_file` - Path to the database index file
/// * `api_key` - Optional OpenAI API key for LLM responses
///
/// # Examples
/// ```no_run
/// use memvid_rs::chat_with_memory;
/// 
/// #[tokio::main] 
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     chat_with_memory("memory.mp4", "memory_index.db", "").await?;
///     Ok(())
/// }
/// ```
pub async fn chat_with_memory(
    video_file: &str,
    index_file: &str,
    api_key: &str
) -> Result<()> {
    chat_with_memory_config(video_file, index_file, api_key, None, None).await
}

/// Interactive chat session with configurable OpenAI-compatible API settings
///
/// # Arguments
/// * `video_file` - Path to the video memory file
/// * `index_file` - Path to the database index file
/// * `api_key` - Optional OpenAI API key for LLM responses
/// * `base_url` - Optional base URL for OpenAI-compatible APIs
/// * `model` - Optional model name
///
/// # Examples
/// ```no_run
/// use memvid_rs::chat_with_memory_config;
/// 
/// #[tokio::main] 
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Use with Ollama
///     chat_with_memory_config(
///         "memory.mp4", 
///         "memory_index.db", 
///         "",  // No API key needed for local Ollama
///         Some("http://localhost:11434/v1"),
///         Some("llama2")
///     ).await?;
///     Ok(())
/// }
/// ```
pub async fn chat_with_memory_config(
    video_file: &str,
    index_file: &str,
    api_key: &str,
    base_url: Option<&str>,
    model: Option<&str>
) -> Result<()> {
    let mut retriever = MemvidRetriever::new(video_file, index_file).await?;
    
    // Display startup message (matching Python)
    println!("💬 Interactive Chat Mode");
    println!("   Type 'quit' or 'exit' to end the session");
    println!("   Type 'help' for more commands");
    
    // Show initial stats (matching Python behavior)
    if let Ok(stats) = retriever.get_stats() {
        println!("\nMemory loaded: {} chunks", stats.total_chunks);
        if api_key.is_empty() && base_url.is_none() {
            println!("LLM: Not available (context-only mode)");
        } else if let Some(base_url) = base_url {
            let model_name = model.unwrap_or("default");
            println!("LLM: {} via {}", model_name, base_url);
        } else {
            println!("LLM: OpenAI GPT-3.5-turbo");
        }
    }
    
    println!("\nType 'help' for commands, 'exit' to quit");
    println!("{}", "-".repeat(50));
    
    loop {
        print!("\nYou: ");
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();
        
        if input.is_empty() {
            continue;
        }
        
        match input.to_lowercase().as_str() {
            "quit" | "exit" | "q" => {
                println!("Goodbye!");
                break;
            }
            "help" => {
                println!("\nCommands:");
                println!("  search <query> - Show raw search results");
                println!("  stats         - Show system statistics");
                println!("  clear         - Clear conversation history");
                println!("  help          - Show this help");
                println!("  exit/quit     - End session");
                continue;
            }
            "stats" => {
                match retriever.get_stats() {
                    Ok(stats) => {
                        println!("\nSystem Statistics:");
                        println!("  Total chunks: {}", stats.total_chunks);
                        println!("  Total frames: {}", stats.total_frames);
                        println!("  Cached frames: {}", stats.cached_frames);
                        println!("  Database size: {} bytes", stats.database_size_bytes);
                    }
                    Err(e) => println!("❌ Error getting stats: {}", e),
                }
                continue;
            }
            "clear" => {
                println!("Conversation history cleared.");
                // Note: In a full implementation, we'd maintain conversation history
                continue;
            }
            _ => {
                // Handle search command
                if input.to_lowercase().starts_with("search ") {
                    let query = &input[7..];
                    println!("\nSearching: '{}'", query);
                    
                    let start_time = std::time::Instant::now();
                    match retriever.search(query, 5).await {
                        Ok(results) => {
                            let elapsed = start_time.elapsed();
                            println!("Found {} results in {:.3}s:\n", results.len(), elapsed.as_secs_f64());
                            
                            for (i, (score, text)) in results.iter().take(3).enumerate() {
                                let preview = if text.len() > 100 {
                                    format!("{}...", &text[..100])
                                } else {
                                    text.clone()
                                };
                                println!("{}. [Score: {:.3}] {}", i + 1, score, preview);
                            }
                        }
                        Err(e) => println!("❌ Search error: {}", e),
                    }
                    continue;
                }
                
                // Regular chat - process the query
                let start_time = std::time::Instant::now();
                let response = quick_chat_with_config(video_file, index_file, input, api_key, base_url, model).await?;
                let elapsed = start_time.elapsed();
                
                println!("\nAssistant: {}", response);
                println!("[{:.1}s]", elapsed.as_secs_f64());
            }
        }
    }
    
    Ok(())
} 