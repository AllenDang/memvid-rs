# Memvid-rs Examples

This directory contains example applications demonstrating how to use memvid-rs.

## Simple Chat Example

The `simple_chat.rs` example demonstrates both quick queries and interactive chat with your video memories, similar to the Python version.

### Features

- **Quick Chat**: One-off queries with OpenAI LLM integration
- **Interactive Chat**: Full conversation mode with command support
- **Context-Only Mode**: Intelligent fallback when no API key provided
- **Search Commands**: Direct search functionality within chat
- **Memory Statistics**: View information about your encoded video memory
- **Smart Context**: Uses top-ranked chunks for better responses

### Usage

1. **First, create a video memory:**
   ```bash
   cargo run -- encode data/bitcoin.pdf -o data/memory.mp4 -i data/memory_index.db
   ```

2. **Run the chat demo:**
   ```bash
   cargo run --example simple_chat
   ```

3. **For full LLM responses, set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   cargo run --example simple_chat
   ```

## Ollama/Local LLM Example

The `ollama_chat.rs` example demonstrates how to use memvid-rs with Ollama or other OpenAI-compatible APIs for local/private LLM inference.

### Features

- **Local LLM Support**: Use Ollama, LocalAI, or any OpenAI-compatible API
- **No Internet Required**: Run completely offline with local models
- **Configurable Endpoints**: Custom base URLs and model names
- **Connection Testing**: Automatic connectivity verification
- **Same Chat Features**: All the interactive features from simple_chat

### Setup for Ollama

1. **Install Ollama:**
   ```bash
   # Visit https://ollama.ai for installation instructions
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Start Ollama server:**
   ```bash
   ollama serve
   ```

3. **Pull a model:**
   ```bash
   ollama pull llama2        # Or llama3, codellama, etc.
   ```

4. **Run the example:**
   ```bash
   cargo run --example ollama_chat
   ```

### Configuration

Set environment variables to customize the setup:

```bash
# For Ollama (defaults)
export OLLAMA_BASE_URL="http://localhost:11434/v1"
export OLLAMA_MODEL="llama2"

# For other OpenAI-compatible APIs
export OLLAMA_BASE_URL="http://your-local-api:8080/v1"
export OLLAMA_MODEL="your-model-name"

cargo run --example ollama_chat
```

### Supported APIs

The configurable API system supports any OpenAI-compatible endpoint:

- **Ollama**: Local LLM serving (`http://localhost:11434/v1`)
- **LocalAI**: Self-hosted OpenAI alternative
- **LM Studio**: Local model serving
- **vLLM**: High-performance inference server
- **Text Generation WebUI**: Gradio-based interface with OpenAI API
- **Any OpenAI-compatible API**: Just set the base URL

### Interactive Commands

When in interactive mode, you can use:
- `help` - Show available commands
- `search <query>` - Show raw search results with scores
- `stats` - Display memory statistics  
- `clear` - Clear conversation history
- `quit` or `exit` - End the session
- Any other text - Ask questions about your documents

### Example Output

```
Memvid Ollama/Local LLM Chat Example
==================================================

Configuration:
  Base URL: http://localhost:11434/v1
  Model: llama2
  API Key: Not required

1. Testing connectivity with quick query:
----------------------------------------
Query: What topics are covered in this knowledge base?
✅ Connection successful!
Response: Based on the knowledge base, I can see several key topics covered including quantum computing concepts, machine learning fundamentals, and blockchain technology...

2. Starting interactive chat session:
----------------------------------------
Type 'help' for commands, 'exit' to quit

Memory loaded: 1,234 chunks
LLM: llama2 (local)

Type 'help' for commands, 'exit' to quit
--------------------------------------------------

You: What is quantum supremacy?
Assistant: Based on the knowledge base, quantum supremacy refers to the point where quantum computers can perform calculations that are practically impossible for classical computers...

[2.1s]

You: search quantum computing  
Searching: 'quantum computing'
Found 3 results in 0.045s:

1. [Score: 0.892] Quantum computing harnesses quantum mechanical phenomena...
2. [Score: 0.756] Unlike classical bits that exist in states of 0 or 1...
3. [Score: 0.689] Quantum algorithms like Shor's algorithm could potentially...

You: stats
System Statistics:
  Total chunks: 1,234
  Total frames: 567
  Cached frames: 12
  Database size: 1,048,576 bytes

You: quit
Goodbye!
```

### Requirements

- Video memory files (created with `cargo run -- encode`)
- SQLite database index file (automatically created during encoding)  
- **For simple_chat**: Optional OpenAI API key for cloud LLM responses
- **For ollama_chat**: Local Ollama installation or other OpenAI-compatible API
- Internet connection (only for OpenAI API calls) 