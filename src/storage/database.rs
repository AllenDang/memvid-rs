//! SQLite database operations for memvid-rs
//!
//! This module provides high-performance metadata storage using embedded SQLite
//! with O(1) chunk lookups and millions of chunks scalability.

use crate::error::{MemvidError, Result};
use crate::text::ChunkMetadata;
use crate::storage::schema::*;
use rusqlite::{Connection, params, Row, OptionalExtension};
use std::path::Path;

/// Database connection and operations
pub struct Database {
    conn: Connection,
}

impl Database {
    /// Create a new database connection
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path)
            .map_err(|e| MemvidError::Storage(format!("Failed to open database: {}", e)))?;

        let mut db = Self { conn };
        db.initialize()?;
        Ok(db)
    }

    /// Create an in-memory database (for testing)
    pub fn memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| MemvidError::Storage(format!("Failed to create in-memory database: {}", e)))?;

        let mut db = Self { conn };
        db.initialize()?;
        Ok(db)
    }

    /// Initialize database schema
    fn initialize(&mut self) -> Result<()> {
        // Enable WAL mode for better concurrency
        let _: String = self.conn.query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))
            .map_err(|e| MemvidError::Storage(format!("Failed to enable WAL mode: {}", e)))?;

        // Create tables
        self.conn.execute(CREATE_CHUNKS_TABLE, [])
            .map_err(|e| MemvidError::Storage(format!("Failed to create chunks table: {}", e)))?;

        self.conn.execute(CREATE_METADATA_TABLE, [])
            .map_err(|e| MemvidError::Storage(format!("Failed to create metadata table: {}", e)))?;

        // Create indexes for O(1) lookups
        self.conn.execute(CREATE_CHUNKS_INDEXES, [])
            .map_err(|e| MemvidError::Storage(format!("Failed to create indexes: {}", e)))?;

        // Set schema version
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', ?)",
            params![SCHEMA_VERSION.to_string()],
        ).map_err(|e| MemvidError::Storage(format!("Failed to set schema version: {}", e)))?;

        log::info!("Database initialized with schema version {}", SCHEMA_VERSION);
        Ok(())
    }

    /// Insert multiple chunks in a transaction
    pub fn insert_chunks(&mut self, chunks: &[ChunkMetadata]) -> Result<()> {
        let tx = self.conn.transaction()
            .map_err(|e| MemvidError::Storage(format!("Failed to start transaction: {}", e)))?;

        {
            let mut stmt = tx.prepare(
                r#"
                INSERT INTO chunks (id, text, source, page, offset, length, frame, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                "#,
            ).map_err(|e| MemvidError::Storage(format!("Failed to prepare statement: {}", e)))?;

            for chunk in chunks {
                let embedding_blob = chunk.embedding.as_ref().map(|emb| {
                    let mut bytes = Vec::new();
                    for &val in emb {
                        bytes.extend_from_slice(&val.to_le_bytes());
                    }
                    bytes
                });

                stmt.execute(params![
                    chunk.id as i64,
                    chunk.text,
                    chunk.source,
                    chunk.page.map(|p| p as i64),
                    chunk.offset as i64,
                    chunk.length as i64,
                    chunk.frame.map(|f| f as i64),
                    embedding_blob,
                ]).map_err(|e| MemvidError::Storage(format!("Failed to insert chunk {}: {}", chunk.id, e)))?;
            }
        }

        tx.commit()
            .map_err(|e| MemvidError::Storage(format!("Failed to commit transaction: {}", e)))?;

        log::info!("Inserted {} chunks into database", chunks.len());
        Ok(())
    }

    /// Get chunk by ID - O(1) lookup
    pub fn get_chunk_by_id(&self, chunk_id: usize) -> Result<Option<ChunkMetadata>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, text, source, page, offset, length, frame, embedding FROM chunks WHERE id = ?"
        ).map_err(|e| MemvidError::Storage(format!("Failed to prepare query: {}", e)))?;

        let chunk = stmt.query_row(params![chunk_id as i64], |row| {
            self.row_to_chunk(row)
        }).optional()
        .map_err(|e| MemvidError::Storage(format!("Failed to query chunk: {}", e)))?;

        Ok(chunk)
    }

    /// Get chunks by frame number - O(log n) indexed lookup
    pub fn get_chunks_by_frame(&self, frame_number: u32) -> Result<Vec<ChunkMetadata>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, text, source, page, offset, length, frame, embedding FROM chunks WHERE frame = ? ORDER BY id"
        ).map_err(|e| MemvidError::Storage(format!("Failed to prepare query: {}", e)))?;

        let chunks = stmt.query_map(params![frame_number as i64], |row| {
            self.row_to_chunk(row)
        }).map_err(|e| MemvidError::Storage(format!("Failed to query chunks by frame: {}", e)))?;

        let mut result = Vec::new();
        for chunk in chunks {
            result.push(chunk.map_err(|e| MemvidError::Storage(format!("Failed to process chunk row: {}", e)))?);
        }

        Ok(result)
    }

    /// Get total chunk count
    pub fn get_chunk_count(&self) -> Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM chunks",
            [],
            |row| row.get(0)
        ).map_err(|e| MemvidError::Storage(format!("Failed to count chunks: {}", e)))?;

        Ok(count as usize)
    }

    /// Search chunks by text content (simple LIKE search)
    pub fn search_chunks(&self, query: &str, limit: usize) -> Result<Vec<ChunkMetadata>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, text, source, page, offset, length, frame, embedding FROM chunks WHERE text LIKE ? ORDER BY id LIMIT ?"
        ).map_err(|e| MemvidError::Storage(format!("Failed to prepare search query: {}", e)))?;

        let search_term = format!("%{}%", query);
        let chunks = stmt.query_map(params![search_term, limit as i64], |row| {
            self.row_to_chunk(row)
        }).map_err(|e| MemvidError::Storage(format!("Failed to search chunks: {}", e)))?;

        let mut result = Vec::new();
        for chunk in chunks {
            result.push(chunk.map_err(|e| MemvidError::Storage(format!("Failed to process search result: {}", e)))?);
        }

        Ok(result)
    }

    /// Get database statistics
    pub fn get_stats(&self) -> Result<DatabaseStats> {
        let chunk_count = self.get_chunk_count()?;
        
        let file_size: i64 = self.conn.query_row(
            "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()",
            [],
            |row| row.get(0)
        ).map_err(|e| MemvidError::Storage(format!("Failed to get database size: {}", e)))?;

        let max_frame: Option<i64> = self.conn.query_row(
            "SELECT MAX(frame) FROM chunks WHERE frame IS NOT NULL",
            [],
            |row| row.get(0)
        ).optional()
        .map_err(|e| MemvidError::Storage(format!("Failed to get max frame: {}", e)))?
        .flatten();

        Ok(DatabaseStats {
            chunk_count,
            frame_count: max_frame.map(|f| f as usize + 1).unwrap_or(0),
            file_size_bytes: file_size as usize,
        })
    }

    /// Helper function to convert database row to ChunkMetadata
    fn row_to_chunk(&self, row: &Row) -> rusqlite::Result<ChunkMetadata> {
        let embedding = if let Some(blob) = row.get::<_, Option<Vec<u8>>>(7)? {
            let mut embedding = Vec::new();
            for chunk in blob.chunks_exact(4) {
                let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                embedding.push(f32::from_le_bytes(bytes));
            }
            Some(embedding)
        } else {
            None
        };

        Ok(ChunkMetadata {
            id: row.get::<_, i64>(0)? as usize,
            text: row.get(1)?,
            source: row.get(2)?,
            page: row.get::<_, Option<i64>>(3)?.map(|p| p as u32),
            offset: row.get::<_, i64>(4)? as usize,
            length: row.get::<_, i64>(5)? as usize,
            frame: row.get::<_, Option<i64>>(6)?.map(|f| f as u32),
            embedding,
        })
    }
}

/// Database statistics
#[derive(Debug, Clone)]
pub struct DatabaseStats {
    pub chunk_count: usize,
    pub frame_count: usize,
    pub file_size_bytes: usize,
} 