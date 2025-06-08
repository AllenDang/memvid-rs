//! Database schema definitions

/// Database schema version
pub const SCHEMA_VERSION: u32 = 1;

/// SQL for creating the chunks table
pub const CREATE_CHUNKS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT,
    page INTEGER,
    offset INTEGER NOT NULL,
    length INTEGER NOT NULL,
    frame INTEGER,
    embedding BLOB
);
"#;

/// SQL for creating the metadata table
pub const CREATE_METADATA_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"#;

/// SQL for creating indexes on chunks table for O(1) and O(log n) lookups
pub const CREATE_CHUNKS_INDEXES: &str = r#"
CREATE INDEX IF NOT EXISTS idx_chunks_frame ON chunks(frame);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page);
"#; 