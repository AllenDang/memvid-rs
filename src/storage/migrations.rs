//! Database migrations and compatibility

use crate::error::{MemvidError, Result};
use rusqlite::{Connection, OptionalExtension};
use std::path::Path;

/// Database migration manager
pub struct MigrationManager {
    db_path: String,
}

impl Default for MigrationManager {
    fn default() -> Self {
        Self::new("memvid.db")
    }
}

impl MigrationManager {
    pub fn new(db_path: &str) -> Self {
        Self {
            db_path: db_path.to_string(),
        }
    }

    /// Run all pending migrations
    pub fn run_migrations(&self) -> Result<()> {
        // Create database file if it doesn't exist
        if !Path::new(&self.db_path).exists() {
            self.create_initial_schema()?;
        } else {
            self.apply_pending_migrations()?;
        }
        Ok(())
    }

    /// Create initial database schema
    fn create_initial_schema(&self) -> Result<()> {
        log::info!("Creating initial database schema at: {}", self.db_path);
        
        // Create the basic tables for memvid
        let connection = rusqlite::Connection::open(&self.db_path)
            .map_err(|e| MemvidError::Storage(format!("Failed to create database: {}", e)))?;

        // Migration tracking table
        connection.execute(
            "CREATE TABLE IF NOT EXISTS migrations (
                id INTEGER PRIMARY KEY,
                version TEXT NOT NULL UNIQUE,
                applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        ).map_err(|e| MemvidError::Storage(format!("Failed to create migrations table: {}", e)))?;

        // Chunks table
        connection.execute(
            "CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                length INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                importance_score REAL DEFAULT 0.5,
                tags TEXT DEFAULT '[]'
            )",
            [],
        ).map_err(|e| MemvidError::Storage(format!("Failed to create chunks table: {}", e)))?;

        // Embeddings table
        connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks (id)
            )",
            [],
        ).map_err(|e| MemvidError::Storage(format!("Failed to create embeddings table: {}", e)))?;

        // Index configuration table
        connection.execute(
            "CREATE TABLE IF NOT EXISTS index_config (
                id INTEGER PRIMARY KEY,
                key TEXT NOT NULL UNIQUE,
                value TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        ).map_err(|e| MemvidError::Storage(format!("Failed to create index_config table: {}", e)))?;

        // Mark initial migration as applied
        connection.execute(
            "INSERT INTO migrations (version) VALUES (?)",
            ["initial_schema"],
        ).map_err(|e| MemvidError::Storage(format!("Failed to record initial migration: {}", e)))?;

        log::info!("Initial database schema created successfully");
        Ok(())
    }

    /// Apply any pending migrations
    fn apply_pending_migrations(&self) -> Result<()> {
        let connection = rusqlite::Connection::open(&self.db_path)
            .map_err(|e| MemvidError::Storage(format!("Failed to open database: {}", e)))?;

        // Get current version
        let mut stmt = connection
            .prepare("SELECT version FROM migrations ORDER BY applied_at DESC")
            .map_err(|e| MemvidError::Storage(format!("Failed to prepare migration query: {}", e)))?;

        let migration_rows = stmt
            .query_map([], |row| {
                row.get::<_, String>(0)
            })
            .map_err(|e| MemvidError::Storage(format!("Failed to execute migration query: {}", e)))?;

        let mut current_versions = Vec::new();
        for version_result in migration_rows {
            let version = version_result
                .map_err(|e| MemvidError::Storage(format!("Failed to read migration version: {}", e)))?;
            current_versions.push(version);
        }

        // Define available migrations in order
        let available_migrations = vec![
            ("initial_schema", "Initial database schema"),
            ("add_metadata_columns", "Add metadata support to chunks"),
            ("add_search_indices", "Add search performance indices"),
        ];

        // Apply missing migrations
        for (version, description) in available_migrations {
            if !current_versions.contains(&version.to_string()) {
                log::info!("Applying migration: {} - {}", version, description);
                self.apply_migration(&connection, version)?;
                
                // Record migration
                connection.execute(
                    "INSERT INTO migrations (version) VALUES (?)",
                    [version],
                ).map_err(|e| MemvidError::Storage(format!("Failed to record migration {}: {}", version, e)))?;
            }
        }

        Ok(())
    }

    /// Apply a specific migration
    fn apply_migration(&self, connection: &Connection, version: &str) -> Result<()> {
        match version {
            "initial_schema" => {
                // Already handled in create_initial_schema
                Ok(())
            }
            "add_metadata_columns" => {
                // Add metadata columns to chunks table
                let _ = connection.execute(
                    "ALTER TABLE chunks ADD COLUMN metadata TEXT DEFAULT '{}'",
                    [],
                ).or_else(|_: rusqlite::Error| -> std::result::Result<usize, rusqlite::Error> {
                    // Column might already exist, ignore error
                    Ok(0)
                })?;
                Ok(())
            }
            "add_search_indices" => {
                // Add indices for better search performance
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_frame ON chunks(frame_number)",
                    [],
                )?;
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON chunks(created_at)",
                    [],
                )?;
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_importance ON chunks(importance_score)",
                    [],
                )?;
                Ok(())
            }
            _ => Err(MemvidError::Storage(format!("Unknown migration version: {}", version)))
        }
    }

    /// Get current database version
    pub fn get_current_version(&self) -> Result<Option<String>> {
        if !Path::new(&self.db_path).exists() {
            return Ok(None);
        }

        let connection = rusqlite::Connection::open(&self.db_path)
            .map_err(|e| MemvidError::Storage(format!("Failed to open database: {}", e)))?;

        let version = connection
            .prepare("SELECT version FROM migrations ORDER BY applied_at DESC LIMIT 1")?
            .query_row([], |row| {
                row.get::<_, String>(0)
            })
            .optional()
            .map_err(|e| MemvidError::Storage(format!("Failed to query current version: {}", e)))?;

        Ok(version)
    }

    /// Check if database is up to date
    pub fn is_up_to_date(&self) -> Result<bool> {
        let current_version = self.get_current_version()?;
        // Latest version is "add_search_indices"
        Ok(current_version.as_deref() == Some("add_search_indices"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_migration_manager_creation() {
        let manager = MigrationManager::new("test.db");
        assert_eq!(manager.db_path, "test.db");
    }

    #[test]
    fn test_initial_schema_creation() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let manager = MigrationManager::new(db_path.to_str().unwrap());
        
        let result = manager.run_migrations();
        assert!(result.is_ok());
        
        // Check that database was created
        assert!(db_path.exists());
        
        // Check that initial migration was recorded
        let version = manager.get_current_version().unwrap();
        assert!(version.is_some());
    }
} 