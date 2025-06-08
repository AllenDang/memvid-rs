//! PDF text extraction functionality
//!
//! This module provides utilities for extracting text content from PDF files
//! using both pdf-extract and lopdf crates for maximum compatibility.

use crate::error::{MemvidError, Result};
use std::path::Path;

/// PDF text extraction processor
pub struct PdfProcessor;

impl PdfProcessor {
    /// Extract text from a PDF file
    pub fn extract_text<P: AsRef<Path>>(path: P) -> Result<String> {
        let path = path.as_ref();
        
        // Try using pdf-extract first (simpler)
        match pdf_extract::extract_text(path) {
            Ok(text) => Ok(text),
            Err(e) => {
                log::warn!("pdf-extract failed, trying lopdf: {}", e);
                Self::extract_with_lopdf(path)
            }
        }
    }

    /// Extract text using lopdf (fallback method)
    fn extract_with_lopdf<P: AsRef<Path>>(path: P) -> Result<String> {
        use lopdf::Document;
        
        let doc = Document::load(path)
            .map_err(|e| MemvidError::Pdf(format!("Failed to load PDF: {}", e)))?;
        
        let mut text = String::new();
        let pages = doc.get_pages();
        
        for (page_num, _) in pages {
            match doc.extract_text(&[page_num]) {
                Ok(page_text) => {
                    text.push_str(&page_text);
                    text.push_str("\n\n");
                }
                Err(e) => {
                    log::warn!("Failed to extract text from page {}: {}", page_num, e);
                }
            }
        }
        
        if text.trim().is_empty() {
            return Err(MemvidError::Pdf("No text extracted from PDF".to_string()));
        }
        
        Ok(text)
    }

    /// Extract text with page information
    pub fn extract_text_with_pages<P: AsRef<Path>>(path: P) -> Result<Vec<(u32, String)>> {
        use lopdf::Document;
        
        let doc = Document::load(path)
            .map_err(|e| MemvidError::Pdf(format!("Failed to load PDF: {}", e)))?;
        
        let mut pages_text = Vec::new();
        let pages = doc.get_pages();
        
        for (page_num, _) in pages {
            match doc.extract_text(&[page_num]) {
                Ok(page_text) => {
                    if !page_text.trim().is_empty() {
                        pages_text.push((page_num, page_text));
                    }
                }
                Err(e) => {
                    log::warn!("Failed to extract text from page {}: {}", page_num, e);
                }
            }
        }
        
        if pages_text.is_empty() {
            return Err(MemvidError::Pdf("No text extracted from PDF".to_string()));
        }
        
        Ok(pages_text)
    }

    /// Check if a file is a valid PDF
    pub fn is_pdf<P: AsRef<Path>>(path: P) -> bool {
        use std::fs::File;
        use std::io::Read;
        
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(_) => return false,
        };
        
        let mut buffer = [0; 4];
        match file.read_exact(&mut buffer) {
            Ok(_) => buffer == b"%PDF"[..],
            Err(_) => false,
        }
    }

    /// Get PDF metadata (page count, title, etc.)
    pub fn get_metadata<P: AsRef<Path>>(path: P) -> Result<PdfMetadata> {
        use lopdf::Document;
        
        let doc = Document::load(path)
            .map_err(|e| MemvidError::Pdf(format!("Failed to load PDF: {}", e)))?;
        
        let page_count = doc.get_pages().len() as u32;
        
        // Try to extract title from document info
        let title = Self::extract_title(&doc);
        
        Ok(PdfMetadata {
            page_count,
            title,
        })
    }

    /// Extract title from PDF document
    fn extract_title(doc: &lopdf::Document) -> Option<String> {
        // Try to get document info dictionary
        if let Ok(info_dict) = doc.trailer.get(b"Info") {
            if let Ok(info_ref) = info_dict.as_reference() {
                if let Ok(info_obj) = doc.get_object(info_ref) {
                    if let Ok(info_dict) = info_obj.as_dict() {
                        // Look for title field
                        if let Ok(title_obj) = info_dict.get(b"Title") {
                            if let Ok(title_bytes) = title_obj.as_str() {
                                if let Ok(title_string) = String::from_utf8(title_bytes.to_vec()) {
                                    return Some(title_string);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Fallback: try to extract from first few lines of text
        let pages = doc.get_pages();
        if let Some((page_num, _)) = pages.into_iter().next() {
            if let Ok(text) = doc.extract_text(&[page_num]) {
                let lines: Vec<&str> = text.lines().take(3).collect();
                for line in lines {
                    let trimmed = line.trim();
                    if trimmed.len() > 10 && trimmed.len() < 200 {
                        // Likely a title if it's reasonably sized
                        return Some(trimmed.to_string());
                    }
                }
            }
        }
        
        None
    }
}

/// PDF document metadata
#[derive(Debug, Clone)]
pub struct PdfMetadata {
    /// Number of pages in the PDF
    pub page_count: u32,
    
    /// Document title (if available)
    pub title: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_is_pdf_detection() {
        // Create a temporary file with PDF header
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "%PDF-1.4").unwrap();
        
        assert!(PdfProcessor::is_pdf(temp_file.path()));
    }

    #[test]
    fn test_non_pdf_detection() {
        // Create a temporary text file
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "This is not a PDF").unwrap();
        
        assert!(!PdfProcessor::is_pdf(temp_file.path()));
    }

    #[test]
    fn test_nonexistent_file() {
        assert!(!PdfProcessor::is_pdf("/nonexistent/file.pdf"));
    }
} 