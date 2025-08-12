"""
Document processing module for handling PDF, DOCX, and TXT files.
"""

import fitz  # PyMuPDF
from docx import Document
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from config import (
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    MAX_CHUNKS_PER_DOCUMENT,
    SUPPORTED_FILE_TYPES,
    MAX_FILE_SIZE_BYTES
)

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing and chunking."""
    
    def __init__(self):
        self.supported_types = SUPPORTED_FILE_TYPES
        self.max_file_size = MAX_FILE_SIZE_BYTES
    
    def validate_file(self, file_path: str, file_size: int) -> bool:
        """
        Validate file type and size.
        
        Args:
            file_path: Path to the file
            file_size: Size of the file in bytes
            
        Returns:
            bool: True if file is valid
        """
        # Check file size
        if file_size > self.max_file_size:
            raise ValueError(f"File size {file_size} exceeds maximum allowed size {self.max_file_size}")
        
        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_types:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        return True
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Extracted text
        """
        try:
            doc = fitz.open(file_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            
            doc.close()
            return text
        
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            raise
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """
        Extract text from DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            str: Extracted text
        """
        try:
            doc = Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text
        
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
            raise
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """
        Extract text from TXT file.
        
        Args:
            file_path: Path to the TXT file
            
        Returns:
            str: Extracted text
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {str(e)}")
            raise
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from file based on its type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: Extracted text
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def create_chunks(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Create chunks from text with overlap.
        
        Args:
            text: Text to chunk
            doc_id: Document ID
            
        Returns:
            List[Dict[str, Any]]: List of chunks with metadata
        """
        chunks = []
        start = 0
        
        while start < len(text) and len(chunks) < MAX_CHUNKS_PER_DOCUMENT:
            end = start + CHUNK_SIZE
            
            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings
                sentence_endings = ['.', '!', '?', '\n\n']
                for ending in sentence_endings:
                    pos = text.rfind(ending, start, end)
                    if pos > start + CHUNK_SIZE // 2:  # Only break if we're not too early
                        end = pos + 1
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunk = {
                    "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                    "doc_id": doc_id,
                    "content": chunk_text,
                    "start_pos": start,
                    "end_pos": end,
                    "chunk_index": len(chunks),
                    "metadata": {
                        "chunk_size": len(chunk_text),
                        "created_at": datetime.utcnow().isoformat()
                    }
                }
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - CHUNK_OVERLAP
            if start >= len(text):
                break
        
        return chunks
    
    def process_document(self, file_path: str, file_size: int) -> Dict[str, Any]:
        """
        Process a document and return chunks.
        
        Args:
            file_path: Path to the document
            file_size: Size of the file in bytes
            
        Returns:
            Dict[str, Any]: Document processing results
        """
        try:
            # Validate file
            self.validate_file(file_path, file_size)
            
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Extract text
            text = self.extract_text(file_path)
            
            # Create chunks
            chunks = self.create_chunks(text, doc_id)
            
            # Prepare result
            result = {
                "doc_id": doc_id,
                "filename": os.path.basename(file_path),
                "file_type": os.path.splitext(file_path)[1].lower(),
                "file_size": file_size,
                "text_length": len(text),
                "chunk_count": len(chunks),
                "chunks": chunks,
                "processed_at": datetime.utcnow().isoformat(),
                "status": "processed"
            }
            
            logger.info(f"Processed document {file_path}: {len(chunks)} chunks created")
            return result
        
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    def process_uploaded_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process an uploaded file from memory.
        
        Args:
            file_content: File content as bytes
            filename: Name of the file
            
        Returns:
            Dict[str, Any]: Document processing results
        """
        try:
            # Validate file size
            file_size = len(file_content)
            if file_size > self.max_file_size:
                raise ValueError(f"File size {file_size} exceeds maximum allowed size {self.max_file_size}")
            
            # Create temporary file
            temp_path = f"/tmp/{filename}"
            with open(temp_path, 'wb') as f:
                f.write(file_content)
            
            try:
                # Process the document
                result = self.process_document(temp_path, file_size)
                return result
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        except Exception as e:
            logger.error(f"Error processing uploaded file {filename}: {str(e)}")
            raise
