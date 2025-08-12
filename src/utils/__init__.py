"""
Utils package for Claimsure.

Contains utility functions for document processing, file operations, and other helpers.
"""

__version__ = "1.0.0"
__author__ = "Claimsure Team"

from .document_loader import (
    load_pdf,
    load_docx,
    load_email,
    load_document,
    clean_text,
    validate_file_path,
    get_file_size
)

from .chunking import (
    semantic_chunk,
    split_into_paragraphs,
    split_into_sentences,
    split_into_words,
    create_chunks_from_segments,
    find_good_split_point,
    validate_chunk_parameters,
    get_chunk_statistics,
    merge_small_chunks
)

__all__ = [
    "load_pdf",
    "load_docx", 
    "load_email",
    "load_document",
    "clean_text",
    "validate_file_path",
    "get_file_size",
    "semantic_chunk",
    "split_into_paragraphs",
    "split_into_sentences",
    "split_into_words",
    "create_chunks_from_segments",
    "find_good_split_point",
    "validate_chunk_parameters",
    "get_chunk_statistics",
    "merge_small_chunks"
]
