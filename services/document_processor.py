print("[DocumentProcessor] Script started.")
"""Document processing module for RAG pipeline."""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
import time
import re

class DocumentProcessor:
    """Process documents for indexing and retrieval."""

    def __init__(self, docs_dir: str = "./docs"):
        """Initialize the document processor.

        Args:
            docs_dir: Directory containing documents
        """
        self.docs_dir = Path(docs_dir)

    def load_text_documents(self, subdirs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Load text documents from directory.

        Args:
            subdirs: Optional list of subdirectories to load from

        Returns:
            List of document dictionaries
        """
        documents = []
        dirs_to_scan = []
        if subdirs:
            for subdir in subdirs:
                subdir_path = self.docs_dir / subdir
                if subdir_path.exists() and subdir_path.is_dir():
                    dirs_to_scan.append(subdir_path)
        else:
            dirs_to_scan.append(self.docs_dir)
            for item in self.docs_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    dirs_to_scan.append(item)
        print(f"[DEBUG] Scanning for text documents in: {[str(d) for d in dirs_to_scan]}")
        # Process each directory
        for dir_path in dirs_to_scan:
            print(f"[DEBUG] Listing *.txt files in {dir_path} ...")
            for file_path in dir_path.glob("*.txt"):
                print(f"[DEBUG] Found text file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    document = {
                        "content": content,
                        "metadata": {
                            "source": file_path.name,
                            "category": dir_path.name if dir_path != self.docs_dir else "general"
                        }
                    }
                    documents.append(document)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")

        return documents

    def load_qa_pairs(self, subdirs: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """Load question-answer pairs from JSON files.

        Args:
            subdirs: Optional list of subdirectories to load from

        Returns:
            List of QA pair dictionaries
        """
        qa_pairs = []
        dirs_to_scan = []
        if subdirs:
            for subdir in subdirs:
                subdir_path = self.docs_dir / subdir
                if subdir_path.exists() and subdir_path.is_dir():
                    dirs_to_scan.append(subdir_path)
        else:
            for item in self.docs_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    dirs_to_scan.append(item)
        print(f"[DEBUG] Scanning for QA pairs in: {[str(d) for d in dirs_to_scan]}")
        # Process each directory
        for dir_path in dirs_to_scan:
            print(f"[DEBUG] Listing *_qa_pairs.json files in {dir_path} ...")
            for file_path in dir_path.glob("*_qa_pairs.json"):
                print(f"[DEBUG] Found QA file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    for qa_pair in data:
                        qa_pair["source"] = file_path.name
                        qa_pairs.append(qa_pair)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")

        return qa_pairs

    def chunk_document(self, document: Dict[str, Any], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
        """Split document into paragraph/section-based chunks, fallback to fixed-size for long sections."""
        content = document["content"]
        metadata = document.get("metadata", {}).copy()
        # Only chunk .txt files, skip JSON
        source = metadata.get("source", "")
        if source.endswith(".json"):
            return [{
                "content": content,
                "metadata": {
                    **metadata,
                    "chunk_id": str(uuid.uuid4())
                }
            }]
        # Split by double newline or section headers
        sections = re.split(r'(?:\n\s*\n|^#.*$)', content, flags=re.MULTILINE)
        chunks = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= chunk_size:
                chunks.append({
                    "content": section,
                    "metadata": {**metadata, "chunk_id": str(uuid.uuid4())}
                })
            else:
                start = 0
                while start < len(section):
                    end = min(start + chunk_size, len(section))
                    chunk_text = section[start:end]
                    chunks.append({
                        "content": chunk_text,
                        "metadata": {**metadata, "chunk_id": str(uuid.uuid4()), "chunk_start": start, "chunk_end": end}
                    })
                    if end == len(section):
                        break
                    start = end - chunk_overlap
        return chunks

if __name__ == "__main__":
    print("[DocumentProcessor] Debug: Listing files in ./docs ...")
    docs_dir = "./docs"
    if os.path.exists(docs_dir):
        for fname in os.listdir(docs_dir):
            print(f" - {fname}")
    else:
        print("[DocumentProcessor] ./docs does not exist!")
    print("[DocumentProcessor] Debug: Listing files in ./docs/tech ...")
    tech_dir = os.path.join(docs_dir, "tech")
    if os.path.exists(tech_dir):
        for fname in os.listdir(tech_dir):
            print(f"   - {fname}")
    else:
        print("[DocumentProcessor] ./docs/tech does not exist!")
    print("[DocumentProcessor] Running basic tests...")
    processor = DocumentProcessor()
    print("\n--- Loading text documents ---")
    docs = processor.load_text_documents()
    print(f"Loaded {len(docs)} text documents.")
    for i, doc in enumerate(docs[:2]):
        print(f"Doc {i+1} (category: {doc['metadata'].get('category')}): {doc['metadata'].get('source')}")
        print(f"Content preview: {doc['content'][:100]}\n---")

    print("\n--- Loading QA pairs ---")
    qa_pairs = processor.load_qa_pairs()
    print(f"Loaded {len(qa_pairs)} QA pairs.")
    for i, qa in enumerate(qa_pairs[:2]):
        print(f"QA {i+1} (source: {qa.get('source')}): Q: {qa.get('question')} | A: {qa.get('answer')}")

    if docs:
        print("\n--- Chunking first document ---")
        t0 = time.time()
        chunks = processor.chunk_document(docs[0])
        t1 = time.time()
        print(f"First document split into {len(chunks)} chunks in {t1-t0:.6f} seconds.")
        for i, chunk in enumerate(chunks[:2]):
            print(f"Chunk {i+1} (chars {chunk['metadata']['chunk_start']}:{chunk['metadata']['chunk_end']}): {chunk['content'][:60]}...")
    print("[DocumentProcessor] Basic tests complete.")
