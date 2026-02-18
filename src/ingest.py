"""
ingest.py - PDF parsing and semantic chunking for chemical documents.

Handles:
- Text extraction from PDFs (body text + tables)
- Semantic chunking (respects sentence boundaries)
- Metadata preservation (source, page number, section)
"""

import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field

import pdfplumber
import fitz  # PyMuPDF
import tiktoken
from rich.console import Console
from rich.progress import track

console = Console()

TOKENIZER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """A single chunk of text ready for embedding."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    token_count: int
    metadata: dict = field(default_factory=dict)


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract text page-by-page using pdfplumber (better for tables)
    with PyMuPDF as fallback for scanned/complex layouts.
    Returns list of {page_number, text, tables}.
    """
    pages = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_data = {"page_number": i + 1, "text": "", "tables": []}

                # Extract tables first (chemical properties often live here)
                raw_tables = page.extract_tables()
                for table in raw_tables:
                    if table:
                        table_text = _table_to_text(table)
                        page_data["tables"].append(table_text)

                # Extract body text (excluding table bounding boxes)
                body_text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                page_data["text"] = body_text.strip()

                pages.append(page_data)

    except Exception as e:
        console.print(f"[yellow]pdfplumber failed ({e}), falling back to PyMuPDF[/yellow]")
        pages = _extract_with_pymupdf(pdf_path)

    return pages


def _table_to_text(table: list[list]) -> str:
    """Convert a table (list of rows) to readable text preserving structure."""
    lines = []
    for row in table:
        cleaned = [str(cell).strip() if cell else "" for cell in row]
        lines.append(" | ".join(cleaned))
    return "\n".join(lines)


def _extract_with_pymupdf(pdf_path: Path) -> list[dict]:
    """Fallback extractor using PyMuPDF."""
    pages = []
    doc = fitz.open(str(pdf_path))
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page_number": i + 1, "text": text.strip(), "tables": []})
    doc.close()
    return pages


def chunk_page(
    page_data: dict,
    source_file: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """
    Split page text into overlapping chunks that respect sentence boundaries.
    Tables are kept as single chunks (don't split mid-table).
    """
    chunks = []
    chunk_index = 0
    page_num = page_data["page_number"]

    # First: add each table as its own chunk (never split tables)
    for table_text in page_data["tables"]:
        if table_text.strip():
            chunk = _make_chunk(
                text=f"[TABLE]\n{table_text}",
                source_file=source_file,
                page_number=page_num,
                chunk_index=chunk_index,
                metadata={"type": "table"},
            )
            chunks.append(chunk)
            chunk_index += 1

    # Second: chunk the body text
    body_text = page_data["text"]
    if not body_text:
        return chunks

    sentences = _split_sentences(body_text)
    current_tokens = 0
    current_sentences = []

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        if current_tokens + sentence_tokens > chunk_size and current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunk = _make_chunk(
                    text=chunk_text,
                    source_file=source_file,
                    page_number=page_num,
                    chunk_index=chunk_index,
                    metadata={"type": "text"},
                )
                chunks.append(chunk)
                chunk_index += 1

            # Overlap: keep last N tokens worth of sentences
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                s_tok = count_tokens(s)
                if overlap_tokens + s_tok <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tok
                else:
                    break

            current_sentences = overlap_sentences + [sentence]
            current_tokens = overlap_tokens + sentence_tokens
        else:
            current_sentences.append(sentence)
            current_tokens += sentence_tokens

    # Flush remaining
    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        if chunk_text:
            chunk = _make_chunk(
                text=chunk_text,
                source_file=source_file,
                page_number=page_num,
                chunk_index=chunk_index,
                metadata={"type": "text"},
            )
            chunks.append(chunk)

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex (fast, no NLTK dependency)."""
    text = re.sub(r'\s+', ' ', text)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
    return [s.strip() for s in sentences if s.strip()]


def _make_chunk(text, source_file, page_number, chunk_index, metadata) -> Chunk:
    chunk_id = hashlib.md5(
        f"{source_file}:{page_number}:{chunk_index}:{text[:50]}".encode()
    ).hexdigest()
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source_file=source_file,
        page_number=page_number,
        chunk_index=chunk_index,
        token_count=count_tokens(text),
        metadata=metadata,
    )


def ingest_pdf(pdf_path: Path, chunk_size: int = 512, chunk_overlap: int = 64) -> list[Chunk]:
    """Full pipeline: PDF -> pages -> chunks."""
    source_file = pdf_path.name
    console.print(f"[cyan]Ingesting:[/cyan] {source_file}")

    pages = extract_text_from_pdf(pdf_path)
    all_chunks = []

    for page_data in track(pages, description=f"  Chunking {source_file}..."):
        chunks = chunk_page(page_data, source_file, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    console.print(
        f"  [green]Done:[/green] {len(pages)} pages → {len(all_chunks)} chunks "
        f"(avg {sum(c.token_count for c in all_chunks) // max(len(all_chunks), 1)} tokens/chunk)"
    )
    return all_chunks


def ingest_directory(data_dir: Path, chunk_size: int = 512, chunk_overlap: int = 64) -> list[Chunk]:
    """Ingest all PDFs in a directory."""
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[red]No PDFs found in {data_dir}[/red]")
        return []

    console.print(f"[bold]Found {len(pdf_files)} PDF(s) to ingest[/bold]")
    all_chunks = []
    for pdf_path in pdf_files:
        chunks = ingest_pdf(pdf_path, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    console.print(f"\n[bold green]Total: {len(all_chunks)} chunks from {len(pdf_files)} files[/bold green]")
    return all_chunks
