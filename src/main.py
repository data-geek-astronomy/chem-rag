"""
main.py - CLI entrypoint for the Chemical RAG pipeline.

Commands:
  ingest   - Parse PDFs and index into Chroma
  query    - Extract properties for a specific compound
  evaluate - Run accuracy evaluation against ground truth
  stats    - Show index statistics
  setup    - Initialize ground truth template and .env file
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Load .env before importing anything else
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import typer
from rich.console import Console

# Add src to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent))

from ingest import ingest_directory, ingest_pdf
from embed import get_chroma_client, get_collection, embed_and_store, hybrid_search, get_collection_stats
from extract import extract_properties, format_extraction_report
from evaluate import load_ground_truth, evaluate_extraction, print_eval_report, save_ground_truth_template

app = typer.Typer(help="Chemical Property RAG Pipeline", add_completion=False)
console = Console()

# ---------------------------------------------------------------------------
# Config defaults (override via .env)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.0-flash")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
GT_PATH = Path(__file__).parent.parent / "data" / "ground_truth" / "ground_truth.csv"


def _require_api_key():
    if not os.getenv("GEMINI_API_KEY"):
        console.print("[red]Error: GEMINI_API_KEY not set. Copy .env.template to .env and add your key.[/red]")
        raise typer.Exit(1)


def _get_collection():
    client = get_chroma_client(CHROMA_DIR)
    return get_collection(client, EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def setup():
    """Initialize .env file and ground truth template."""
    env_path = Path(__file__).parent.parent / ".env"
    template_path = Path(__file__).parent.parent / ".env.template"

    if not env_path.exists():
        import shutil
        shutil.copy(template_path, env_path)
        console.print(f"[green]Created .env from template.[/green] Edit it and add your OPENAI_API_KEY.")
    else:
        console.print("[yellow].env already exists.[/yellow]")

    GT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not GT_PATH.exists():
        save_ground_truth_template(GT_PATH)
    else:
        console.print(f"[yellow]Ground truth file already exists at {GT_PATH}[/yellow]")

    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. Add your OpenAI API key to [cyan].env[/cyan]")
    console.print("2. Drop PDF files into [cyan]data/raw/[/cyan]")
    console.print("3. Run: [cyan]python src/main.py ingest[/cyan]")
    console.print("4. Run: [cyan]python src/main.py query 'Ethanol'[/cyan]")


@app.command()
def ingest(
    pdf: Optional[str] = typer.Option(None, help="Ingest a single PDF file"),
    data_dir: str = typer.Option(str(DATA_DIR), help="Directory containing PDFs"),
    chunk_size: int = typer.Option(CHUNK_SIZE, help="Max tokens per chunk"),
    chunk_overlap: int = typer.Option(CHUNK_OVERLAP, help="Overlap tokens between chunks"),
):
    """Parse PDFs and index them into the vector store."""
    _require_api_key()
    collection = _get_collection()

    if pdf:
        pdf_path = Path(pdf)
        if not pdf_path.exists():
            console.print(f"[red]File not found: {pdf}[/red]")
            raise typer.Exit(1)
        from ingest import ingest_pdf
        chunks = ingest_pdf(pdf_path, chunk_size, chunk_overlap)
    else:
        dir_path = Path(data_dir)
        if not dir_path.exists():
            console.print(f"[red]Directory not found: {data_dir}[/red]")
            raise typer.Exit(1)
        chunks = ingest_directory(dir_path, chunk_size, chunk_overlap)

    if chunks:
        embed_and_store(chunks, collection)
    else:
        console.print("[yellow]No chunks produced. Check your PDF files.[/yellow]")


@app.command()
def query(
    compound: str = typer.Argument(..., help="Chemical compound name to query"),
    n_results: int = typer.Option(10, help="Number of context chunks to retrieve"),
    show_chunks: bool = typer.Option(False, help="Print retrieved chunks before extraction"),
):
    """Extract chemical properties for a compound from indexed documents."""
    _require_api_key()
    collection = _get_collection()

    stats = get_collection_stats(collection)
    if stats["total_chunks"] == 0:
        console.print("[red]Index is empty. Run 'ingest' first.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Searching for:[/bold] {compound}")
    console.print(f"[dim]Index: {stats['total_chunks']} chunks from {len(stats['sources'])} source(s)[/dim]\n")

    # Retrieve relevant chunks
    search_query = f"{compound} boiling point flash point flammability melting point physical properties"
    chunks = hybrid_search(search_query, collection, n_results=n_results)

    if show_chunks:
        console.print(f"\n[bold]Retrieved {len(chunks)} chunks:[/bold]")
        for i, c in enumerate(chunks):
            console.print(f"\n[dim]--- Chunk {i+1} | Score: {c.get('hybrid_score', c['score']):.3f} | {c['source_file']} p.{c['page_number']} ---[/dim]")
            console.print(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))

    # Extract properties via LLM
    props = extract_properties(compound, chunks, model=CHAT_MODEL)

    # Display results
    console.print(format_extraction_report(props))

    return props


@app.command()
def evaluate(
    compound: str = typer.Argument(..., help="Compound to evaluate (must exist in ground truth CSV)"),
    gt_file: str = typer.Option(str(GT_PATH), help="Path to ground truth CSV"),
):
    """Evaluate extraction accuracy against ground truth data."""
    _require_api_key()

    gt_path = Path(gt_file)
    if not gt_path.exists():
        console.print(f"[red]Ground truth file not found: {gt_path}[/red]")
        console.print("Run 'python src/main.py setup' to create a template.")
        raise typer.Exit(1)

    ground_truths = load_ground_truth(gt_path)
    compound_gts = [g for g in ground_truths if g.compound_name.lower() == compound.lower()]

    if not compound_gts:
        console.print(f"[red]No ground truth entries found for '{compound}'[/red]")
        console.print(f"Available compounds: {list(set(g.compound_name for g in ground_truths))}")
        raise typer.Exit(1)

    # Run extraction
    collection = _get_collection()
    search_query = f"{compound} boiling point flash point flammability melting point physical properties"
    chunks = hybrid_search(search_query, collection, n_results=10)
    props = extract_properties(compound, chunks, model=CHAT_MODEL)

    # Evaluate
    report = evaluate_extraction(props, compound_gts)
    print_eval_report(report, title=f"Evaluation: {compound}")


@app.command()
def stats():
    """Show index statistics."""
    _require_api_key()
    collection = _get_collection()
    info = get_collection_stats(collection)

    console.print(f"\n[bold cyan]Index Statistics[/bold cyan]")
    console.print(f"  Total chunks: [bold]{info['total_chunks']}[/bold]")
    console.print(f"  Sources ({len(info['sources'])}):")
    for src in info["sources"]:
        console.print(f"    - {src}")


if __name__ == "__main__":
    app()
