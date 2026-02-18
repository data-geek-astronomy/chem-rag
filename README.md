# chem-rag

GenAI application that extracts chemical properties (boiling points, flash points, flammability limits) from technical literature using **Retrieval-Augmented Generation (RAG)** with **Google Gemini** and **hybrid search**.

## Overview

Processes millions of pages of chemical documents (SDS sheets, handbooks, technical literature) and extracts structured property data on demand. Built with a hybrid search architecture that combines vector similarity with keyword re-ranking to maximize retrieval precision.

**Evaluation results on benchmark compounds:**

| Compound | Precision | Recall | F1 | Delta (BP) |
|----------|-----------|--------|----|------------|
| Ethanol  | 100% | 100% | **1.000** | ±0.03°C |
| Acetone  | 100% | 100% | **1.000** | ±0.05°C |

## Architecture

```
PDF Documents
     │
     ▼
┌─────────────┐     pdfplumber + PyMuPDF
│   ingest.py │  ── table-aware extraction
│             │  ── sentence-boundary chunking
│             │  ── 512 token chunks, 64 overlap
└──────┬──────┘
       │ chunks
       ▼
┌─────────────┐     Gemini gemini-embedding-001
│   embed.py  │  ── 3072-dimensional embeddings
│             │  ── ChromaDB (cosine similarity)
│             │  ── idempotent (skip existing)
└──────┬──────┘
       │ vector store
       ▼
┌─────────────┐     Hybrid Search
│   embed.py  │  ── vector similarity (HNSW)
│ hybrid_search│ ── keyword re-ranking (+0.15 boost)
└──────┬──────┘
       │ top-k chunks
       ▼
┌─────────────┐     Gemini gemini-2.5-flash
│  extract.py │  ── JSON schema-guided extraction
│             │  ── Pydantic validation
│             │  ── unit normalization (°F/K → °C)
└──────┬──────┘
       │ ChemicalProperties
       ▼
┌──────────────┐
│  evaluate.py │  ── Precision / Recall / F1
│              │  ── ±tolerance matching
│              │  ── ground truth CSV
└──────────────┘
```

## Extracted Properties

For each compound:

| Property | Type |
|----------|------|
| Boiling point | °C (normalized) |
| Melting point | °C (normalized) |
| Flash point | °C (closed cup) |
| Flammability class | NFPA 30 (IA / IB / IC / II / III) |
| LEL / UEL | % v/v in air |
| Molecular weight | g/mol |
| Density | g/cm³ |
| Vapor pressure | kPa at temperature |
| Solubility | qualitative |
| CAS number | string |
| Confidence score | 0.0 – 1.0 |

## Setup

**Requirements:** Python 3.12+, pip

```bash
# 1. Clone the repo
git clone https://github.com/data-geek-astronomy/chem-rag.git
cd chem-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
cp .env.template .env
# Edit .env and add your GEMINI_API_KEY from https://aistudio.google.com
```

## Configuration

Edit `.env` to configure models and chunking:

```env
GEMINI_API_KEY=your-key-here

# Models
EMBEDDING_MODEL=models/gemini-embedding-001
CHAT_MODEL=models/gemini-2.5-flash

# ChromaDB storage path
CHROMA_PERSIST_DIR=./data/chroma_db

# Chunking
CHUNK_SIZE=512       # tokens per chunk
CHUNK_OVERLAP=64     # overlap between chunks
```

## Usage

### 1. Add documents

Drop PDF files (SDS sheets, chemical handbooks, technical manuals) into `data/raw/`:

```
data/raw/
├── ethanol_sds.pdf
├── solvent_handbook.pdf
└── msds_collection.pdf
```

### 2. Ingest

Parse and embed all PDFs into the vector store:

```bash
python src/main.py ingest

# Or a single file:
python src/main.py ingest --pdf data/raw/ethanol_sds.pdf
```

### 3. Query

Extract properties for a compound:

```bash
python src/main.py query "Ethanol"
python src/main.py query "Acetone"
python src/main.py query "Diethyl Ether"

# Show retrieved chunks before extraction:
python src/main.py query "Toluene" --show-chunks
```

**Example output:**
```
Chemical: Ethanol
CAS: 64-17-5
Confidence: 90%

Physical Properties:
  Boiling Point: 78.4 °C — at 1 atm
  Melting Point: -114.1 °C
  Flash Point: 13.0 °C — Closed cup
  Molecular Weight: 46.07 g/mol
  Density: 0.789 g/cm³
  Vapor Pressure: 5.95 kPa (at 20 °C)
  Solubility: Miscible with water in all proportions

Flammability:
  Flammable: True
  Flash Point: 13.0 °C — Closed cup
  Class: Class IB Flammable Liquid
  LEL: 3.3%  UEL: 19.0%
```

### 4. Evaluate

Compare extracted values against ground truth:

```bash
python src/main.py evaluate "Ethanol"
python src/main.py evaluate "Acetone"
```

### 5. Stats

Show index statistics:

```bash
python src/main.py stats
```

## Ground Truth

The evaluation CSV lives at `data/ground_truth/ground_truth.csv`. Add entries to benchmark more compounds:

```csv
compound_name,property_name,expected_value,tolerance,source
Ethanol,boiling_point_celsius,78.37,2.0,NIST
Ethanol,flash_point_celsius,13.0,2.0,NIST
Methanol,boiling_point_celsius,64.7,2.0,NIST
```

Supported `property_name` values: `boiling_point_celsius`, `melting_point_celsius`, `flash_point_celsius`, `molecular_weight`, `density`

## Project Structure

```
chem-rag/
├── src/
│   ├── main.py        # CLI entrypoint (Typer)
│   ├── ingest.py      # PDF parsing + chunking
│   ├── embed.py       # Gemini embeddings + ChromaDB + hybrid search
│   ├── extract.py     # LLM property extraction (Pydantic models)
│   └── evaluate.py    # Precision/Recall/F1 evaluation
├── data/
│   ├── raw/           # Drop PDFs here (gitignored)
│   ├── chroma_db/     # Vector store (gitignored, rebuilt via ingest)
│   └── ground_truth/
│       └── ground_truth.csv
├── notebooks/         # Exploration notebooks
├── .env.template      # Config template
├── requirements.txt
└── README.md
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Gemini Embedding 001 (3072-dim) |
| Vector store | ChromaDB (local, persistent) |
| PDF parsing | pdfplumber + PyMuPDF |
| Data validation | Pydantic v2 |
| CLI | Typer + Rich |

## Notes

- The vector store is **local** — no data leaves your machine except for Gemini API calls
- Ingestion is **idempotent** — re-running skips already-indexed chunks
- Supports PDFs in any format: text-based, scanned (via PyMuPDF fallback), and table-heavy SDS sheets
- Temperature values are automatically normalized to Celsius from °F or K
