# PRECOG Data Curation Pipeline

This document provides a comprehensive guide to the PRECOG data curation pipeline for the 2025 dataset. The pipeline consists of four main stages that process evaluation datasets from LLMEvalDB to create a curated, well-documented collection.

## Prerequisites

Before running the curation pipeline, ensure you have:

1. **OpenAI API Key**: Required for LLM-based processing steps
2. **Working Directory**: Navigate to the PRECOG repository root
3. **Python Environment**: Python 3.8+ with required dependencies installed

### Setup

```bash
cd PRECOG
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

---

## Pipeline Overview

The curation pipeline consists of four sequential stages:

1. **Preprocess**: Download and preprocess datasets from HuggingFace, including PDF extraction
2. **Get Sources**: Identify and extract source paper information for each dataset
3. **Sample**: Create a representative sample of the processed datasets
4. **Get Descriptions**: Generate comprehensive descriptions for sampled datasets

---

## Stage 1: Preprocess

### Purpose
Downloads datasets from HuggingFace's LLMEvalDB, preprocesses the data, and extracts relevant PDF documents for downstream processing. 

### Key Operations
- Fetches datasets from the specified HuggingFace dataset repository
- Downloads associated PDF papers for each dataset
- Trims PDFs to relevant sections
- Uses OpenAI APIs for filtering out papers not of interest.

### Command

```bash
python3 -m src.curation.preprocess \
    --hf_ds jungsoopark/LLMEvalDB \
    --processed_data_dir data/PRECOG_2025/LLMEvalDB_preprocessed \
    --download_pdf \
    --download_pdf_dir data/PRECOG_2025/PDF_DIR \
    --trimmed_pdf_dir data/PRECOG_2025/TRIMMED_PDF_DIR \
    --year_suffix 25 \
    --batch_jsonl_path data/PRECOG_2025/preprocess_requests.jsonl \
    --batch_cache_path data/PRECOG_2025/preprocess_cache.jsonl \
    --batch_output_path data/PRECOG_2025 \
    --prompt_path ./prompts/filter_out_multimodal_paper.txt
```

### Parameters Explained
- `--hf_ds`: HuggingFace dataset identifier to process
- `--processed_data_dir`: Output directory for preprocessed dataset files
- `--download_pdf`: Flag to enable PDF downloading
- `--download_pdf_dir`: Directory to store original downloaded PDFs
- `--trimmed_pdf_dir`: Directory to store trimmed/processed PDFs
- `--year_suffix`: Year identifier (25 for 2025) for versioning
- `--batch_jsonl_path`: Path for batch API request files
- `--batch_cache_path`: Path for caching batch results
- `--batch_output_path`: Base directory for batch processing outputs

### Output
- Preprocessed datasets in `data/PRECOG_2025/LLMEvalDB_preprocessed`
- Downloaded PDFs in `data/PRECOG_2025/PDF_DIR`
- Trimmed PDFs in `data/PRECOG_2025/TRIMMED_PDF_DIR`

---

## Stage 2: Get Sources

### Purpose
Extracts and identifies the source papers and original publications for each dataset using LLM-based analysis of the PDF documents.

### Key Operations
- Analyzes preprocessed datasets and associated PDFs
- Uses OpenAI models to identify source papers
- Extracts paper titles, authors, and publication information
- Creates structured source metadata for each dataset

### Command

```bash
python3 -m src.curation.get_sources \
    --hf_ds data/PRECOG_2025/LLMEvalDB_preprocessed \
    --prompt_path ./prompts/get_dataset_source_paper_title.txt \
    --download_pdf_dir data/PRECOG_2025/PDF_DIR \
    --trimmed_pdf_dir data/PRECOG_2025/TRIMMED_PDF_DIR \
    --processed_data_dir data/PRECOG_2025/LLMEvalDB_sources \
    --batch_jsonl_path data/PRECOG_2025/sources_requests.jsonl \
    --batch_cache_path data/PRECOG_2025/sources_cache.jsonl \
    --batch_output_path data/PRECOG_2025 \
    --reasoning_effort medium
```

### Parameters Explained
- `--hf_ds`: Input directory containing preprocessed datasets
- `--prompt_path`: Path to the prompt template for source extraction
- `--download_pdf_dir`: Directory containing original PDFs
- `--trimmed_pdf_dir`: Directory containing trimmed PDFs
- `--processed_data_dir`: Output directory for datasets with source information
- `--batch_jsonl_path`: Path for batch API request files
- `--batch_cache_path`: Path for caching batch results
- `--batch_output_path`: Base directory for batch processing outputs
- `--reasoning_effort`: Level of reasoning effort for GPT models (low/medium/high)

### Output
- Datasets with source metadata in `data/PRECOG_2025/LLMEvalDB_sources`
- API request logs in `data/PRECOG_2025/sources_requests.jsonl`
- Cached results in `data/PRECOG_2025/sources_cache.jsonl`

---

## Stage 3: Sample

### Purpose
Creates a representative sample of the source-annotated datasets for efficient description generation in the final stage.

### Key Operations
- Selects a statistically representative subset of datasets
- Ensures diversity across different types and sources
- Reduces computational cost for the description generation phase

### Command

```bash
python3 -m src.curation.sample \
    --hf_ds data/PRECOG_2025/LLMEvalDB_sources \
    --processed_data_dir data/PRECOG_2025/LLMEvalDB_sources_sampled
```

### Parameters Explained
- `--hf_ds`: Input directory containing source-annotated datasets
- `--processed_data_dir`: Output directory for sampled datasets

### Output
- Sampled dataset subset in `data/PRECOG_2025/LLMEvalDB_sources_sampled`

---

## Stage 4: Get Descriptions

### Purpose
Generates comprehensive, high-quality descriptions for each sampled dataset using advanced LLM analysis with high reasoning effort.

### Key Operations
- Analyzes sampled datasets and their associated PDFs
- Generates detailed descriptions including:
  - Dataset purpose and evaluation objectives
  - Task types and formats
  - Key characteristics and statistics
- Uses high reasoning effort for maximum quality

### Command

```bash
python3 -m src.curation.get_descriptions \
    --hf_ds data/PRECOG_2025/LLMEvalDB_sources_sampled \
    --download_pdf_dir data/PRECOG_2025/PDF_DIR \
    --trimmed_pdf_dir data/PRECOG_2025/TRIMMED_PDF_DIR \
    --processed_data_dir data/PRECOG_2025/LLMEvalDB_descriptions \
    --batch_jsonl_path data/PRECOG_2025/descriptions_requests.jsonl \
    --batch_cache_path data/PRECOG_2025/descriptions_cache.jsonl \
    --batch_output_path data/PRECOG_2025 \
    --reasoning_effort high
```

### Parameters Explained
- `--hf_ds`: Input directory containing sampled datasets
- `--download_pdf_dir`: Directory containing original PDFs
- `--trimmed_pdf_dir`: Directory containing trimmed PDFs
- `--processed_data_dir`: Output directory for datasets with descriptions
- `--batch_jsonl_path`: Path for batch API request files
- `--batch_cache_path`: Path for caching batch results
- `--batch_output_path`: Base directory for batch processing outputs
- `--reasoning_effort`: Level of reasoning effort (set to `high` for best quality)

### Output
- Fully curated datasets with descriptions in `data/PRECOG_2025/LLMEvalDB_descriptions`
- API request logs in `data/PRECOG_2025/descriptions_requests.jsonl`
- Cached results in `data/PRECOG_2025/descriptions_cache.jsonl`

---

## Pipeline Execution

### Sequential Execution
Run each stage in order, ensuring each completes successfully before proceeding:

```bash
# Stage 1
python3 -m src.curation.preprocess [args...]

# Stage 2
python3 -m src.curation.get_sources [args...]

# Stage 3
python3 -m src.curation.sample [args...]

# Stage 4
python3 -m src.curation.get_descriptions [args...]
```

### Notes
- Each stage depends on the output of the previous stage
- Batch processing uses caching to resume from failures
- All outputs are stored in `data/PRECOG_2025/` with appropriate subdirectories

---

## Output Structure

After running the full pipeline, your `data/PRECOG_2025/` directory will contain:

```
PRECOG_2025/
├── LLMEvalDB_preprocessed/      # Stage 1 output
├── LLMEvalDB_sources/           # Stage 2 output
├── LLMEvalDB_sources_sampled/   # Stage 3 output
├── LLMEvalDB_descriptions/      # Stage 4 output (final)
├── PDF_DIR/                     # Original PDFs
├── TRIMMED_PDF_DIR/             # Processed PDFs
├── preprocess_*.jsonl           # Stage 1 batch files
├── sources_*.jsonl              # Stage 2 batch files
└── descriptions_*.jsonl         # Stage 4 batch files
```