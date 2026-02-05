import argparse

parser = argparse.ArgumentParser(description="Constructing PRECOG")
parser.add_argument("--hf_ds", type=str, default="jungsoopark/LLMEvalDB", help="Hugging Face dataset identifier")
parser.add_argument("--cache_dir", type=str, help="Path to the cache directory")
parser.add_argument("--download_pdf", action='store_true', help="Whether to download PDF files")
parser.add_argument("--download_pdf_dir", type=str, default="data/LLMEvalDB_PDF_2025", help="Directory to save PDFs")
parser.add_argument("--trimmed_pdf_dir", type=str, default="data/LLMEvalDB_Trimmed_PDF_2025", help="Directory to save PDFs")
parser.add_argument("--processed_data_dir", type=str, default="data/LLMEvalDB_preprocessed", help="Directory to save processed dataset")
parser.add_argument("--loglevel", type=str, default='INFO', help="Logging level")

parser.add_argument("--year_suffix", type=str, default="25", help="Year suffix for constructing the PRECOG dataset")
parser.add_argument("--max_pages", type=int, default=12, help="Maximum number of pages to process per PDF")
parser.add_argument("--reasoning_effort", default="minimal", choices=["minimal", "low", "medium", "high"])
parser.add_argument("--model", default="gpt-5-mini", help="Model for /v1/responses")
parser.add_argument("--prompt_path", default="./prompts/filter_out_multimodal_paper.txt")
parser.add_argument("--prompt_path_benchmark", type=str, default="./prompts/extract_description_benchmark.txt", help="System prompt for benchmark (table == data source) descriptions")
parser.add_argument("--prompt_path_experimental", type=str, default="./prompts/extract_description_experiment.txt", help="System prompt for experimental (table != data source) descriptions")
parser.add_argument("--batch_jsonl_path", default="data/extraction_requests.jsonl")
parser.add_argument("--batch_cache_path", default="data/extraction_cache.jsonl")
parser.add_argument("--batch_output_path", default="data")

def get_args():
	return parser.parse_args()