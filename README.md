# Dataset Curator

[![CI](https://github.com/marlonbarreto-git/dataset-curator/actions/workflows/ci.yml/badge.svg)](https://github.com/marlonbarreto-git/dataset-curator/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A pipeline for creating, cleaning, and curating fine-tuning datasets with deduplication and quality scoring.

## Overview

Dataset Curator transforms raw text examples into clean, deduplicated training datasets in chat format. The pipeline chains together text cleaning (unicode normalization, whitespace handling), format conversion to chat messages, fuzzy deduplication using sequence matching, and quality scoring based on content length heuristics. It outputs datasets ready for LLM fine-tuning in JSONL format.

## Architecture

```
Raw Examples
     │
     v
┌─────────────┐    ┌────────────────┐    ┌──────────────┐
│ DataCleaner │───>│ FormatConverter│───>│ Deduplicator │
│             │    │                │    │              │
│ normalize   │    │ to chat format │    │ similarity   │
│ validate    │    │ system prompt  │    │ threshold    │
│ filter      │    │ JSONL export   │    │ 0.95 default │
└─────────────┘    └────────────────┘    └──────────────┘
                                                │
                                                v
                                    ┌───────────────────┐
                                    │ Quality Scoring   │
                                    │ + DatasetStats    │
                                    └───────────────────┘
```

## Features

- Unicode normalization (NFKC) and whitespace cleaning
- Length-based content validation (configurable min/max)
- Conversion to chat format with optional system prompts
- Fuzzy deduplication using SequenceMatcher (configurable threshold)
- Quality scoring based on content length heuristics
- JSONL export for fine-tuning frameworks
- Dataset statistics (total examples, avg quality, source breakdown, duplicates removed)

## Tech Stack

- Python 3.11+
- Pydantic >= 2.10

## Quick Start

```bash
git clone https://github.com/marlonbarreto-git/dataset-curator.git
cd dataset-curator
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Project Structure

```
src/dataset_curator/
  __init__.py
  pipeline.py      # CurationPipeline orchestrating all processing steps
  models.py        # RawExample, ChatMessage, TrainingExample, DatasetStats
  cleaner.py       # DataCleaner with unicode normalization and validation
  converter.py     # FormatConverter for chat format and JSONL export
  deduplicator.py  # Deduplicator with fuzzy matching
tests/
  test_pipeline.py
  test_models.py
  test_cleaner.py
  test_converter.py
  test_deduplicator.py
```

## Testing

```bash
pytest -v --cov=src/dataset_curator
```

51 tests covering text cleaning, format conversion, deduplication, pipeline integration, and model validation.

## License

MIT