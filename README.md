# Amazon Historic Text Bridge

## Project Overview

This project is a full-scale pipeline for extracting, structuring, and searching information from digitized historical texts related to the Amazon region.  
Unlike most solutions, this is not a Jupyter notebook but a modular, production-grade system built with FastAPI and PostgreSQL, focused on real archival sources rather than modern databases.

## Key Features

- **Full-text and semantic search** across a large corpus of old books (1500–1900), all cleaned and digitized.
- **Chunking and embedding:** Books are split into large, context-preserving fragments; embeddings are generated for each chunk to enable meaningful search.
- **Automated metadata extraction:** GPT-4/4o is used to extract key facts, locations, and names from every chunk.
- **Automatic translation:** All texts are translated to English, preserving the quirks of the original language.
- **Modular FastAPI endpoints:** All functionality is available via API endpoints, enabling flexible pipeline assembly for specific research tasks.
- **Storage:** Only PostgreSQL is used — all texts, embeddings, and metadata are stored in a single database.

## Architecture

- **FastAPI:** The core server; each pipeline stage is implemented as a separate endpoint.
- **PostgreSQL:** Unified storage for texts, embeddings, metadata, and search results.
- **LLM (OpenAI GPT-4/4o):** Used for OCR post-processing, translation, fact extraction, answer generation, and semantic reranking.
- **OCR:** Text recognition from page images is handled by Tesseract + LLM-assisted cleanup.

## Main Endpoints

- `/ocr_pdf_to_text` — digitize PDFs, extract text from images.
- `/clean_ocr` — clean and normalize text.
- `/refine_query` — refine user queries for better semantic retrieval.
- `/vector_search` — search across embeddings in the database.
- `/rerank_bge` — cross-encoder reranking of candidate chunks.
- `/rerank_semantic_v5` — final LLM-based semantic rerank.
- `/search_pipeline` — orchestrate the entire pipeline for end-to-end search.
- `/extract_facts` — extract structured facts from text fragments.
- `/general_knowledge` — LLM-generated background info (when relevant).
- `/web_search` — use external web search to augment answers when needed.

## Examples

See live demos and analysis in the Kaggle notebook:  
https://www.kaggle.com/code/antonsibilev/amazon-data-bridge-v2

## Limitations

- Some images are not fully processed by the LLM — pipeline coverage is not always 100%.
- Web search module is planned, not yet implemented.

## Roadmap

We are actively developing a controller and research LLM architecture for orchestrating deep search and reasoning across a multi-source pipeline. Upcoming milestones include a mixed-batch fact extractor, dynamic prompt generation, and advanced hypothesis building for historical data exploration.

## Contacts

**Anton Sibilev**  
https://www.kaggle.com/code/antonsibilev/amazon-data-bridge-v2  
Telegram: @anton_sibilev  
OpenAI & ChatGPT (co-authors)
