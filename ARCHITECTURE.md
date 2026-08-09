# Architecture Overview

This document provides a high-level overview of the architectural design and data flow of the GSEB Class 12 Multimodal AI Tutor.

## 🧩 System Architecture

The project is built on a modular architecture leveraging **LangChain** for orchestration, **ChromaDB** for vector storage, and **Mistral AI** for LLM and Embeddings. 

The architecture is divided into two main phases: **Data Ingestion (Offline)** and **Retrieval & Generation (Online)**.

### 1. Data Ingestion Pipeline (`create_database.py`)
This pipeline processes raw PDFs and populates the vector databases.

1. **Document Loading**: `modules/pdf_processor.py` iterates over PDF pages.
2. **Text Extraction**: Text is extracted, chunked using `RecursiveCharacterTextSplitter`, and converted to embeddings using `MistralAIEmbeddings`.
3. **Image Extraction & OCR**: `modules/image_extractor.py` extracts images/diagrams from pages based on size thresholds. `modules/ocr_processor.py` runs Tesseract OCR on these images to extract embedded text.
4. **Vector Storage**: `modules/embedding_manager.py` handles the creation of two separate ChromaDB collections:
   - `tuition_assistant` (Text chunks)
   - `tuition_images` (Image records + OCR text metadata)

### 2. Retrieval & Generation Pipeline (`app.py` / `main.py`)
This is the runtime pipeline that interacts with the user.

1. **Question Processing**: User input is combined with conversation history. `modules/answer_generator.py` uses the LLM to reformulate follow-up questions into standalone queries.
2. **Dual-Retrieval**: `modules/retriever.py` queries both ChromaDB collections concurrently.
   - It retrieves top-K text chunks.
   - It retrieves top-K image records.
3. **Diagram Intent Detection**: The system uses keyword heuristics (e.g., "diagram", "show", "figure") to detect if the user wants visual content.
4. **Answer Generation**:
   - **Standard Flow**: The LLM synthesizes an answer using the retrieved text context (`generate_text_answer`).
   - **Vision Flow (Optional)**: If a diagram is requested and Pixtral vision models are enabled, `modules/vision_pipeline.py` passes the actual image and text context to the Vision LLM for a multimodal response.
   - **Fallback Diagram Flow**: If vision is disabled, the LLM explains the diagram using the retrieved text and the OCR metadata of the image (`generate_diagram_answer`).
5. **Presentation**: The UI (`app.py`) displays the LLM's text response and renders the retrieved original textbook diagrams inline.

## 📁 Directory Structure Overview

```text
.
├── Document Loaders/       # Directory for raw source PDFs.
├── chroma_db/              # Local ChromaDB vector storage (generated).
├── database_images/        # Extracted images cropped from PDFs (generated).
├── modules/                # Core engine modules
│   ├── answer_generator.py # LLM orchestration for generating responses
│   ├── embedding_manager.py# ChromaDB collection management
│   ├── image_display.py    # UI utilities for cropping/rendering images
│   ├── image_extractor.py  # PyMuPDF logic to extract images from PDFs
│   ├── ocr_processor.py    # Tesseract OCR integrations
│   ├── pdf_processor.py    # PDF iteration and text extraction
│   ├── retriever.py        # LangChain logic for vector retrieval
│   └── vision_pipeline.py  # Mistral Pixtral multimodal integrations
├── app.py                  # Streamlit frontend application.
├── main.py                 # CLI frontend application.
├── create_database.py      # Ingestion script.
├── config.py               # Global configuration, paths, and feature flags.
├── Requirements.txt        # Python dependencies.
└── .env                    # Environment variables (API Keys).
```
