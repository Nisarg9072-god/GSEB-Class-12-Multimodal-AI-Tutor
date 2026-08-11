# GSEB Class 12 Multimodal AI Tutor !🎓

A powerful, multimodal Retrieval-Augmented Generation (RAG) educational assistant tailored for GSEB Class 12 students (Physics, Chemistry, and Biology). This project leverages Mistral AI, ChromaDB, and LangChain to provide highly accurate answers and intelligently retrieve inline diagrams from textbook PDFs.

## ✨ Features
- **Multimodal RAG Pipeline**: Extracts both text and diagrams/images from PDF textbooks.
- **Context-Aware Answers**: Remembers chat history (up to 10 turns) to reformulate follow-up questions accurately.
- **Diagram & Image Retrieval**: Automatically retrieves and displays original textbook diagrams when users ask visually oriented questions (e.g., "Show me the diagram of a human heart").
- **Optical Character Recognition (OCR)**: Uses Tesseract OCR to index text within extracted textbook images, making diagrams searchable.
- **Dual Interfaces**: 
  - **Web Interface**: Beautiful, responsive Streamlit web app.
  - **CLI Interface**: Terminal-based interactive tutor.
- **Vision LLM Support (Optional)**: Capable of using Mistral's Pixtral vision models to analyze retrieved images and answer questions based on the visual content.

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (Optional, but recommended for indexing text inside images)
- A [Mistral AI API Key](https://console.mistral.ai/)

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone <your-repo-url>
cd <your-repo-name>

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r Requirements.txt
```

### 3. Configuration
1. Create a `.env` file in the project root and add your Mistral API Key:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
```
2. Adjust any model settings or thresholds in `config.py` (e.g., enabling Vision LLM or OCR).

### 4. Build the Database
Place your textbook PDFs into the `Document Loaders` directory. Then, build the ChromaDB vector database:
```bash
python create_database.py
```
*(This extracts text, crops images, runs OCR, computes embeddings, and stores them in ChromaDB. You only need to run this once or when adding new PDFs).*

### 5. Run the Application
You can run the tutor via the sleek web interface or the command line.

**Option A: Web Interface (Recommended)**
```bash
# On Windows, you can simply double-click START_TUTOR.bat
# Or run manually via terminal:
streamlit run app.py
```

**Option B: CLI Interface**
```bash
python main.py
```
