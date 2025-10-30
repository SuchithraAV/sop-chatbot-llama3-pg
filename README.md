# SOP Chatbot (Ollama + PostgreSQL + pgvector)

This repository is a Retrieval-Augmented Generation (RAG) chatbot that:
- Accepts document uploads (.txt, .md, .pdf, .docx, images, and legacy .doc with optional tooling)
- Stores chunked text and embeddings in PostgreSQL with the pgvector extension
- Uses SentenceTransformers to create embeddings
- Queries a local Ollama LLM (e.g. Llama 3) for answers using retrieved context

Default Ollama model used by the backend: `llama3:latest` (set via `OLLAMA_MODEL`).

## Quickstart (recommended)
Below are the steps to run the project locally. There are two folders: `backend` (FastAPI) and `frontend` (React/Vite).

Prereqs
- Python 3.10+ (create and use a virtualenv)
- Node.js + npm (for frontend)
- Docker (optional - for local Postgres)
- Ollama installed and running (see Ollama section)

1) Start Postgres (recommended via docker-compose)

```powershell
# from project root
docker-compose up -d db
```

2) Backend (Windows PowerShell)

```powershell
cd backend
python -m venv .venv
# PowerShell activate
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Set env vars (example values)
$env:DATABASE_URL = "postgresql+psycopg2://dev_admin:password@localhost:5432/sop_dev_db?sslmode=require"
$env:OLLAMA_URL = "http://localhost:11434"
$env:OLLAMA_MODEL = "llama3:latest"
# Run backend
uvicorn main:app --reload --port 8000
```

On macOS / Linux, activate the venv with `source .venv/bin/activate` and export env vars with `export`.

3) Frontend

```powershell
cd frontend
npm install
npm run dev
# Open http://localhost:5173 (Vite default)
```

## Ollama (local LLM)
- Install and run Ollama following its official docs. Start the server with:

```powershell
ollama serve
```

- List available models:

```powershell
ollama list
```

- Use the exact model name shown by `ollama list` when setting `OLLAMA_MODEL` (for example `llama3:latest`).

If the backend shows LLM timeouts (HTTP read timed out), wait for the model to finish loading — large models can take a minute or more on first load.

## Supported upload types
- Text: `.txt`, `.md`
- PDF: `.pdf` (text-based PDFs work out of the box)
- Word: `.docx` (supported)
- Legacy Word: `.doc` — best-effort via `textract` or LibreOffice conversion (see notes below)
- Images: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.webp` — currently stored as metadata; OCR is optional (see notes)

If an uploaded PDF contains only scanned images (no embedded text), the backend may not extract text — see "Optional tools" below to enable OCR.

## Endpoints (examples)
- Health: GET http://localhost:8000/health
- Upload: POST http://localhost:8000/api/upload (form file field `file`)
- Chat (RAG): POST http://localhost:8000/api/chat with JSON {"query": "your question"}

Curl examples:

```powershell
# Upload
curl -X POST http://localhost:8000/api/upload -F "file=@C:\path\to\document.pdf"
# Chat
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"query":"hi"}'
```

## Optional tools & troubleshooting
These tools improve extraction for PDFs, images, and legacy Word files. They are optional — the backend will try fallbacks when available.

- pdfplumber: better PDF text extraction (pip)

```powershell
pip install pdfplumber
```

- Tesseract OCR + pytesseract: OCR for scanned/image PDFs and images. Install the Tesseract binary first, then the Python wrapper.

Windows install (example using winget):
```powershell
winget install --id=UB-Mannheim.Tesseract -e
pip install pytesseract
```

macOS / Linux: install tesseract via your package manager (brew/apt) then `pip install pytesseract`.

- LibreOffice (soffice): enables server-side conversion of legacy `.doc` files to `.docx` if `textract` is not available.

Windows example (winget):
```powershell
winget install --id=LibreOffice.LibreOffice -e
```

- textract: supports `.doc` extraction but may have packaging issues with newer pip versions and often requires system helpers (antiword/unrtf). If you want to use `textract` on Windows you may need to install compatible dependencies and/or downgrade pip to <24.1 before installing textract:

```powershell
python -m pip install "pip<24.1"
python -m pip install textract
```

Note: downgrading pip affects other installs; LibreOffice conversion is a safer server-side option for many setups.

## Behavior notes / debugging
- If an upload returns the error "No text content found in file", it usually means the extractor couldn't find text (scanned PDF or image). Try installing pdfplumber or Tesseract or convert the file to PDF/DOCX with selectable text.
- If the backend returns LLM timeouts, check the Ollama server logs and wait for the model to finish loading. Also ensure `OLLAMA_MODEL` matches the model name from `ollama list`.
- The backend logs helpful diagnostics for uploads (filename, size, content-type) and stack traces for failures — check the backend terminal for details.

## Development notes
- Embeddings are generated with `sentence-transformers` model `all-mpnet-base-v2` and stored using `pgvector` (dimension 768). If you change the embedding model, update the `Vector(768)` dimension accordingly in `backend/main.py`.
- DB schema is created automatically on startup from SQLAlchemy models, but for production schema changes use Alembic migrations (an `alembic` folder is included).

## Contributing / Next steps
- Add `pdf2image` + OCR pipeline to improve text extraction for mixed PDFs.
- Add admin endpoints to list and delete documents in the vector DB for easier debugging.

If you want, I can update `requirements.txt` to include optional packages (pdfplumber, pytesseract) and add a short `docs/` page with platform-specific install steps.

---
Happy hacking — if anything in this README doesn't work on your machine, paste the terminal output and I'll help troubleshoot.
