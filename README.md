# UniAssist AI — University Helpdesk AI Assistant

UniAssist AI is an enterprise-grade University Helpdesk Assistant designed with strict Retrieval-Augmented Generation (RAG) grounding. It uses **OpenRouter API** via an **OpenAI-compatible client** to provide accurate, reliable answers exclusively grounded in official university documents.

---

## 🚀 Live Demo

**[Try UniAssist AI](https://lpu-uni-assist-ai.vercel.app/)**

Ask questions about admissions, scholarships, course registration, campus IT, placements, and university services.

## 🛠️ Tech Stack

- Frontend: React, Vite
- Backend: FastAPI, Python
- AI: OpenRouter LLM
- Retrieval: RAG, embeddings, ChromaDB
- Deployment: Vercel and Render

## 🔗 Links

- **Live Demo:** https://lpu-uni-assist-ai.vercel.app/
- **Backend API:** https://lpu-uniassist-ai.onrender.com/
- **API Documentation:** https://lpu-uniassist-ai.onrender.com/docs


## 🌟 Why OpenRouter?

1. **OpenAI-Compatible Interface**: OpenRouter exposes standard OpenAI-compatible endpoints (`https://openrouter.ai/api/v1`), allowing us to use the battle-tested `openai` Python SDK without vendor lock-in.
2. **Model Flexibility**: Easily swap between leading open and commercial models (such as `meta-llama/llama-3.3-70b-instruct`, `mistralai/mistral-large`, or `anthropic/claude-3-haiku`) simply by updating the `OPENROUTER_MODEL` environment variable.
3. **Decoupled Architecture**: All LLM communications are strictly isolated inside `backend/app/services/llm_service.py`. The rest of the RAG pipeline interacts only with `generate_answer()`, ensuring that changing the LLM provider or model does not touch the RAG indexing, retrieval, or frontend code.
4. **No Gemini Dependency**: The project strictly does not use or install the Gemini API (`google-generativeai`). Embeddings and vector similarity are computed locally using ChromaDB's ONNX model.

---

## 🏛️ System Architecture

The end-to-end pipeline enforces strict grounding at every step:

```
User Question
      ↓
Conversation Memory (Session context tracking)
      ↓
Query Understanding (Coreference resolution & out-of-scope detection)
      ↓
Embedding (Local embeddings via ONNX all-MiniLM-L6-v2)
      ↓
ChromaDB (Vector similarity search)
      ↓
Top-K Relevant Documents
      ↓
Relevance Threshold Filtering (Score >= 0.35)
      ↓
Retrieved Context (Ranked & prioritized)
      ↓
OpenRouter LLM (via backend/app/services/llm_service.py)
      ↓
Grounded Answer
      ↓
Sources & Citations
      ↓
React Frontend
```

### Strict Grounding Rules
- **No Hallucinations**: If relevant context is not found in the knowledge base (or score < threshold), the bot responds:
  > *"I couldn't find that information in the university helpdesk knowledge base."*
- **Out-of-Scope Filtering**: If a user asks questions unrelated to university services, the bot responds:
  > *"I'm the University Helpdesk Assistant, so I can only help with questions related to university services and support."*
- **No Direct Answering**: The user's question is **never** sent raw to the LLM to answer using general world knowledge; facts must originate from retrieved context.

---

## ⚙️ API Configuration & Environment Variables

### Backend Configuration (`backend/.env`)

Copy `backend/.env.example` to `backend/.env`:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct
OPENROUTER_SITE_URL=http://localhost:5173
OPENROUTER_APP_NAME=University Helpdesk AI
TOP_K=4
RELEVANCE_THRESHOLD=0.35
```

| Variable | Description | Default |
| :--- | :--- | :--- |
| `OPENROUTER_API_KEY` | Your OpenRouter API key. **Required for live LLM calls.** | *None* |
| `OPENROUTER_MODEL` | Model ID on OpenRouter (e.g. `meta-llama/llama-3.3-70b-instruct`). | `meta-llama/llama-3.3-70b-instruct` |
| `OPENROUTER_SITE_URL` | Sent as `HTTP-Referer` header for OpenRouter analytics. | `http://localhost:5173` |
| `OPENROUTER_APP_NAME` | Sent as `X-Title` header for OpenRouter rankings. | `University Helpdesk AI` |
| `TOP_K` | Number of candidate document chunks retrieved from ChromaDB. | `4` |
| `RELEVANCE_THRESHOLD` | Minimum similarity score required to include context. | `0.35` |

### Frontend Configuration (`frontend/.env`)

```env
VITE_API_URL=http://localhost:8000
```

> [!CAUTION]
> **Security Notice**:
> - Never commit `OPENROUTER_API_KEY` to GitHub. Keep `.env` in `.gitignore`.
> - **Never** expose `OPENROUTER_API_KEY` to the React frontend. The key must remain exclusively in the FastAPI backend environment.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Node.js 18+ and npm

### 2. Backend Setup

```powershell
# Navigate to backend directory
cd backend

# Create virtual environment (if not already created)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies (contains openai and chromadb; NO google-generativeai)
pip install -r requirements.txt

# Create .env from template
cp .env.example .env
# Edit .env and insert your OPENROUTER_API_KEY

# Start the FastAPI server
uvicorn app.main:app --reload --port 8000
```

API documentation will be available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Frontend Setup

```powershell
# In a new terminal, navigate to frontend
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be running at: [http://localhost:5173](http://localhost:5173)

---

## 🧪 Testing & Verification

The test suite covers all critical edge cases and failure modes:
1. **Normal FAQ Question**: Verifies knowledge retrieval from ChromaDB, source citation extraction, and grounded answer generation.
2. **Follow-Up Question Using Memory**: Verifies that pronouns and references (e.g., "Where is it located?") use session history to resolve the antecedent subject.
3. **Unknown Question**: Verifies that queries below the `RELEVANCE_THRESHOLD` return `"I couldn't find that information in the university helpdesk knowledge base."` without hallucinating.
4. **Out-of-Scope Question**: Verifies that non-university queries (e.g., recipes, sports, random trivia) return `"I'm the University Helpdesk Assistant, so I can only help with questions related to university services and support."`
5. **OpenRouter API Failure**: Verifies graceful error handling when OpenRouter API encounters network, rate limit, or authentication issues.
6. **Missing API Key**: Verifies clean configuration error reporting when `OPENROUTER_API_KEY` is unset.
7. **Grounding Verification**: Verifies that general world knowledge cannot bypass the RAG retrieval mechanism.

### Running Tests

```powershell
cd backend
.\venv\Scripts\pytest.exe -v
```

---

## 📁 Project Structure

```
UniAssist AI/
├── backend/
│   ├── .env.example              # Example environment variables
│   ├── .env                      # Local configuration (never committed)
│   ├── requirements.txt          # Python dependencies (OpenAI SDK, ChromaDB, FastAPI)
│   ├── pytest.ini                # Pytest configuration
│   ├── app/
│   │   ├── main.py               # FastAPI application entry point
│   │   ├── core/
│   │   │   └── config.py         # Pydantic settings management
│   │   ├── services/
│   │   │   ├── llm_service.py    # Isolated OpenRouter LLM service (OpenAI SDK)
│   │   │   ├── embedding_service.py # Local ONNX embeddings (no Gemini)
│   │   │   ├── vector_store.py   # ChromaDB management & document indexing
│   │   │   ├── memory_service.py # Multi-turn conversation session history
│   │   │   ├── query_service.py  # Query understanding & out-of-scope filtering
│   │   │   └── rag_service.py    # End-to-end RAG orchestrator
│   │   └── api/
│   │       └── routes.py         # /api/chat, /api/health, /api/seed endpoints
│   ├── data/
│   │   └── knowledge_base/       # Official university policy documents (Markdown)
│   └── tests/
│       └── test_rag_pipeline.py  # Comprehensive test suite
├── frontend/
│   ├── .env.example              # Frontend environment template
│   ├── .env                      # Frontend environment (VITE_API_URL)
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── App.jsx               # Interactive chat interface & citation viewer
│       ├── main.jsx              # React mounting
│       └── index.css             # Responsive university UI styling
└── README.md                     # Comprehensive documentation
```
