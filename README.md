# VaxAsk

**A classification-guided, communication-adapted retrieval-augmented generation (RAG) system for vaccine-hesitancy counseling.**

**Live at [vaxask.org](https://vaxask.org)** — production deployment.

VaxAsk answers vaccine questions by grounding every response in a curated scientific knowledge base **and** adapting *how* it responds to the person asking — their concern, their attitude, and their emotional state. Unlike a plain chatbot or a generic RAG pipeline, VaxAsk first reads the question, then tailors both the evidence it retrieves and the communication strategy it uses.

> ⚠️ VaxAsk is a research prototype and an informational tool. It does **not** provide medical advice and does not replace a clinician.

---

## How it works

Each incoming question flows through a single pipeline:

```
Question
   │
   ▼
1. Classification  ──►  concern category · attitude · dominant emotion ·
   (LLM classifier)     clinical red-flag · audience (self / child / other)
   │
   ├──────────────► 2. Evidence path (whole corpus, prioritized — not restricted)
   │                   • the category's designated "anchor" source is guaranteed
   │                     the top slots
   │                   • sources from the question's concern category are prioritized
   │                     next
   │                   • the rest of the corpus fills any remaining slots, so answers
   │                     never depend on a category being well-populated
   │
   └──────────────► 3. Communication path
                       • a policy (category × attitude × emotion) is injected into
                         the prompt: tone, approach, information dose, allowed and
                         forbidden moves — grounded in motivational-interviewing
                         and public-health communication guidance
                       • a safety gate is appended on clinical red flags
   │
   ▼
4. Answer  ──►  evidence-grounded, numbered citations, tailored to the person
```

**Anchors.** For each concern category, one or more sources can be flagged as the most authoritative ("anchor"). During retrieval, the anchor's most relevant passage is guaranteed a place in the answer's context, so the definitive evidence is never left to chance.

**Communication policy.** The classifier's output selects a policy row (e.g. *resistant + anger + distrust* → short, non-confrontational, autonomy-preserving, avoid "backfire"). This is the core of VaxAsk's adaptation.

---

## Repository layout

```
backend/
  main.py                 FastAPI app: /api/chat, /api/classify, /api/stats, admin endpoints
  rag.py                  the pipeline: retrieval + anchor + policy injection + generation
  classifier.py           question classifier (category / attitude / emotion / red-flag / audience)
  policy.py               builds the injected communication policy from a matrix
  ingest.py               PDF -> chunks -> embeddings -> ChromaDB
  metadata_extractor.py   auto-extract citation metadata from a PDF
  models.py               Pydantic models
  assets/
    ASI_Yanit_Politikasi_Matrisi.xlsx   the communication-policy matrix
  requirements.txt
frontend/
  index.html              patient-facing chat UI (Turkish / English)
  admin.html              knowledge-base management UI
.env.example
```

The **knowledge base itself (embeddings, source PDFs) is not distributed** — you build it from your own sources via the admin panel (copyright/licensing of source documents is your responsibility).

**Study data.** The [`data/`](data/) folder contains the 129 knowledge-base source references (`knowledge_base_references.xlsx`) and the 180 evaluation questions with their design metadata (`evaluation_questions.xlsx`).

---

## Quick start

```bash
# 1. Install
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
cp ../.env.example ../.env        # then edit ../.env with your API key + admin password

# 3. Run
uvicorn main:app --host 127.0.0.1 --port 8003
```

Open `http://127.0.0.1:8003/` for the chat UI and `http://127.0.0.1:8003/admin` for the admin panel (Basic Auth with the credentials from `.env`).

### Building the knowledge base

Open `/admin`, upload a PDF, review the auto-extracted metadata, and assign:

- a **concern category** (1–12) — the taxonomy the classifier also uses;
- optionally an **anchor** flag + rank, to make the source a priority for its category;
- a **citation string** (printed verbatim under answers).

The document is chunked (~1200 chars, 150 overlap), embedded with `intfloat/multilingual-e5-large`, and stored in ChromaDB (cosine similarity).

---

## Configuration

All settings are environment variables (see `.env.example`). Key ones:

| Variable | Purpose |
|---|---|
| `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` | OpenAI-compatible LLM endpoint |
| `RESPONDER_MODEL` | model that writes the answer |
| `CLASSIFIER_MODEL` | model that classifies the question |
| `CHROMA_DB_PATH` | where the vector store lives |
| `PDF_STORAGE_PATH` | where uploaded PDFs are kept |
| `ADMIN_USERNAME`, `ADMIN_PASSWORD` | admin-panel credentials |

The LLM interface is OpenAI-compatible, so any compatible provider/model can be used. Step-by-step reasoning ("thinking") is disabled in code so answers are returned in full.

---

## Disclaimer

VaxAsk is provided for informational and research purposes only. It is not a medical device, does not provide personalized medical advice, and is not a substitute for consultation with a qualified health professional. Answers are only as reliable as the knowledge base you build.

## License

Released under the MIT License — see [LICENSE](LICENSE).
