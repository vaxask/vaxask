from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import shutil
import secrets
from dotenv import load_dotenv
from models import (
    ChatRequest, ChatResponse, ArticleMetadata,
    IngestRequest, MetadataExtractResponse
)
from rag import chat
from ingest import get_chroma_client, get_collection
from metadata_extractor import extract_metadata_from_pdf

load_dotenv()

app = FastAPI(title="VaxAsk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="../frontend"), name="static")

security = HTTPBasic()

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(
        credentials.username, os.getenv('ADMIN_USERNAME', 'admin')
    )
    correct_password = secrets.compare_digest(
        credentials.password, os.getenv('ADMIN_PASSWORD', 'changeme')
    )
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/")
def serve_frontend():
    return FileResponse("../frontend/index.html")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    result = chat(
        message=request.message,
        conversation_history=request.conversation_history,
        lang=request.lang or "tr",
    )

    return ChatResponse(
        answer=result['answer'],
        sources=result['sources'],
        source_indices=list(range(1, len(result['sources']) + 1))
    )

@app.post("/api/classify")
def classify_endpoint(request: ChatRequest):
    from classifier import classify
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    return classify(request.message, request.lang or "tr")

@app.get("/api/stats")
def get_stats():
    client = get_chroma_client()
    collection = get_collection(client)
    all_items = collection.get()

    source_ids = set(
        m['source_id'] for m in all_items['metadatas']
    ) if all_items['metadatas'] else set()

    return {
        "total_articles": len(source_ids),
        "total_chunks": len(all_items['ids']),
    }

@app.get("/admin")
def serve_admin():
    return FileResponse("../frontend/admin.html")

@app.post("/api/admin/extract-metadata")
async def extract_metadata_endpoint(
    file: UploadFile = File(...),
    _: str = Depends(verify_admin)
):
    pdf_dir = os.getenv('PDF_STORAGE_PATH', './pdfs')
    os.makedirs(pdf_dir, exist_ok=True)

    temp_path = os.path.join(pdf_dir, f"temp_{file.filename}")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        metadata = extract_metadata_from_pdf(temp_path, file.filename)
        confidence = 0.9 if metadata.doi else 0.5
        return MetadataExtractResponse(
            suggested_metadata=metadata,
            confidence=confidence
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/admin/ingest")
async def ingest_endpoint(
    file: UploadFile = File(...),
    title: str = Form(""),
    authors: str = Form(""),
    journal: str = Form(""),
    year: int = Form(0),
    doi: str = Form(""),
    pubmed_id: str = Form(""),
    source_id: str = Form(""),
    kategori_id: int = Form(0),
    is_anchor: bool = Form(False),
    anchor_rank: int = Form(0),
    citation_string: str = Form(""),
    _: str = Depends(verify_admin)
):
    from ingest import ingest_pdf

    pdf_dir = os.getenv('PDF_STORAGE_PATH', './pdfs')
    os.makedirs(pdf_dir, exist_ok=True)

    pdf_path = os.path.join(pdf_dir, file.filename)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    metadata = ArticleMetadata(
        source_id=source_id,
        title=title,
        authors=authors,
        journal=journal,
        year=year,
        doi=doi if doi else None,
        doi_url=f"https://doi.org/{doi}" if doi else None,
        pubmed_id=pubmed_id if pubmed_id else None,
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/" if pubmed_id else None,
        filename=file.filename,
        pdf_filename=file.filename,
        kategori_id=kategori_id,
        is_anchor=is_anchor,
        anchor_rank=anchor_rank,
        citation_string=citation_string or None,
        kunye_dogrulandi=bool(citation_string),
    )

    result = ingest_pdf(pdf_path, metadata)
    return {"status": "success", "kunye_dogrulandi": metadata.kunye_dogrulandi, **result}

@app.get("/api/admin/articles")
def list_articles(_: str = Depends(verify_admin)):
    client = get_chroma_client()
    collection = get_collection(client)
    all_items = collection.get()

    if not all_items['metadatas']:
        return {"articles": []}

    seen = {}
    for meta in all_items['metadatas']:
        sid = meta['source_id']
        if sid not in seen:
            seen[sid] = meta

    return {"articles": list(seen.values())}

@app.patch("/api/admin/articles/{source_id}")
def update_article_metadata(
    source_id: str,
    updated: ArticleMetadata,
    _: str = Depends(verify_admin)
):
    client = get_chroma_client()
    collection = get_collection(client)

    existing = collection.get(
        where={"source_id": source_id},
        include=["documents", "embeddings", "metadatas"]
    )

    if not existing['ids']:
        raise HTTPException(status_code=404, detail="Source not found")

    chunk_count = len(existing['ids'])
    prev = existing['metadatas'][0] if existing['metadatas'] else {}

    new_meta_base = {
        "source_id":  updated.source_id,
        "title":      updated.title,
        "authors":    updated.authors,
        "journal":    updated.journal,
        "year":       updated.year,
        "doi":        updated.doi or "",
        "doi_url":    updated.doi_url or (f"https://doi.org/{updated.doi}" if updated.doi else ""),
        "pubmed_id":  updated.pubmed_id or "",
        "pubmed_url": updated.pubmed_url or (
                        f"https://pubmed.ncbi.nlm.nih.gov/{updated.pubmed_id}/"
                        if updated.pubmed_id else ""
                      ),
        "filename":   updated.filename,
        "pdf_filename":     updated.pdf_filename or prev.get("pdf_filename", updated.filename),
        "kategori_id":      updated.kategori_id if updated.kategori_id is not None else prev.get("kategori_id", 0),
        "is_anchor":        bool(updated.is_anchor),
        "anchor_rank":      updated.anchor_rank or 0,
        "citation_string":  updated.citation_string if updated.citation_string is not None else prev.get("citation_string", ""),
        "kunye_dogrulandi": bool(updated.kunye_dogrulandi),
    }

    new_metadatas = [
        {**new_meta_base, "chunk_index": m.get("chunk_index", i)}
        for i, m in enumerate(existing['metadatas'])
    ]
    new_ids = [f"{updated.source_id}_chunk_{i}" for i in range(chunk_count)]

    collection.delete(ids=existing['ids'])
    collection.add(
        ids=new_ids,
        embeddings=existing['embeddings'],
        documents=existing['documents'],
        metadatas=new_metadatas,
    )

    return {"status": "updated", "source_id": updated.source_id, "chunks_updated": chunk_count}

@app.delete("/api/admin/articles/{source_id}")
def delete_article(source_id: str, _: str = Depends(verify_admin)):
    client = get_chroma_client()
    collection = get_collection(client)

    existing = collection.get(where={"source_id": source_id})
    if not existing['ids']:
        raise HTTPException(status_code=404, detail="Source not found")

    collection.delete(ids=existing['ids'])
    return {"status": "deleted", "source_id": source_id, "chunks_deleted": len(existing['ids'])}
