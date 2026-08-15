import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from models import ArticleMetadata
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

MAX_CHUNKS_PER_DOC = int(os.getenv('MAX_CHUNKS_PER_DOC', '0'))

_embedder = None
_chroma_client = None

def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=os.getenv('CHROMA_DB_PATH', './chroma_db')
        )
    return _chroma_client

def get_collection(client=None):
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name="vaccine_papers",
        metadata={"hnsw:space": "cosine"}
    )

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 50:
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

def ingest_pdf(pdf_path: str, metadata: ArticleMetadata) -> dict:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    chunks = chunk_text(full_text)

    if not chunks:
        return {"chunks_added": 0, "source_id": metadata.source_id}

    if MAX_CHUNKS_PER_DOC and len(chunks) > MAX_CHUNKS_PER_DOC:
        print(f"    [SINIR] {metadata.filename}: {len(chunks)} -> "
              f"{MAX_CHUNKS_PER_DOC} chunk (kirpildi)")
        chunks = chunks[:MAX_CHUNKS_PER_DOC]

    embedder = get_embedder()

    passages = [f"passage: {chunk}" for chunk in chunks]
    embeddings = embedder.encode(passages, normalize_embeddings=True).tolist()

    client = get_chroma_client()
    collection = get_collection(client)

    try:
        existing = collection.get(where={"source_id": metadata.source_id})
        if existing['ids']:
            collection.delete(ids=existing['ids'])
    except Exception:
        pass

    chunk_ids = [f"{metadata.source_id}_chunk_{i}" for i in range(len(chunks))]
    chunk_metadatas = [{
        "source_id": metadata.source_id,
        "title": metadata.title,
        "authors": metadata.authors,
        "journal": metadata.journal,
        "year": metadata.year,
        "doi": metadata.doi or "",
        "doi_url": metadata.doi_url or "",
        "pubmed_id": metadata.pubmed_id or "",
        "pubmed_url": metadata.pubmed_url or "",
        "filename": metadata.filename,
        "chunk_index": i,
        "pdf_filename": metadata.pdf_filename or metadata.filename,
        "kategori_id": metadata.kategori_id,
        "is_anchor": metadata.is_anchor,
        "anchor_rank": metadata.anchor_rank,
        "citation_string": metadata.citation_string or "",
        "kunye_dogrulandi": metadata.kunye_dogrulandi,
    } for i in range(len(chunks))]

    collection.add(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=chunk_metadatas
    )

    return {"chunks_added": len(chunks), "source_id": metadata.source_id}
