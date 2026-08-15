from pydantic import BaseModel
from typing import Optional, List

class ArticleMetadata(BaseModel):
    source_id: str
    title: str
    authors: str
    journal: str
    year: int
    doi: Optional[str] = None
    doi_url: Optional[str] = None
    pubmed_id: Optional[str] = None
    pubmed_url: Optional[str] = None
    abstract: Optional[str] = None
    filename: str

    pdf_filename: Optional[str] = None
    kategori_id: int = 0
    is_anchor: bool = False
    anchor_rank: int = 0
    citation_string: Optional[str] = None
    kunye_dogrulandi: bool = False

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = []
    lang: Optional[str] = "tr"

class ChatResponse(BaseModel):
    answer: str
    sources: List[ArticleMetadata]
    source_indices: List[int]

class MetadataExtractResponse(BaseModel):
    suggested_metadata: ArticleMetadata
    confidence: float

class IngestRequest(BaseModel):
    metadata: ArticleMetadata
