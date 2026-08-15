import fitz
import re
import requests
import hashlib
from models import ArticleMetadata

def extract_doi(text: str) -> str | None:
    pattern = r'10\.\d{4,9}/[-._;()/:A-Z0-9a-z]+'
    match = re.search(pattern, text)
    return match.group(0) if match else None

def lookup_doi_metadata(doi: str) -> dict | None:
    try:
        url = f"https://api.crossref.org/works/{doi}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()['message']
            return {
                'title': data.get('title', [''])[0],
                'authors': ', '.join(
                    f"{a.get('family', '')} {a.get('given', '')}"
                    for a in data.get('author', [])[:3]
                ) + (' et al.' if len(data.get('author', [])) > 3 else ''),
                'journal': data.get('container-title', [''])[0],
                'year': data.get('published', {}).get('date-parts', [[0]])[0][0],
                'doi': doi,
                'doi_url': f"https://doi.org/{doi}",
            }
    except Exception:
        return None

def lookup_pubmed(doi: str) -> str | None:
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": f"{doi}[doi]", "retmode": "json"}
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            ids = resp.json().get('esearchresult', {}).get('idlist', [])
            return ids[0] if ids else None
    except Exception:
        return None

def extract_metadata_from_pdf(pdf_path: str, filename: str) -> ArticleMetadata:
    doc = fitz.open(pdf_path)

    first_pages_text = ""
    for i in range(min(2, len(doc))):
        first_pages_text += doc[i].get_text()

    pdf_meta = doc.metadata

    doi = extract_doi(first_pages_text)

    source_id = hashlib.md5(filename.encode()).hexdigest()[:8]

    meta = {
        'source_id': source_id,
        'title': pdf_meta.get('title', '') or '',
        'authors': pdf_meta.get('author', '') or '',
        'journal': '',
        'year': 0,
        'doi': doi,
        'doi_url': f"https://doi.org/{doi}" if doi else None,
        'filename': filename,
    }

    if doi:
        crossref = lookup_doi_metadata(doi)
        if crossref:
            meta.update({k: v for k, v in crossref.items() if v})

        pmid = lookup_pubmed(doi)
        if pmid:
            meta['pubmed_id'] = pmid
            meta['pubmed_url'] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    if not meta.get('year'):
        year_match = re.search(r'\b(19|20)\d{2}\b', first_pages_text)
        if year_match:
            meta['year'] = int(year_match.group(0))

    doc.close()
    return ArticleMetadata(**meta)
