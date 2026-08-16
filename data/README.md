# Data

Supplementary data for the VaxAsk study.

- **`knowledge_base_references.xlsx`** — the 129 sources that make up the knowledge base
  (concern category, anchor rank, title, journal, year, PubMed/source link, full verified
  citation, chunk count). The source PDFs themselves are **not distributed**
  (copyright/licensing is the user's responsibility); this file lists exactly which sources
  the corpus was built from. The `capa` (anchor) column is populated only for the sources
  designated as an authoritative anchor for their category; a blank means the source is a
  regular (non-anchor) member of the corpus. A category can have several anchors — every one
  of them is guaranteed a place at the top of the retrieved context. A source may also anchor
  more than one category (shown as, e.g., `rank-1 (kat-1 + kat-3)`).

- **`evaluation_questions.xlsx`** — the 180 evaluation questions (12 concern categories ×
  15), each given in the original Turkish (`Soru (TR)`) and in a careful English
  translation (`Question (EN)`) that preserves the patient/parent voice.
