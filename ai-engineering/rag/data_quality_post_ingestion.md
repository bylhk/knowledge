# Data Quality: Post-Ingestion, Pre-Chunking

## Overview

After extracting content from documents (PDFs, images, scans) but before chunking and embedding, you need to validate and clean the raw extracted text. This step catches OCR errors, encoding issues, layout artefacts, and incomplete extractions that would otherwise pollute your vector store.

## Where This Fits

```
[Document] → [Ingestion/OCR] → ⭐ DATA QUALITY ⭐ → [Chunking] → [Embedding] → [Vector DB]
```

## Common Issues After Ingestion

| Issue | Cause | Example |
|-------|-------|---------|
| Mojibake / garbled text | Encoding mismatch | `â€™` instead of `'` |
| OCR garbage | Low-quality scan | `|l1I` confusion, `rn` → `m` |
| Merged words | Missing spaces in PDF | `thepriceincreased` |
| Split words | Hyphenation / line breaks | `re-\ncommendation` |
| Layout noise | Headers, footers, page numbers | `Page 3 of 47` on every page |
| Table destruction | Columns read as rows | Jumbled numbers and labels |
| Empty/whitespace pages | Blank pages, separator sheets | `\n\n\n\n` |
| Duplicate pages | Re-scanned / appended twice | Same content repeated |
| Language detection failures | Mixed-language docs | Wrong OCR language model used |
| Truncated content | Parser timeout or size limit | Text cuts off mid-sentence |

---

## Strategy 1: Structural Validation

### Check Extraction Completeness

```python
from dataclasses import dataclass
from pathlib import Path
import pymupdf

@dataclass
class ExtractionReport:
    source: str
    total_pages: int
    extracted_pages: int
    total_chars: int
    avg_chars_per_page: float
    empty_pages: list[int]
    is_complete: bool
    issues: list[str]

def validate_extraction(source_path: str, extracted_text: str, expected_pages: int = None) -> ExtractionReport:
    issues = []

    # Get expected page count from source
    if source_path.endswith(".pdf"):
        doc = pymupdf.open(source_path)
        expected_pages = expected_pages or len(doc)
        doc.close()

    # Split by page markers (depends on your ingestion output)
    pages = extracted_text.split("\n--- Page ")
    pages = [p for p in pages if p.strip()]
    extracted_pages = len(pages)

    # Check page count
    if expected_pages and extracted_pages < expected_pages * 0.9:
        issues.append(f"missing_pages: got {extracted_pages}/{expected_pages}")

    # Check for empty pages
    empty_pages = []
    for i, page in enumerate(pages, 1):
        if len(page.strip()) < 20:
            empty_pages.append(i)

    if len(empty_pages) > expected_pages * 0.3:
        issues.append(f"too_many_empty_pages: {len(empty_pages)}")

    # Total content length
    total_chars = len(extracted_text)
    avg_chars = total_chars / max(extracted_pages, 1)

    if avg_chars < 100:
        issues.append("very_low_content_density")
    if total_chars < 50:
        issues.append("near_empty_extraction")

    # Truncation check
    if extracted_text.rstrip()[-1:] not in ".!?\"')]}":
        if len(extracted_text) > 1000:
            issues.append("possible_truncation")

    return ExtractionReport(
        source=source_path,
        total_pages=expected_pages or extracted_pages,
        extracted_pages=extracted_pages,
        total_chars=total_chars,
        avg_chars_per_page=avg_chars,
        empty_pages=empty_pages,
        is_complete=len(issues) == 0,
        issues=issues,
    )
```

---

## Strategy 2: Text Quality Scoring

### OCR Confidence & Readability

```python
import re
from collections import Counter

def text_quality_score(text: str) -> dict:
    """Score extracted text quality on multiple dimensions."""
    scores = {}

    # 1. Dictionary word ratio (proxy for OCR accuracy)
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text)
    # Simple heuristic: real words tend to have vowels
    has_vowel = [w for w in words if re.search(r"[aeiou]", w.lower())]
    scores["vowel_ratio"] = len(has_vowel) / max(len(words), 1)

    # 2. Character distribution (garbled text has unusual char frequencies)
    chars = [c for c in text if c.isalpha()]
    if chars:
        freq = Counter(chars)
        most_common_pct = freq.most_common(1)[0][1] / len(chars)
        scores["char_distribution_ok"] = most_common_pct < 0.2  # No single char > 20%

    # 3. Whitespace ratio
    whitespace_count = sum(1 for c in text if c in " \t\n")
    scores["whitespace_ratio"] = whitespace_count / max(len(text), 1)

    # 4. Special character density (high = likely garbled)
    special = re.findall(r"[^\w\s.,;:!?'\"\-()£$%/\\@#&]", text)
    scores["special_char_ratio"] = len(special) / max(len(text), 1)

    # 5. Average word length (OCR errors create very short/long "words")
    word_lengths = [len(w) for w in words]
    if word_lengths:
        avg_len = sum(word_lengths) / len(word_lengths)
        scores["avg_word_length"] = avg_len
        scores["word_length_ok"] = 3.0 < avg_len < 12.0

    # 6. Line coherence (complete sentences vs fragments)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    short_lines = sum(1 for l in lines if len(l) < 20)
    scores["fragment_ratio"] = short_lines / max(len(lines), 1)

    # Overall score (0-1)
    checks = [
        scores.get("vowel_ratio", 0) > 0.7,
        scores.get("char_distribution_ok", False),
        scores.get("whitespace_ratio", 0) < 0.5,
        scores.get("special_char_ratio", 0) < 0.05,
        scores.get("word_length_ok", False),
        scores.get("fragment_ratio", 0) < 0.5,
    ]
    scores["overall"] = sum(checks) / len(checks)

    return scores
```

### Per-Page Quality Check

```python
def validate_pages(pages: list[str], min_quality: float = 0.5) -> dict:
    """Score each page and flag low-quality ones."""
    results = {"good": [], "bad": [], "scores": []}

    for i, page in enumerate(pages):
        score = text_quality_score(page)
        results["scores"].append(score["overall"])

        if score["overall"] >= min_quality:
            results["good"].append(i)
        else:
            results["bad"].append({"page": i, "score": score["overall"], "details": score})

    avg = sum(results["scores"]) / max(len(results["scores"]), 1)
    print(f"Average quality: {avg:.2f} | Good: {len(results['good'])} | Bad: {len(results['bad'])}")
    return results
```

---

## Strategy 3: Content Cleaning (Pre-Chunking)

### Fix Common Extraction Artefacts

```python
import re
import unicodedata

def clean_extracted_text(text: str) -> str:
    """Clean raw extracted text before chunking."""

    # 1. Unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # 2. Fix encoding artefacts
    encoding_fixes = {
        "â€™": "'", "â€˜": "'",
        "â€œ": '"', "â€\x9d": '"', "â€": '"',
        "â€"": "—", "â€"": "–",
        "Â£": "£", "Â©": "©", "Â®": "®",
        "\u00a0": " ",  # non-breaking space
        "\ufeff": "",   # BOM
    }
    for old, new in encoding_fixes.items():
        text = text.replace(old, new)

    # 3. Remove null bytes and control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 4. Fix hyphenated line breaks (re-\njoin → rejoin)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 5. Fix broken sentences across lines (but preserve paragraph breaks)
    text = re.sub(r"(?<=[a-z,;])\n(?=[a-z])", " ", text)

    # 6. Collapse excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 7. Remove page numbers and repeated headers/footers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*Page \d+(\s*of\s*\d+)?\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)

    return text.strip()
```

### Remove Boilerplate / Repeated Headers

```python
def remove_repeated_content(pages: list[str], threshold: int = 3) -> list[str]:
    """Remove lines that appear on most pages (headers, footers, watermarks)."""
    from collections import Counter

    # Count line frequency across pages
    line_counts = Counter()
    for page in pages:
        unique_lines = set(line.strip() for line in page.split("\n") if line.strip())
        line_counts.update(unique_lines)

    # Lines appearing on > threshold pages are likely boilerplate
    boilerplate = {line for line, count in line_counts.items() if count >= threshold and len(line) < 200}

    if boilerplate:
        print(f"Detected {len(boilerplate)} boilerplate lines:")
        for line in list(boilerplate)[:5]:
            print(f"  '{line[:80]}'")

    # Remove boilerplate from all pages
    cleaned_pages = []
    for page in pages:
        lines = page.split("\n")
        cleaned = [l for l in lines if l.strip() not in boilerplate]
        cleaned_pages.append("\n".join(cleaned))

    return cleaned_pages
```

---

## Strategy 4: Language & Encoding Detection

```python
from langdetect import detect, detect_langs, LangDetectException

def check_language(text: str, expected_lang: str = "en") -> dict:
    """Verify extracted text is in the expected language."""
    try:
        detected = detect(text[:5000])  # Sample first 5000 chars
        probabilities = detect_langs(text[:5000])

        return {
            "detected_language": detected,
            "expected_language": expected_lang,
            "match": detected == expected_lang,
            "probabilities": {str(p.lang): p.prob for p in probabilities[:3]},
        }
    except LangDetectException:
        return {
            "detected_language": None,
            "expected_language": expected_lang,
            "match": False,
            "probabilities": {},
            "error": "detection_failed",
        }

def detect_encoding_issues(text: str) -> list[str]:
    """Detect common encoding problems in extracted text."""
    issues = []

    # Check for mojibake patterns
    mojibake_patterns = [
        (r"Ã[€-¿]", "UTF-8 interpreted as Latin-1"),
        (r"â€[™˜œ\x9d""]", "Smart quotes garbled"),
        (r"Â[£©®]", "Currency/symbol encoding issue"),
        (r"[\ufffd]", "Replacement character (failed decode)"),
    ]

    for pattern, description in mojibake_patterns:
        matches = re.findall(pattern, text)
        if matches:
            issues.append(f"{description}: found {len(matches)} instances")

    return issues
```

---

## Strategy 5: Table Integrity Validation

```python
def validate_table_extraction(table_text: str) -> dict:
    """Check if an extracted table has consistent structure."""
    lines = [l for l in table_text.strip().split("\n") if l.strip()]

    if not lines:
        return {"valid": False, "issue": "empty_table"}

    # Check for consistent column count (markdown table)
    if "|" in lines[0]:
        col_counts = [line.count("|") for line in lines]
        consistent = len(set(col_counts)) <= 2  # Allow header separator variation
        return {
            "valid": consistent,
            "rows": len(lines),
            "columns": col_counts[0] - 1 if col_counts else 0,
            "issue": None if consistent else "inconsistent_columns",
        }

    # Check for tab-separated or space-aligned
    separators = ["\t", "  "]
    for sep in separators:
        if sep in lines[0]:
            col_counts = [line.count(sep) + 1 for line in lines]
            consistent = max(col_counts) - min(col_counts) <= 1
            return {
                "valid": consistent,
                "rows": len(lines),
                "separator": repr(sep),
                "issue": None if consistent else "inconsistent_columns",
            }

    return {"valid": False, "issue": "unrecognised_table_format"}
```

---

## Strategy 6: Duplicate Document Detection

```python
import hashlib

def detect_duplicate_documents(documents: list[dict]) -> dict:
    """Detect fully or partially duplicate source documents before chunking."""
    seen_hashes = {}
    duplicates = []

    for doc in documents:
        # Full-content hash
        content_hash = hashlib.sha256(doc["text"].strip().encode()).hexdigest()

        if content_hash in seen_hashes:
            duplicates.append({
                "source": doc["source"],
                "duplicate_of": seen_hashes[content_hash],
                "type": "exact",
            })
        else:
            seen_hashes[content_hash] = doc["source"]

    # Partial overlap (first 500 chars match — likely same doc, different version)
    prefix_hashes = {}
    for doc in documents:
        prefix = hashlib.md5(doc["text"][:500].strip().encode()).hexdigest()
        if prefix in prefix_hashes and doc["source"] != prefix_hashes[prefix]:
            duplicates.append({
                "source": doc["source"],
                "duplicate_of": prefix_hashes[prefix],
                "type": "partial_overlap",
            })
        else:
            prefix_hashes[prefix] = doc["source"]

    print(f"Found {len(duplicates)} duplicate documents")
    return {"duplicates": duplicates, "unique_count": len(seen_hashes)}
```

---

## Full Pre-Chunking Quality Pipeline

```python
def pre_chunking_quality_pipeline(documents: list[dict]) -> list[dict]:
    """
    Run quality checks on ingested documents before chunking.

    Each document: {"source": str, "text": str, "pages": list[str], "metadata": dict}
    """
    print(f"Quality pipeline: {len(documents)} documents")
    clean_documents = []

    # Step 1: Remove duplicate documents
    dedup_result = detect_duplicate_documents(documents)
    duplicate_sources = {d["source"] for d in dedup_result["duplicates"]}
    documents = [d for d in documents if d["source"] not in duplicate_sources]

    for doc in documents:
        issues = []

        # Step 2: Validate extraction completeness
        report = validate_extraction(doc["source"], doc["text"])
        if not report.is_complete:
            issues.extend(report.issues)

        # Step 3: Check language
        lang_check = check_language(doc["text"])
        if not lang_check["match"]:
            issues.append(f"unexpected_language: {lang_check['detected_language']}")

        # Step 4: Detect encoding issues
        encoding_issues = detect_encoding_issues(doc["text"])
        issues.extend(encoding_issues)

        # Step 5: Clean text
        doc["text"] = clean_extracted_text(doc["text"])

        # Step 6: Remove boilerplate (if page-level data available)
        if doc.get("pages"):
            doc["pages"] = remove_repeated_content(doc["pages"])
            doc["text"] = "\n\n".join(doc["pages"])

        # Step 7: Quality score
        quality = text_quality_score(doc["text"])

        if quality["overall"] >= 0.5:
            doc["metadata"]["quality_score"] = quality["overall"]
            doc["metadata"]["issues"] = issues
            clean_documents.append(doc)
        else:
            print(f"  REJECTED: {doc['source']} (quality={quality['overall']:.2f}, issues={issues})")

    print(f"Result: {len(clean_documents)}/{len(documents)} documents passed quality checks")
    return clean_documents
```

---

## Decision: When to Re-Ingest vs Clean

| Quality Score | Action |
|---------------|--------|
| > 0.8 | Pass through — clean and chunk |
| 0.5–0.8 | Clean aggressively, flag for review |
| 0.3–0.5 | Re-ingest with different strategy (e.g. Vision LLM instead of OCR) |
| < 0.3 | Reject — source document is too poor quality |

## Best Practices

- Validate immediately after ingestion, before any chunking
- Keep the raw extracted text alongside cleaned versions for debugging
- Use page-level granularity — don't reject entire documents for one bad page
- Log quality metrics per source document for trend monitoring
- Set different thresholds per document type (scans tolerate lower quality)
- Re-ingest with a better strategy rather than over-cleaning garbled text
- Encoding detection should run first — fixes propagate to all other checks
- Track ingestion quality over time to catch upstream changes (new scanner, format shift)
