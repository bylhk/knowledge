# Document Ingestion: PDFs, Images & Scanned Documents

## Overview

Ingesting unstructured documents (PDFs, images, scans) into LLM pipelines requires extracting text, tables, and visual elements into a format suitable for chunking, embedding, and retrieval. The approach depends on whether the document is digitally-born (selectable text) or scanned (image-only).

## Document Types & Challenges

| Type | Example | Challenge |
|------|---------|-----------|
| Digital PDF | Reports, contracts | Tables, multi-column layouts, headers/footers |
| Scanned PDF | Old forms, signed docs | No selectable text — needs OCR |
| Images | Screenshots, diagrams | Layout detection + OCR |
| Mixed PDF | Scan with digital overlay | Partial text layer, inconsistent quality |

---

## Approach 1: Traditional OCR Pipeline

### Tesseract + pdf2image

```python
from pdf2image import convert_from_path
import pytesseract

# Convert PDF pages to images
images = convert_from_path("scanned_document.pdf", dpi=300)

# OCR each page
full_text = ""
for i, image in enumerate(images):
    text = pytesseract.image_to_string(image, lang="eng")
    full_text += f"\n--- Page {i + 1} ---\n{text}"

print(full_text)
```

### Pros & Cons

- ✅ Free, runs locally, no API calls
- ✅ Good for simple single-column text
- ❌ Struggles with tables, multi-column layouts
- ❌ No semantic understanding of structure

---

## Approach 2: Layout-Aware Parsing (Unstructured)

### unstructured library

The `unstructured` library detects document layout (titles, paragraphs, tables, images) and extracts structured elements.

```bash
pip install "unstructured[all-docs]"
# Requires: poppler, tesseract, libmagic
```

```python
from unstructured.partition.pdf import partition_pdf

# Partition with layout detection
elements = partition_pdf(
    filename="report.pdf",
    strategy="hi_res",          # Uses layout model (detectron2/yolox)
    infer_table_structure=True, # Extract tables as HTML
    extract_images_in_pdf=True, # Pull embedded images
    extract_image_block_types=["Image", "Table"],
)

# Elements are typed (Title, NarrativeText, Table, Image, etc.)
for element in elements:
    print(f"[{element.category}] {str(element)[:100]}")

# Filter by type
tables = [el for el in elements if el.category == "Table"]
narratives = [el for el in elements if el.category == "NarrativeText"]

# Tables come with HTML structure
for table in tables:
    print(table.metadata.text_as_html)
```

### With LangChain Integration

```python
from langchain_community.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader(
    "report.pdf",
    mode="elements",           # One Document per element
    strategy="hi_res",
    post_processors=[],
)

docs = loader.load()
for doc in docs:
    print(f"[{doc.metadata['category']}] {doc.page_content[:80]}")
```

---

## Approach 3: Vision LLM Extraction (Multimodal)

Use a vision-capable LLM to "read" document pages as images — best for complex layouts, handwriting, and understanding context.

### Google Gemini

```python
import google.generativeai as genai
from pdf2image import convert_from_path
from PIL import Image

model = genai.GenerativeModel("gemini-2.0-flash")

# Convert PDF to images
pages = convert_from_path("complex_report.pdf", dpi=200)

extracted = []
for i, page_image in enumerate(pages):
    response = model.generate_content([
        "Extract all text from this document page. "
        "Preserve structure: headings, bullet points, tables (as markdown). "
        "If there are tables, format them as markdown tables.",
        page_image,
    ])
    extracted.append({"page": i + 1, "content": response.text})

# Result: structured markdown from each page
print(extracted[0]["content"])
```

### OpenAI GPT-4o

```python
import base64
from openai import OpenAI
from pdf2image import convert_from_path
from io import BytesIO

client = OpenAI()

def pdf_page_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode()

pages = convert_from_path("document.pdf", dpi=200)

for page in pages:
    b64 = pdf_page_to_base64(page)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all text and tables from this page as markdown."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
    )
    print(response.choices[0].message.content)
```

---

## Approach 4: Specialised Document AI

### DocTR (Document Text Recognition)

Open-source, deep learning OCR that handles complex layouts.

```python
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Load model (downloads pre-trained weights)
predictor = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)

# Process PDF or image
doc = DocumentFile.from_pdf("scanned_form.pdf")
result = predictor(doc)

# Export structured output
json_output = result.export()

# Get plain text per page
for page in result.pages:
    for block in page.blocks:
        for line in block.lines:
            text = " ".join([word.value for word in line.words])
            print(text)
```

### Google Document AI

```python
from google.cloud import documentai_v1 as documentai

def process_document(project_id: str, location: str, processor_id: str, file_path: str):
    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(project_id, location, processor_id)

    with open(file_path, "rb") as f:
        content = f.read()

    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=content, mime_type="application/pdf"),
    )
    result = client.process_document(request=request)
    document = result.document

    # Full text
    print(document.text)

    # Entities (for form parsing processors)
    for entity in document.entities:
        print(f"{entity.type_}: {entity.mention_text} (confidence: {entity.confidence:.2f})")

    # Tables
    for page in document.pages:
        for table in page.tables:
            print(f"Table with {len(table.header_rows)} headers, {len(table.body_rows)} rows")

    return document
```

---

## Approach 5: PyMuPDF (Fast Digital PDF Extraction)

Best for digitally-born PDFs where text is already selectable.

```python
import pymupdf  # pip install pymupdf

doc = pymupdf.open("digital_report.pdf")

for page in doc:
    # Plain text
    text = page.get_text()

    # Structured blocks (preserves layout)
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        if block["type"] == 0:  # Text block
            for line in block["lines"]:
                for span in line["spans"]:
                    print(f"[{span['font']} {span['size']}pt] {span['text']}")
        elif block["type"] == 1:  # Image block
            print(f"Image: {block['width']}x{block['height']}")

    # Extract tables (PyMuPDF 1.23+)
    tables = page.find_tables()
    for table in tables:
        df = table.to_pandas()
        print(df)
```

---

## Approach 6: LlamaParse (Cloud Document Parsing)

LlamaParse is LlamaIndex's managed parsing service that uses LLMs to extract structured content from complex documents. Handles tables, charts, and multi-modal content well.

```bash
pip install llama-index llama-parse
```

### Basic Usage

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    api_key="llx-...",        # or set LLAMA_CLOUD_API_KEY env var
    result_type="markdown",    # "markdown" or "text"
    num_workers=4,
    verbose=True,
)

# Parse a document
documents = parser.load_data("complex_report.pdf")

for doc in documents:
    print(doc.text[:500])
```

### With Parsing Instructions

```python
parser = LlamaParse(
    result_type="markdown",
    parsing_instruction=(
        "This document contains financial tables and numerical data. "
        "Extract all tables preserving their structure as markdown tables. "
        "Preserve all numerical values exactly as they appear."
    ),
    # Use premium mode for complex documents
    premium_mode=True,
)

documents = parser.load_data("quarterly_report.pdf")
```

### With LlamaIndex RAG Pipeline

```python
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Use LlamaParse as the PDF parser in a directory reader
parser = LlamaParse(result_type="markdown")

file_extractor = {".pdf": parser}
reader = SimpleDirectoryReader("./documents", file_extractor=file_extractor)
documents = reader.load_data()

# Build index directly
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("What were the Q4 performance changes?")
print(response)
```

### With LangChain Integration

```python
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter

parser = LlamaParse(result_type="markdown")
documents = parser.load_data("report.pdf")

# Convert to LangChain documents
from llama_index.core.schema import Document as LIDocument
from langchain_core.documents import Document as LCDocument

lc_docs = [LCDocument(page_content=doc.text, metadata=doc.metadata) for doc in documents]

# Then chunk and embed as normal
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(lc_docs)
```

### Key Features

- Cloud-based — no local GPU or model setup needed
- LLM-powered parsing understands context and structure
- Handles tables, charts, images, and complex layouts
- Parsing instructions let you guide extraction behaviour
- Supports 10+ file formats (PDF, DOCX, PPTX, HTML, etc.)
- Free tier: 1000 pages/day

### Pros & Cons

- ✅ Excellent table and chart extraction
- ✅ No local dependencies or GPU required
- ✅ Parsing instructions for domain-specific needs
- ❌ Requires API key (cloud service)
- ❌ Not suitable for sensitive/private documents (data leaves your network)
- ❌ Rate limited on free tier

---

## Approach 7: Marker (ML-Powered PDF → Markdown)

Converts PDFs to clean markdown using deep learning models for layout detection, OCR, and table recognition.

```bash
pip install marker-pdf
```

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

models = create_model_dict()
converter = PdfConverter(artifact_dict=models)

# Convert entire PDF to markdown
result = converter("complex_report.pdf")
markdown_text = result.markdown

print(markdown_text)  # Clean markdown with headers, tables, lists
```

```bash
# CLI usage
marker_single input.pdf output/ --parallel_factor 2
```

---

## Full RAG Ingestion Pipeline

```python
from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb

# 1. Extract structured elements
elements = partition_pdf("report.pdf", strategy="hi_res", infer_table_structure=True)

# 2. Separate by type for different chunking strategies
texts = [el for el in elements if el.category in ("NarrativeText", "Title", "ListItem")]
tables = [el for el in elements if el.category == "Table"]

# 3. Chunk text elements
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = splitter.split_text("\n\n".join([str(el) for el in texts]))

# 4. Tables as single chunks (don't split tables)
table_chunks = [el.metadata.text_as_html for el in tables]

# 5. Embed and store
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")

all_chunks = text_chunks + table_chunks
all_embeddings = embeddings.embed_documents(all_chunks)

collection.add(
    documents=all_chunks,
    embeddings=all_embeddings,
    ids=[f"chunk_{i}" for i in range(len(all_chunks))],
    metadatas=[{"type": "text"} for _ in text_chunks] + [{"type": "table"} for _ in table_chunks],
)
```

---

## Tools Comparison

| Tool | Best For | OCR | Tables | Layout | Cost |
|------|----------|-----|--------|--------|------|
| PyMuPDF | Digital PDFs | No | Yes | Basic | Free |
| Tesseract | Simple scans | Yes | No | No | Free |
| Unstructured | General ingestion | Yes | Yes | Yes | Free |
| DocTR | Complex OCR | Yes | No | Yes | Free |
| Marker | PDF → Markdown | Yes | Yes | Yes | Free |
| LlamaParse | Complex layouts (cloud) | Yes | Yes | Yes | Free tier / paid |
| Gemini/GPT-4o | Complex/handwritten | Yes | Yes | Yes | Pay per page |
| Google Document AI | Enterprise forms | Yes | Yes | Yes | Pay per page |

## When to Use What

| Scenario | Recommended |
|----------|-------------|
| Clean digital PDFs | PyMuPDF (fast, free) |
| Scanned single-column docs | Tesseract or DocTR |
| Complex layouts + tables | Unstructured (hi_res) or Marker |
| Handwritten / messy scans | Vision LLM (Gemini/GPT-4o) |
| Enterprise form extraction | Google Document AI |
| RAG pipeline ingestion | Unstructured → LangChain |

## Best Practices

- Always check if the PDF has a text layer first (PyMuPDF) before running OCR
- Use 300 DPI when converting to images for OCR — lower resolution degrades accuracy
- Don't split tables across chunks — they lose meaning without full context
- Add metadata (page number, section, document title) to chunks for better retrieval
- For mixed documents, route pages through different pipelines based on content type
- Vision LLMs are expensive per page — use them selectively for complex pages only
- Pre-process images (deskew, denoise, binarise) before OCR for scanned documents
