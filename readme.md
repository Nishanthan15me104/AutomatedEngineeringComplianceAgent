# 📑 Table of Contents

- [🤖 Automated Engineering Compliance Agent](#-automated-engineering-compliance-agent)
- [🧠 Project Objective](#-project-objective)
- [🏗️ High-Level Architecture](#️-high-level-architecture)
- [📁 Project Structure](#-project-structure)
- [🔥 Core Technologies](#-core-technologies)
- [🧩 System Components](#-system-components)

  - [1️⃣ Multi-Modal RFQ Extraction Pipeline](#1️⃣-multi-modal-rfq-extraction-pipeline)
    - [Features](#features)
    - [✅ Text Extraction](#-text-extraction)
    - [✅ Smart Text Chunking](#-smart-text-chunking)
    - [✅ Table Extraction](#-table-extraction)
    - [✅ Image Captioning](#-image-captioning)
    - [✅ Semantic Domain Classification](#-semantic-domain-classification)

  - [2️⃣ Vector Database Pipeline](#2️⃣-vector-database-pipeline)
    - [Workflow](#workflow)
    - [RFQ Processing](#rfq-processing)
    - [Product Catalog Indexing](#product-catalog-indexing)
    - [Embedding Model](#embedding-model)
    - [Vector Database](#vector-database)

  - [3️⃣ Agentic Compliance Engine](#3️⃣-agentic-compliance-engine)
    - [🧠 LangGraph Workflow](#-langgraph-workflow)
    - [Agent Steps](#agent-steps)
    - [STEP 1 — Retrieve RFQ Requirements](#step-1--retrieve-rfq-requirements)
    - [STEP 2 — Retrieve Product Specifications](#step-2--retrieve-product-specifications)
    - [STEP 3 — Compliance Reasoning](#step-3--compliance-reasoning)

- [✅ Compliance Checks Performed](#-compliance-checks-performed)
- [📄 Final Output](#-final-output)
- [⚡ Key Engineering Features](#-key-engineering-features)
  - [✅ Multi-Modal Intelligence](#-multi-modal-intelligence)
  - [✅ Semantic Retrieval](#-semantic-retrieval)
  - [✅ Metadata Filtering](#-metadata-filtering)
  - [✅ Agentic Reasoning](#-agentic-reasoning)
  - [✅ Local Vector Storage](#-local-vector-storage)

- [⚙️ Installation](#️-installation)
  - [1️⃣ Clone Repository](#1️⃣-clone-repository)
  - [2️⃣ Create Virtual Environment](#2️⃣-create-virtual-environment)
  - [3️⃣ Install Dependencies](#3️⃣-install-dependencies)

- [🔑 Environment Variables](#-environment-variables)
- [🚀 Running the System](#-running-the-system)
  - [STEP 1 — Extract RFQ Intelligence](#step-1--extract-rfq-intelligence)
  - [STEP 2 — Build Vector Database](#step-2--build-vector-database)
  - [STEP 3 — Run Compliance Agent](#step-3--run-compliance-agent)

- [📊 Example Compliance Output](#-example-compliance-output)

- [🛠️ Engineering Challenges Solved](#️-engineering-challenges-solved)
  - [1️⃣ Semantic RFQ Understanding](#1️⃣-semantic-rfq-understanding)
  - [2️⃣ Table Continuation Detection](#2️⃣-table-continuation-detection)
  - [3️⃣ Domain Classification](#3️⃣-domain-classification)
  - [4️⃣ Retrieval Precision](#4️⃣-retrieval-precision)
  - [5️⃣ Scalable Agent Orchestration](#5️⃣-scalable-agent-orchestration)

- [🚧 Project Evolution: Challenges & Solutions](#-project-evolution-challenges--solutions)
  - [1️⃣ Environment & Dependency Isolation](#1️⃣-environment--dependency-isolation)
  - [2️⃣ Vector Storage Evolution — FAISS → Qdrant](#2️⃣-vector-storage-evolution--faiss--qdrant)
  - [3️⃣ Physics-Aware Retrieval](#3️⃣-physics-aware-retrieval)
  - [4️⃣ Technical Robustness & Stability](#4️⃣-technical-robustness--stability)
  - [5️⃣ Transition to Agentic Logic](#5️⃣-transition-to-agentic-logic)
  - [6️⃣ Migration to LangGraph State-Based Workflow](#6️⃣-migration-to-langgraph-state-based-workflow)
  - [7️⃣ Hard Metadata Filtering & Vector Consistency](#7️⃣-hard-metadata-filtering--vector-consistency)
  - [8️⃣ Embedding Consistency Synchronization](#8️⃣-embedding-consistency-synchronization)

- [🏆 Final Outcome](#-final-outcome)
- [📈 Future Improvements](#-future-improvements)
- [🧪 Potential Enterprise Use Cases](#-potential-enterprise-use-cases)
- [🏆 Key Achievements](#-key-achievements)
- [👨‍💻 Tech Stack Summary](#-tech-stack-summary)

---




# 🤖 Automated Engineering Compliance Agent

An enterprise-grade **Agentic AI Compliance System** designed to automate engineering RFQ (Request for Quotation) auditing against industrial product catalogs using:

- ✅ Multi-modal document extraction
- ✅ Semantic chunking
- ✅ Vector databases
- ✅ RAG (Retrieval-Augmented Generation)
- ✅ LangGraph agent workflows
- ✅ Qdrant vector search
- ✅ LLM-powered compliance reasoning

The system extracts structured intelligence from RFQ PDFs, semantically indexes engineering requirements, retrieves matching product specifications, and performs automated compliance audits.

---

# 🧠 Project Objective

Industrial RFQ auditing is traditionally:
- manual
- time-consuming
- error-prone
- engineering-intensive

This project automates the process using:
- AI agents
- vector search
- semantic retrieval
- LLM reasoning
- multi-modal understanding

The result:
- Faster engineering validation
- Automated compliance checking
- Reduced human review effort
- Scalable procurement intelligence

---

# 🏗️ High-Level Architecture

```text
                  ┌──────────────────────┐
                  │      RFQ PDF         │
                  └──────────┬───────────┘
                             │
                             ▼
              ┌───────────────────────────┐
              │ Multi-Modal Extraction    │
              │ PyMuPDF + pdfplumber      │
              └──────────┬────────────────┘
                         │
         ┌───────────────┼────────────────┐
         │               │                │
         ▼               ▼                ▼
    Text Chunks      Tables          Images
         │               │                │
         └───────────────┼────────────────┘
                         ▼
           ┌─────────────────────────┐
           │ Semantic Enrichment     │
           │ BLIP + CLIP Tagging     │
           └──────────┬──────────────┘
                      ▼
           ┌─────────────────────────┐
           │ Qdrant Vector Database  │
           └──────────┬──────────────┘
                      ▼
           ┌─────────────────────────┐
           │ LangGraph Agent Flow    │
           └──────────┬──────────────┘
                      ▼
           ┌─────────────────────────┐
           │ LLM Compliance Engine   │
           └──────────┬──────────────┘
                      ▼
           ┌─────────────────────────┐
           │ PASS / FAIL Report      │
           └─────────────────────────┘
```

---

# 📁 Project Structure

```text
AutomatedEngineeringComplianceAgent/
│
├── data/
│   ├── input/
│   │   ├── sample_rfq.pdf
│   │   └── products.csv
│   │
│   ├── extracted/
│   │   ├── images/
│   │   └── metadata.json
│   │
│   └── qdrant_storage/
│
├── src/
│   ├── extract_pipeline.py
│   ├── build_vector_db.py
│   └── compliance_agent.py
│
├── .env
├── requirements.txt
└── README.md
```

---

# 🔥 Core Technologies

| Category | Technologies |
|---|---|
| Language | Python 3.11 |
| PDF Extraction | PyMuPDF, pdfplumber |
| Computer Vision | BLIP |
| Semantic Similarity | CLIP |
| Embeddings | Sentence Transformers |
| Vector DB | Qdrant |
| Agent Framework | LangGraph |
| LLM | Groq Llama 3.3 70B |
| Orchestration | LangChain |
| Data Processing | Pandas, NumPy |
| Deep Learning | PyTorch |

---

# 🧩 System Components

---

# 1️⃣ Multi-Modal RFQ Extraction Pipeline

File:

```python
extract_pipeline.py
```

This pipeline converts raw engineering PDFs into semantic intelligence.

---

## Features

### ✅ Text Extraction
Uses:
- :contentReference[oaicite:0]{index=0}
- :contentReference[oaicite:1]{index=1}

Extracts:
- paragraphs
- specifications
- engineering clauses
- technical requirements

---

### ✅ Smart Text Chunking

Large RFQ documents are split into overlapping semantic chunks.

```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
```

### Why Overlap Matters

Overlap preserves:
- engineering context
- specification continuity
- semantic relationships

This improves:
- vector retrieval quality
- downstream LLM reasoning

---

### ✅ Table Extraction

Extracts:
- dimensional tables
- engineering specifications
- operating parameters

Includes:
- table continuation detection
- multi-page merge logic

---

### ✅ Image Captioning

Uses:
- :contentReference[oaicite:2]{index=2}

Automatically generates captions for:
- engineering diagrams
- machinery images
- industrial schematics

Example:

```text
"Image showing: industrial air compressor with cooling system"
```

---

### ✅ Semantic Domain Classification

Uses:
- :contentReference[oaicite:3]{index=3} embeddings

Automatically classifies extracted content into domains:

| Domain | Description |
|---|---|
| Structural | Dimensions, steel, weight |
| Thermal | Cooling, temperature |
| Flow | Ventilation, air pressure |
| General | Commercial/business |

---

# 2️⃣ Vector Database Pipeline

File:

```python
build_vector_db.py
```

Creates the semantic retrieval layer.

---

## Workflow

### RFQ Processing

Each extracted chunk becomes:
- embedding vector
- searchable semantic unit

Stored metadata includes:
- source
- page number
- type
- engineering domain

---

### Product Catalog Indexing

Product CSV data is converted into:
- engineering specification embeddings
- searchable catalog vectors

Example Product Features:
- CFM capacity
- ISO certification
- noise level
- product type

---

## Embedding Model

Uses:

```python
BAAI/bge-small-en-v1.5
```

Benefits:
- lightweight
- fast inference
- strong semantic retrieval performance

---

## Vector Database

Uses:
- :contentReference[oaicite:4]{index=4}

### Why Qdrant?

- high-speed similarity search
- local deployment support
- metadata filtering
- scalable vector indexing

---

# 3️⃣ Agentic Compliance Engine

File:

```python
compliance_agent.py
```

This is the reasoning layer of the system.

---

# 🧠 LangGraph Workflow

```text
get_rfq
   ↓
get_product
   ↓
compliance_check
   ↓
END
```

---

## Agent Steps

---

### STEP 1 — Retrieve RFQ Requirements

The system:
- converts query into embedding vectors
- searches RFQ semantic chunks
- retrieves engineering constraints

Hard filtering ensures:
- ONLY RFQ documents are searched

---

### STEP 2 — Retrieve Product Specifications

Searches product catalog vectors using:
- product ID
- semantic similarity
- metadata filtering

Hard filtering ensures:
- ONLY catalog vectors are searched

---

### STEP 3 — Compliance Reasoning

Uses:
- :contentReference[oaicite:5]{index=5} hosted
- :contentReference[oaicite:6]{index=6}

The LLM compares:
- RFQ engineering requirements
- product specifications

---

# ✅ Compliance Checks Performed

The agent evaluates:

- temperature limits
- airflow requirements
- dimensional mismatches
- pressure constraints
- ISO certification compliance
- engineering tolerances
- operational compatibility

---

# 📄 Final Output

The system generates:

```text
PASS / FAIL Verdict
```

Along with:
- detailed engineering reasoning
- mismatch explanations
- specification comparisons

---

# ⚡ Key Engineering Features

---

## ✅ Multi-Modal Intelligence

Supports:
- text
- tables
- diagrams
- images

---

## ✅ Semantic Retrieval

Uses vector similarity instead of keyword search.

Benefits:
- understands engineering meaning
- improves retrieval quality
- handles terminology variation

---

## ✅ Metadata Filtering

Qdrant hard filters isolate:
- RFQ data
- catalog data

Prevents retrieval contamination.

---

## ✅ Agentic Reasoning

LangGraph enables:
- modular workflows
- deterministic execution
- scalable AI orchestration

---

## ✅ Local Vector Storage

Qdrant local persistence allows:
- offline usage
- no cloud dependency
- lightweight deployment

---

# ⚙️ Installation

---

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/AutomatedEngineeringComplianceAgent.git

cd AutomatedEngineeringComplianceAgent
```

---

## 2️⃣ Create Virtual Environment

### Windows

```powershell
python -m venv .venv

.\.venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create `.env`

```env
GROQ_API_KEY=your_groq_api_key
```

---

# 🚀 Running the System

---

# STEP 1 — Extract RFQ Intelligence

```bash
python src/extract_pipeline.py
```

Output:
- extracted semantic chunks
- image captions
- metadata.json

---

# STEP 2 — Build Vector Database

```bash
python src/build_vector_db.py
```

Output:
- local Qdrant vector database

---

# STEP 3 — Run Compliance Agent

```bash
python src/compliance_agent.py
```

---

# 📊 Example Compliance Output

```text
========================================
FINAL COMPLIANCE REPORT
========================================

VERDICT: FAIL

Reason:
- Product airflow capacity below RFQ requirement
- Missing ISO 9001 certification
- Noise level exceeds engineering tolerance

Recommendation:
Use AC-200-XP alternative model.
```

---

# 🛠️ Engineering Challenges Solved

---

# 1️⃣ Semantic RFQ Understanding

### Problem
Engineering RFQs contain:
- unstructured language
- large specifications
- mixed modalities

### Solution
Implemented:
- chunking
- semantic embeddings
- multi-modal extraction

---

# 2️⃣ Table Continuation Detection

### Problem
Engineering tables span multiple pages.

### Solution
Custom heuristic:
- column-count matching
- intelligent merge logic

---

# 3️⃣ Domain Classification

### Problem
Need engineering-aware retrieval.

### Solution
CLIP-based semantic tagging:
- thermal
- structural
- flow
- general

---

# 4️⃣ Retrieval Precision

### Problem
Catalog vectors contaminating RFQ retrieval.

### Solution
Hard metadata filtering inside Qdrant.

---

# 5️⃣ Scalable Agent Orchestration

### Problem
Complex compliance workflow management.

### Solution
LangGraph deterministic state machine.

---
# 🚧 Project Evolution: Challenges & Solutions

This project evolved from a simple semantic search prototype into a production-style **Agentic Engineering Compliance System**.

The journey involved solving multiple real-world engineering, architecture, retrieval, and infrastructure challenges.

---

# 1️⃣ Environment & Dependency Isolation

## ❌ Challenge

Installing Python packages globally caused:
- dependency conflicts
- unstable environments
- OS-level package pollution
- inconsistent reproducibility across machines

---

## ✅ Solution

Implemented isolated Python virtual environments using:

```bash
python -m venv .venv
```

### Benefits
- dependency isolation
- reproducible builds
- safer experimentation
- cleaner deployment environments

---

# 2️⃣ Vector Storage Evolution — FAISS → Qdrant

## ❌ Challenge

Initial implementation used :contentReference[oaicite:0]{index=0}.

However, FAISS is:
- a vector indexing library
- NOT a full database

This created architectural limitations:
- vectors stored separately
- metadata stored in JSON files
- synchronization risks
- potential data corruption
- difficult persistence management

The architecture relied on:

```text
FAISS Index + Separate JSON Metadata
```

which introduced operational complexity.

---

## ✅ Solution

Migrated to :contentReference[oaicite:1]{index=1}.

Qdrant stores:
- vectors
- metadata
- payloads
- searchable attributes

inside a single atomic structure:

```python
PointStruct(
    id=idx,
    vector=vector,
    payload={...}
)
```

### Benefits
- unified storage architecture
- metadata-aware search
- safer persistence
- scalable retrieval
- production-grade vector database design

---

# 3️⃣ Physics-Aware Retrieval

## ❌ Challenge

Basic semantic retrieval struggled to understand engineering domains such as:
- Thermal systems
- Structural specifications
- Flow mechanics

A generic embedding search could retrieve:
- semantically similar text
- but physically irrelevant engineering concepts

Example:

```text
Cooling airflow requirements
```

could incorrectly match:

```text
Structural airflow housing dimensions
```

---

## ✅ Solution

Implemented **Metadata Injection**.

Before embedding, domain tags are injected directly into the text:

```text
[Thermal] Heat dissipation requirement...
```

This ensures vector embeddings mathematically preserve:
- engineering context
- domain semantics
- physics-aware relationships

---

## 🚀 Result

Retrieval quality significantly improved for:
- engineering constraints
- domain-specific searches
- specification matching

---

# 4️⃣ Technical Robustness & Stability

## ❌ Challenge

Encountered multiple infrastructure issues:

### DeprecationWarnings
Older Qdrant collection methods became deprecated.

### ImportErrors During Shutdown
Python interpreter shutdown occasionally interrupted:
- local Qdrant persistence
- background cleanup operations

This risked:
- incomplete writes
- corrupted local database states

---

## ✅ Solution

### Updated Collection Management

Migrated to newer APIs:

```python
client.collection_exists()
client.create_collection()
```

instead of deprecated recreation methods.

---

### Safe Database Shutdown

Wrapped database lifecycle in:

```python
try:
    ...
finally:
    client.close()
```

This guarantees:
- safe persistence
- graceful shutdown
- stable local vector storage

---

# 5️⃣ Transition to Agentic Logic

## ❌ Challenge

Traditional vector search systems only retrieve text.

They do NOT:
- reason
- compare specifications
- validate constraints
- make engineering decisions

The system needed actual compliance intelligence.

---

## ✅ Solution

Migrated toward an **Agentic AI Architecture** using:
- :contentReference[oaicite:2]{index=2}
- LLM reasoning
- deterministic workflow orchestration

The system evolved from:
- semantic retrieval

to:
- engineering decision-making

---

## Example Engineering Logic

The agent can now compare:

```text
RFQ Requirement: 50°C operating temperature
```

against:

```text
Product Rating: 40°C maximum
```

and intelligently determine:

```text
FAIL — Product violates thermal requirement
```

---

# 6️⃣ Migration to LangGraph State-Based Workflow

## 🔄 Architectural Change

Transitioned from:
- linear procedural scripts

to:
- state-managed agent workflows

using :contentReference[oaicite:3]{index=3}.

---

## ✅ Implemented Workflow

```text
Extract Requirements
        ↓
Fetch Product Specs
        ↓
Verify Compliance
```

---

## 🚀 Benefits

The workflow now guarantees:
- deterministic execution
- modular orchestration
- traceable reasoning
- controlled LLM behavior

This prevents the LLM from:
- hallucinating answers
- skipping validation steps
- making unsupported compliance claims

---

# 7️⃣ Hard Metadata Filtering & Vector Consistency

## ❌ Challenge — "Data Blindness"

The system initially struggled to retrieve:
- exact product IDs
- precise catalog records

because fuzzy vector similarity alone was insufficient.

This caused:
- incorrect retrievals
- weak compliance verification
- noisy search results

---

## ✅ Solution — Hard Metadata Filtering

Implemented Qdrant metadata filters:

```python
MatchValue(value="CATALOG")
```

This enforces:
- strict source isolation
- exact metadata constraints
- deterministic retrieval behavior

---

## ❌ Additional Challenge — AssertionError

Encountered compatibility issues with:
- updated Qdrant query APIs
- mismatched query parameter structures

---

## ✅ Fix

Aligned all query logic with the latest:
- Qdrant client version
- query parameter expectations

---

# 8️⃣ Embedding Consistency Synchronization

## ❌ Challenge

Different embedding models between:
- vector database creation
- agent retrieval

caused vector-space inconsistency.

This breaks:
- semantic math alignment
- similarity calculations
- retrieval accuracy

---

## ✅ Solution

Standardized the entire system on:

```text
BAAI/bge-small-en-v1.5
```

for:
- indexing
- retrieval
- semantic querying

---

## 🚀 Result

Achieved:
- mathematically consistent vector search
- stable semantic retrieval
- improved engineering precision

---

# 🏆 Final Outcome

The project successfully evolved into a:

## ✅ Production-Style Agentic Engineering Compliance Platform

with:
- multi-modal RFQ understanding
- physics-aware semantic retrieval
- vector database architecture
- deterministic agent workflows
- engineering-grade compliance reasoning
- scalable modular infrastructure

---

# 📈 Future Improvements

- Multi-agent engineering reviewers
- OCR for scanned PDFs
- CAD drawing understanding
- Hybrid BM25 + vector retrieval
- Streaming compliance reports
- GPU vector acceleration
- Kubernetes deployment
- Real-time RFQ ingestion APIs

---

# 🧪 Potential Enterprise Use Cases

- Industrial procurement
- Vendor compliance auditing
- Engineering bid evaluation
- Manufacturing validation
- Supply-chain qualification
- Technical specification matching

---

# 🏆 Key Achievements

- ✅ End-to-end Agentic AI workflow
- ✅ Multi-modal document intelligence
- ✅ Production-ready vector retrieval
- ✅ Automated engineering reasoning
- ✅ Modular scalable architecture
- ✅ Real-world procurement automation

---

# 👨‍💻 Tech Stack Summary

| Layer | Technologies |
|---|---|
| Extraction | PyMuPDF, pdfplumber |
| Vision | BLIP, CLIP |
| Embeddings | SentenceTransformers |
| Vector DB | Qdrant |
| Agent Framework | LangGraph |
| LLM | Groq Llama 3.3 |
| Backend | Python |
| ML Framework | PyTorch |

---






Project Evolution: Challenges & Solutions
We have moved from a basic script to a professional-grade vector architecture. Here is the summary of the "Engineering Compliance Agent" journey:

1. Environment & Setup
Challenge: Installing packages globally leads to "OS pollution" and version conflicts.

Solution: Implemented Virtual Environments (venv) to isolate dependencies.

2. The Vector Storage Pivot (FAISS vs. Qdrant)
Challenge: FAISS is a library, not a database. It required keeping two separate files (Index + JSON), which is risky for data integrity.

Solution: Migrated to Qdrant. It uses a PointStruct to store the Vector, the Text, and the Payload (Metadata) in a single, atomic record.

3. Physics-Aware Retrieval
Challenge: Simple keyword search doesn't understand engineering pillars like [Thermal], [Structural], or [Flow].

Solution: Implemented Metadata Injection. We "bake" domain tags into the text before embedding, ensuring the vector math respects engineering context.

4. Technical Robustness
Challenge: Encountered DeprecationWarnings for collection management and ImportErrors during Python shutdown.

Solution:

Updated logic to use client.collection_exists() and client.create_collection().

Wrapped the build process in a try/finally block to ensure client.close() is always called, protecting the database state.

5. Transition to Agentic Logic
Challenge: A search engine just gives text; it doesn't make decisions.

Solution: We are now moving toward LangGraph, transforming the script into an "Agent" that can perform 1D Logic (comparing 50°C requirement vs 40°C product rating).

6. Migration to Agentic Workflow (LangGraph)The Change: Transitioned from a linear script to a state-managed LangGraph agent.The Fix: Implemented a three-stage pipeline: Extract Requirements $\rightarrow$ Fetch Specs $\rightarrow$ Verify Compliance. This ensures the LLM doesn't just "guess" but follows a strict engineering verification process.

7. "Hard" Metadata Filtering & Point ID FixesThe Change: Resolved "Data Blindness" where the agent could not find specific product IDs using fuzzy vector search.The Fix:Implemented Qdrant Hard Filters (MatchValue) to ensure 100% accuracy when retrieving specific product IDs.Fixed the AssertionError in the Qdrant API by aligning query parameters with the latest client version.Synchronized the SentenceTransformer across the database and agent to ensure text-to-vector mathematical consistency.