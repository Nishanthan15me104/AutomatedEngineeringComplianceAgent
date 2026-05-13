import os
from typing import TypedDict, List
from pathlib import Path
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue # Added for hard filtering
from sentence_transformers import SentenceTransformer # Added for text-to-vector
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# Suppress HuggingFace warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------
# 1. Configuration & Environment
# -----------------------------
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
QDRANT_PATH = BASE_DIR / "data" / "qdrant_storage"
COLLECTION_NAME = "compliance_audit"

class AgentState(TypedDict):
    query: str
    product_id: str
    rfq_constraints: List[str]
    product_specs: List[str]
    compliance_report: str

# Initialize Models
print("Loading Embedding Model & LLM...")
embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5') # The missing piece!
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_retries=2)

# Connect to Qdrant
client = QdrantClient(path=str(QDRANT_PATH))

# -----------------------------
# 2. Modular Tools
# -----------------------------

def retrieve_rfq_requirements(state: AgentState):
    """Extracts technical limits from the indexed RFQ."""
    print("--- STEP: Extracting RFQ Requirements ---")
    search_query = f"Technical requirements and limits for {state['query']}"
    
    # Encode text to vector
    query_vector = embed_model.encode(search_query).tolist()
    
    # Search with a Hard Filter: ONLY look at RFQ sources
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value="RFQ"))]
        ),
        limit=5
    ).points
    
    constraints = [res.payload['content'] for res in results]
    return {"rfq_constraints": constraints}

def retrieve_product_catalog(state: AgentState):
    """Fetches specs for the target product ID."""
    print(f"--- STEP: Fetching Specs for {state['product_id']} ---")
    
    # Encode text to vector
    query_vector = embed_model.encode(state['product_id']).tolist()
    
    # Search with a Hard Filter: ONLY look at CATALOG sources
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value="CATALOG"))]
        ),
        limit=2
    ).points
    
    specs = [res.payload['content'] for res in results]
    return {"product_specs": specs}

def check_compliance(state: AgentState):
    """Performs the engineering comparison (1D Logic)."""
    print("--- STEP: Comparing Constraints (1D Logic) ---")
    
    prompt = f"""
    You are an Industrial Compliance Engineer. Compare the RFQ Requirements against the Product Specs.
    
    RFQ REQUIREMENTS:
    {state['rfq_constraints']}
    
    PRODUCT SPECIFICATIONS:
    {state['product_specs']}
    
    TASK:
    1. Identify specific numerical mismatches (Temp, Pressure, etc.).
    2. Check for certification mismatches (ISO 9001).
    3. Issue a final 'PASS' or 'FAIL' verdict.
    4. Provide clear reasoning for the decision.
    """
    
    response = llm.invoke(prompt)
    return {"compliance_report": response.content}

# -----------------------------
# 3. Graph Construction
# -----------------------------

workflow = StateGraph(AgentState)

workflow.add_node("get_rfq", retrieve_rfq_requirements)
workflow.add_node("get_product", retrieve_product_catalog)
workflow.add_node("compliance_check", check_compliance)

workflow.set_entry_point("get_rfq")
workflow.add_edge("get_rfq", "get_product")
workflow.add_edge("get_product", "compliance_check")
workflow.add_edge("compliance_check", END)

app = workflow.compile()

# -----------------------------
# 4. Execution
# -----------------------------
if __name__ == "__main__":
    inputs = {
        "query": "Screw type Air Compressor",
        "product_id": "AC-100-XP"
    }
    
    try:
        print("🚀 Starting Agentic Audit...")
        final_state = app.invoke(inputs)
        
        print("\n" + "="*40)
        print("FINAL COMPLIANCE REPORT")
        print("="*40)
        print(final_state['compliance_report'])
        
    finally:
        client.close()
        print("\n📁 Database connection closed.")