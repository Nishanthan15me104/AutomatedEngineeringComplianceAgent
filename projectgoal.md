Conversational CAD & Automated RFQ Analysis."

Between RAG and Image Extraction, you should actually do a Multi-Modal RAG project. This is because engineering documents (RFQs) are full of tables and diagrams that simple text-RAG cannot read.

Here is a project idea that will make you stand out from all the M.Tech freshers:

Project Title: "Automated Engineering Compliance Agent"
The Goal: Build an AI tool that reads a technical RFQ (PDF) and tells an engineer if a specific product "fits" the requirements based on physical constraints.

Step-by-Step Implementation:
1. The Data Input (Multi-Modal Extraction)
Instead of just reading text, use a library like unstructured or PyMuPDF to extract:

Text: Technical requirements (e.g., "Operating temperature: 50°C").

Tables: Performance charts (e.g., Voltage vs. Efficiency).

Images: Schematic diagrams of the space constraints.

2. The Physics-Aware Vector Store (ChromaDB / FAISS)
Store the extracted data, but add Metadata that categorizes each chunk as [Structural], [Thermal], or [Flow]. This shows you understand the job's core pillars.

3. The "Conversational CAD" Logic (The Agent)
Build a LangGraph or LangChain agent that performs "Constraint Checking."

User Query: "Does our 'Model-X' breaker work for this RFQ?"

Agent Action:

Retrieve the RFQ's temperature requirement (Thermal).

Retrieve the physical dimensions allowed (Structural).

Compare these against a "Product Database" (a simple CSV you create).

Output: "Model-X is Incompatible. The RFQ requires 50°C, but Model-X is only rated for 40°C. Also, the width exceeds the 2m limit by 10cm."

Why this project beats others:
It solves their RFQ problem: You are proving you can automate the "Literature Studies" they mentioned.

It handles 1D Logic: By comparing numbers (50°C vs 40°C), you are doing basic 1D Modelling logic in code.

It uses your BA skills: You are turning a messy business document into a clear "Pass/Fail" decision.

How to talk about it in the interview:
"I developed a Multi-Modal RAG system specifically for Industrial RFQs. It doesn't just find text; it extracts physical constraints like thermal limits and structural dimensions from tables and diagrams. I then used a custom agentic workflow to automate Compliance Mapping, reducing the time an engineer spends reviewing a technical bid by 80%."