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