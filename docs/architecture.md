# Architecture

```mermaid
flowchart LR
    subgraph IDX["Indexing workflow (run once per document set)"]
        A[PDF upload<br/>Colab file picker] --> B[Validation<br/>extension, size, readable]
        B --> C[Extraction<br/>PyMuPDF, page by page]
        C --> D[Cleaning<br/>conservative normalisation]
        D --> E[Chunking<br/>900 chars / 150 overlap]
        E --> F[Embeddings<br/>MiniLM multilingual, normalised, float32]
        F --> G[(FAISS IndexFlatIP<br/>+ chunks.json metadata)]
    end
    subgraph QA["Question answering workflow (per question)"]
        Q[User question<br/>notebook or Gradio] --> R[Encode question<br/>same embedding model]
        R --> S[Top-K search<br/>cosine similarity]
        G -.-> S
        S --> T[Build prompt<br/>numbered sources + filename + page]
        T --> U[Qwen2.5 Instruct<br/>chat template, greedy decoding]
        U --> V[Answer + citation check<br/>sources, pages, scores]
    end
    subgraph EV["Evaluation"]
        W[Eval questions<br/>page-level labels] --> X[Precision@K, Recall@K, MRR]
        S -.-> X
        V -.-> Y[Human rubric 1-5<br/>+ abstention tests]
    end
```

**Key invariant:** the *i*-th FAISS vector corresponds to the *i*-th entry of `chunks.json`; `validate_knowledge_base` checks this on every build and load.
