# Intelligent Document Assistant: Building a Retrieval-Augmented Generation System

An end-to-end, reproducible **RAG** pipeline for Google Colab: upload PDFs, extract and chunk their text, index the chunks with
**Sentence Transformers + FAISS**, ask questions in natural language and receive answers **grounded in retrieved passages**,
with filenames, page numbers and similarity scores. Includes retrieval metrics (Precision@K, Recall@K, MRR), a human evaluation
rubric, reproducible experiments and a Gradio interface.

> Academic project - 5th year Computer Science Engineering (Artificial Intelligence). Author: _[your name]_.

## Objectives
1. Build a document index from PDFs (extract, clean, chunk, embed, store).
2. Retrieve relevant passages with semantic search.
3. Generate answers from the retrieved context with a Hugging Face LLM, and abstain when evidence is missing.
4. Evaluate retrieval and answers, inspect failures and document limitations.

## Architecture
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

## Technologies
Python - Google Colab - PyMuPDF - Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`) - FAISS (`IndexFlatIP`) -
Hugging Face Transformers (`Qwen/Qwen2.5-3B-Instruct`, fallback `Qwen/Qwen2.5-1.5B-Instruct`) - PyTorch - NumPy - Pandas - Matplotlib - Gradio.

## Repository structure
```
rag-intelligent-document-assistant/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── RAG_Intelligent_Document_Assistant.ipynb   # the complete implementation
├── src/
│   └── rag_utils.py        # model-free helpers: extraction, cleaning, chunking, metrics
├── evaluation/
│   ├── eval_questions.json # questions + page-level labels (for RAG_Detailed.pdf)
│   └── results/            # export YOUR run results here
├── docs/
│   └── architecture.md     # Mermaid diagram
└── data/
    └── documents/          # PDFs go here locally (git-ignored)
```

## Google Colab execution
1. Push the repository to GitHub, then open the notebook in Colab:
   `https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/notebooks/RAG_Intelligent_Document_Assistant.ipynb`
   (or upload the `.ipynb` through *File -> Upload notebook*).
2. `Runtime -> Change runtime type -> GPU (T4)` (recommended; CPU works but generation is very slow).
3. `Runtime -> Run all`. When the upload cell appears, choose your PDF files (e.g. `RAG_Detailed.pdf`).
4. Use the Gradio interface cell, or call `ask_question("your question", top_k=4)`.

No token is required for the default public models. For gated models, store `HF_TOKEN` in Colab *Secrets* - never in code.

## Local installation (optional)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/RAG_Intelligent_Document_Assistant.ipynb
```
(Install the PyTorch build matching your hardware from https://pytorch.org/get-started/locally/ if needed. The upload cell is Colab-specific; locally copy PDFs into `data/documents/`.)

## Example usage
```python
result = ask_question("What chunk size and overlap does the guide use?", top_k=4)
result["answer"]      # grounded answer with [source numbers]
result["sources"]     # [{'rank', 'source', 'page', 'chunk_id', 'score'}, ...]
```

## Evaluation methodology
* **Dataset:** `evaluation/eval_questions.json` - 8 answerable and 3 unanswerable questions about `RAG_Detailed.pdf`, with reference answers and **page-level** relevance labels (chunk ids change with the chunking configuration). Labels must be re-checked manually against the actual PDF.
* **Retrieval:** Precision@K = relevant retrieved / K; Recall@K = relevant retrieved / all relevant; RR = 1 / rank of first relevant; MRR = mean RR. Page-level handling and edge cases are explained in the notebook (Step 14).
* **Generation:** human rubric (1-5) for factual correctness, relevance, completeness, grounding, clarity and source accuracy; automatic checks only for abstention and citation format - they do **not** prove correctness.
* **Experiments:** one parameter changed at a time (chunk size, overlap, Top-K) on the same questions; latency, index build time and number of chunks recorded.

## Results
> Fill in **after running the notebook**. Do not copy numbers from elsewhere.

Environment of the run: _GPU/CPU: ... | generation model used: ... | date: ..._  (see `run_summary.json`)

| Configuration (size/overlap, K) | Precision@K | Recall@K | MRR | Retrieval ms | #Chunks |
|---|---|---|---|---|---|
| Baseline 900/150, K=4 | _to fill_ | _to fill_ | _to fill_ | _to fill_ | _to fill_ |
| 600/150, K=4 | _to fill_ | _to fill_ | _to fill_ | _to fill_ | _to fill_ |
| 1200/150, K=4 | _to fill_ | _to fill_ | _to fill_ | _to fill_ | _to fill_ |

Human evaluation (mean 1-5): _to fill_ - Abstention on unanswerable questions: _to fill_ / 3.

## Limitations
Retrieval errors, missing information, generation errors, extraction problems (scans need OCR, tables, columns), embedder truncation (128 word-pieces for the baseline model), small evaluation set labelled by one annotator, privacy of uploaded documents, and third-party licenses. RAG reduces reliance on unsupported model knowledge but **cannot guarantee correctness**; verify answers against the cited pages.

## License
Original code: MIT (see `LICENSE`). Third-party libraries, pretrained models and indexed documents keep their own licenses.

## References
* Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020.
* PyMuPDF: https://pymupdf.readthedocs.io/ - Sentence Transformers: https://www.sbert.net/ - FAISS: https://github.com/facebookresearch/faiss
* Hugging Face Transformers: https://huggingface.co/docs/transformers/ - Qwen2.5-3B-Instruct: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
* PyTorch: https://pytorch.org/get-started/locally/ - Course material: *A Practical Guide to RAG* (`RAG_Detailed.pdf`).
