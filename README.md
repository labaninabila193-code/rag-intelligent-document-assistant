# Intelligent Document Assistant: Building a Retrieval-Augmented Generation System

An end-to-end, reproducible **RAG** pipeline for Google Colab: upload PDFs, extract and chunk their text, index the chunks with
**Sentence Transformers + FAISS**, ask questions in natural language and receive answers **grounded in retrieved passages**,
with filenames, page numbers and similarity scores. Includes retrieval metrics (Precision@K, Recall@K, MRR), a human evaluation
rubric, reproducible experiments and a Gradio interface.

> Academic project - 5th year Computer Science Engineering (Artificial Intelligence). Author: **LABANI NABILA NOUR ELHOUDA**.

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
│   └── results/            # exported results of the run
├── docs/
│   └── architecture.md     # Mermaid diagram
└── data/
    └── documents/          # PDFs go here locally (git-ignored)
```

## Google Colab execution
1. Open the notebook in Colab:
   `https://colab.research.google.com/github/labaninabila193-code/rag-intelligent-document-assistant/blob/main/notebooks/RAG_Intelligent_Document_Assistant.ipynb`
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
* **Dataset:** `evaluation/eval_questions.json` - 8 answerable and 3 unanswerable questions about `RAG_Detailed.pdf`, with reference answers and **page-level** relevance labels (chunk ids change with the chunking configuration). Labels were written by one annotator and must be re-checked against the PDF.
* **Retrieval:** Precision@K = relevant retrieved / K; Recall@K = relevant retrieved / all relevant; RR = 1 / rank of first relevant; MRR = mean RR. A retrieved chunk counts as relevant if its `(file, page)` is labelled relevant; Recall counts distinct relevant pages.
* **Generation:** human rubric (1-5) for factual correctness, relevance, completeness, grounding, clarity and source accuracy; automatic checks only for abstention and citation format - they do **not** prove correctness.
* **Experiments:** one parameter changed at a time (chunk size, overlap, Top-K) on the same questions; latency, index build time and number of chunks recorded.

---

## Results

All numbers below come from one execution of the notebook on **2026-11-25** (Google Colab, text PDFs, no OCR needed). Nothing was copied from elsewhere.

### Run environment

| Item | Value |
|---|---|
| Hardware | Tesla T4 GPU (15.6 GB), 13.6 GB system RAM |
| PyTorch / Transformers / Sentence-Transformers / FAISS | 2.14.1+cu130 / 5.22.1 / 5.7.0 / 1.16.0 |
| Embedding model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions) |
| Generation model (requested = used) | `Qwen/Qwen2.5-3B-Instruct` (no fallback needed) |
| Baseline parameters | chunk 900 / overlap 150 / Top-K 4, greedy decoding, `max_new_tokens=300` |

### Knowledge base used in this run

* **13 PDF files, 302 pages with text, 482 chunks** (baseline). Index of 482 vectors x 384 dimensions, encoding time 1.6 s.
* Files: five French "Recherche d'Information" course chapters, `Comprendre le RAG...`, `RAG.pdf`, `RAGEXPLANATION.pdf` and `RAG_Detailed.pdf`. **Four of these files were uploaded twice** (Colab renamed the copies `... (1).pdf`), see "Validity of the results" below.
* Extraction: no document flagged for OCR; only 2 empty pages in total (one in each copy of `RAG.pdf`, no images). Cleaning removed 1.3 % of the characters (254,173 to 250,970).
* Chunking comparison on this corpus:

| Chunk size / overlap | #Chunks | Mean length (chars) | Median | Max |
|---|---|---|---|---|
| 600 / 100 | 611 | 461.2 | 598 | 600 |
| **900 / 150 (baseline)** | **482** | 576.6 | 522 | 900 |
| 1200 / 200 | 422 | 651.5 | 486 | 1200 |

* **Embedder truncation:** the embedding model reads at most 128 word-pieces. For the baseline chunks the mean length is 146.2 tokens (max 300) and **54.6 % of the chunks are longer than 128 tokens**, so the end of many chunks is not represented in its vector.
* Index integrity: saved index reloaded with identical search results; the mismatch detector correctly flagged a deliberately broken copy (482 vectors vs 481 metadata entries).

### Retrieval quality (8 answerable questions, baseline 900/150)

Per question at K = 4:

| Question | Precision@4 | Recall@4 | RR | Relevant pages |
|---|---|---|---|---|
| Q1 chunk size / overlap | 0.25 | 0.333 | 0.333 | 3 |
| Q2 FAISS index type | 0.25 | 1.000 | 1.000 | 1 |
| Q3 MRR definition | 0.25 | 0.500 | 1.000 | 2 |
| Q4 Precision@k / Recall@k | 0.50 | 1.000 | 1.000 | 2 |
| Q5 generation model | 0.25 | 0.500 | 0.500 | 2 |
| Q6 empty extraction | 0.25 | 0.500 | 0.333 | 2 |
| Q7 chunks.json order | 0.00 | 0.000 | 0.000 | 4 |
| Q8 privacy / licenses | 0.00 | 0.000 | 0.000 | 2 |

Mean over the 8 questions:

| K | Precision@K | Recall@K | MRR |
|---|---|---|---|
| 1 | 0.375 | 0.250 | 0.375 |
| 2 | 0.250 | 0.312 | 0.438 |
| **4** | **0.219** | **0.479** | **0.521** |
| 6 | 0.188 | 0.583 | 0.521 |

### Experiments (one factor at a time, same 8 questions, K = 4 unless stated)

| Factor | Chunk / overlap | #Chunks | Precision@4 | Recall@4 | MRR | Retrieval (ms) | Index build (s) |
|---|---|---|---|---|---|---|---|
| Baseline | 900 / 150 | 482 | 0.219 | 0.479 | 0.521 | 13.6 | 1.02 |
| Chunk size | 600 / 150 | 633 | 0.250 | 0.510 | 0.562 | 13.6 | 1.34 |
| Chunk size | 1200 / 150 | 420 | 0.219 | 0.479 | 0.521 | 18.6 | 0.94 |
| Overlap | 900 / 0 | 455 | 0.219 | 0.479 | 0.521 | 18.8 | 1.12 |
| Overlap | 900 / 300 | 515 | 0.219 | 0.479 | 0.521 | 14.6 | 1.17 |
| Course variant A (2 factors) | 600 / 100 | 611 | 0.250 | 0.510 | 0.646 | 13.6 | 1.28 |
| Course variant C (2 factors) | 1200 / 200 | 422 | 0.219 | 0.479 | 0.521 | 13.4 | 0.93 |

Top-K sweep on the baseline index:

| Top-K | Precision@K | Recall@K | MRR | Retrieval (ms) |
|---|---|---|---|---|
| 2 | 0.188 | 0.250 | 0.375 | 13.6 |
| 4 | 0.219 | 0.479 | 0.521 | 13.6 |
| 6 | 0.188 | 0.583 | 0.521 | 14.2 |
| 8 | 0.156 | 0.625 | 0.521 | 13.5 |

**Reading the experiments.** Recall rises with K while Precision@K falls, as expected. Smaller chunks (600) gave slightly higher Precision, Recall and MRR than 900, while 1200 and every overlap setting (0, 150, 300) gave the same scores as the baseline. With only 8 questions, one question changes a mean by 12.5 points, so these differences are **not statistically meaningful**. Retrieval latency is around 13-19 ms and is dominated by query encoding, not by the exact FAISS search over 482 vectors.

### Abstention analysis (top-1 cosine similarity)

| Answerable | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 |
|---|---|---|---|---|---|---|---|---|
| top-1 similarity | 0.623 | 0.747 | 0.675 | 0.799 | 0.588 | 0.636 | 0.518 | 0.639 |

| Unanswerable | U1 (capital of Australia) | U2 (MMLU accuracy) | U3 (author of the guide) |
|---|---|---|---|
| top-1 similarity | 0.228 | 0.493 | 0.425 |

On this sample the two groups do not overlap (lowest answerable 0.518, highest unanswerable 0.493), but the margin is small and the sample tiny, so no threshold was activated (`MIN_SIMILARITY = None`).

### Generation results

Demonstration (7 questions, Qwen2.5-3B-Instruct, Top-K 4). Generation time was 2.2-8.2 s per answer; mean over 3 evaluation questions: **7.2 s**; retrieval 12-17 ms (114 ms for one of the unanswerable queries).

| # | Question | Outcome | Observation |
|---|---|---|---|
| 1 | Chunk size and overlap | answered, cited [3][4] | 900 characters / 150 overlap: matches the guide |
| 2 | IndexFlatIP and cosine | answered, cited [1][2] | exact search, inner product = cosine when normalised: matches the guide |
| 3 | How MRR is calculated | answered, cited [1] | correct definition (average of reciprocal ranks, RR = 1/rank, 0 if none) |
| 4 | Generation model and model card | answered, cited [1][2] | Qwen2.5-3B-Instruct; license, languages, memory: matches the guide |
| 5 | Empty extracted text | answered, cited [3] | **incomplete**: said to keep the original PDF; the OCR advice of the guide was not given |
| 6 | Capital of Australia (unanswerable) | **abstained** | correct behaviour |
| 7 | MMLU accuracy (unanswerable) | **abstained** | correct behaviour |

Evaluation set (11 questions, automatic indicators only, heuristic and not proof of correctness):

| Indicator | Result |
|---|---|
| Unanswerable questions answered with an abstention | **2 / 3** (U1, U2 yes; U3 no) |
| Answerable questions not wrongly refused | **6 / 8** (Q7 and Q8 were refused) |
| Answers containing at least one citation | 8 / 11 |
| Invalid source numbers or unretrieved page numbers | **0** in all 11 answers |

Observed failure cases:
* **U3** ("author of the guide"): the model named an author (cited as [1]) although the guide itself states no author. The name probably comes from another uploaded document or a mis-attributed passage, so this is a grounding failure for the question as asked and a case where the strict prompt was not enough.
* **Q7 and Q8**: the system refused to answer; both questions have Recall@4 = 0 under the page labels, i.e. the labelled pages were not among the top-4 chunks (see the validity note).
* **Q6**: partial answer, as shown above.

The 1-5 human rubric (factual correctness, relevance, completeness, grounding, clarity, source accuracy) was **not scored** in this run (`HUMAN_SCORES` empty); the table above is a qualitative reading of the logged answers.

### Validity of the results (please read)

1. **Duplicate and extra documents.** The run indexed 13 files, four of them uploaded twice (`RAG_Detailed.pdf` and `RAG_Detailed (1).pdf`, etc.). Duplicate chunks have identical scores and fill Top-K slots (for Q1, ranks 1-2 and 3-4 are the same two pages from the two copies). The relevance labels only cover `RAG_Detailed.pdf`, so the `(1)` copies and the other documents are counted as irrelevant. **Precision and Recall are therefore underestimated**, and ties between copies are broken arbitrarily: the baseline at K = 2 gives P/R/MRR = 0.250/0.312/0.438 in Step 14 and 0.188/0.250/0.375 in Step 16 for the identical configuration, while K = 4 and K = 6 agree.
2. **Small labelled set**: 8 questions, one annotator, page-level labels, one document family. No significance test is possible.
3. **Embedder truncation**: more than half of the baseline chunks exceed the model's 128-token window, which weakens the comparison between chunk sizes.
4. A cleaner re-run uses only `RAG_Detailed.pdf` (delete the `(1)` copies before the extraction cell) or labels the duplicate pages as relevant too; the numbers above should then be replaced.

## Limitations
Retrieval errors (Q7, Q8), missing information and unanswerable questions (U3 was answered), generation errors (Q6 incomplete), extraction problems (scans need OCR, tables, columns), embedder truncation (128 word-pieces for the baseline model), a small evaluation set labelled by one annotator, duplicate documents in the tested knowledge base, privacy of uploaded documents, and third-party licenses. RAG reduces reliance on unsupported model knowledge but **cannot guarantee correctness**; verify answers against the cited pages.

## Possible improvements
Remove duplicate files automatically (hash PDFs on upload); use an embedding model with a longer input window or sentence/paragraph-aware chunking; add a cross-encoder re-ranker or hybrid BM25 + dense retrieval; activate a similarity threshold calibrated on a larger question set; enlarge the labelled set with a second annotator and complete the human rubric.

## License
Original code: MIT (see `LICENSE`). Third-party libraries, pretrained models and indexed documents keep their own licenses.

## References
* Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020.
* PyMuPDF: https://pymupdf.readthedocs.io/ - Sentence Transformers: https://www.sbert.net/ - FAISS: https://github.com/facebookresearch/faiss
* Hugging Face Transformers: https://huggingface.co/docs/transformers/ - Qwen2.5-3B-Instruct: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
* PyTorch: https://pytorch.org/get-started/locally/ - Course material: *A Practical Guide to RAG* (`RAG_Detailed.pdf`).
