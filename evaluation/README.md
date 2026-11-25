# evaluation/

* `eval_questions.json` - evaluation questions with reference answers and **page-level** relevance labels
  for the course guide `RAG_Detailed.pdf` (8 answerable + 3 unanswerable questions).
  Labels must be verified manually against the actual PDF; replace them when you use other documents.
* `results/` - put exported results of *your own runs* here (e.g. `experiments.csv`, `generation_review.csv`,
  `run_summary.json`, downloaded from Colab's `artifacts/` folder). Nothing is pre-filled.
