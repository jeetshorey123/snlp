Match unresolved queries to resolved queries using fuzzy matching and TF-IDF.

Files:
- `match_queries.py`: script that evaluates fuzzy methods (ratio, partial_ratio, token_sort_ratio, token_set_ratio) and TF-IDF+cosine similarity. It picks best thresholds by evaluating against the ground-truth mapping included in `new_queries.csv` and writes CSV outputs with predictions.
- `requirements.txt`: Python dependencies.

How to run (PowerShell):

python -m pip install -r requirements.txt; python match_queries.py

Outputs:
- `new_queries_with_fuzzy_matches.csv`
- `new_queries_with_tfidf_matches.csv`
- `match_comparison_examples.csv`
