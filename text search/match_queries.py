import re
import csv
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # normalize unicode quotes
    text = text.replace("\u2019", "'").replace('"', ' ').replace('\u201c', ' ').replace('\u201d', ' ')
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']+", ' ', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text


def evaluate_fuzzy(resolved, new, scorer, thresholds=range(50,101)):
    # resolved: list of (id, text)
    resolved_texts = [t for (_id, t) in resolved]
    resolved_ids = [ _id for (_id, t) in resolved]

    best = {'threshold': None, 'accuracy': -1, 'preds': None}

    # pre-build choices mapping for process.extract
    choices = {t: _id for _id, t in resolved}

    for thr in thresholds:
        preds = []
        for q in new['proc_query']:
            match = process.extractOne(q, choices, scorer=scorer)
            if match is None:
                preds.append(None)
            else:
                matched_text, score, _ = match
                if score >= thr:
                    preds.append(choices[matched_text])
                else:
                    preds.append(None)
        # compare to ground truth
        true = new['Matches_With_Query_ID'].fillna(-1).astype(int).tolist()
        pred = [(-1 if p is None else int(p)) for p in preds]
        acc = sum(1 for a,b in zip(true, pred) if a==b) / len(true)
        if acc > best['accuracy']:
            best.update({'threshold': thr, 'accuracy': acc, 'preds': pred})
    return best


def evaluate_tfidf(resolved, new, thresholds=np.arange(0.1, 1.01, 0.01)):
    resolved_texts = [t for (_id, t) in resolved]
    resolved_ids = [ _id for (_id, t) in resolved]

    vec = TfidfVectorizer().fit(resolved_texts + new['proc_query'].tolist())
    R = vec.transform(resolved_texts)
    Q = vec.transform(new['proc_query'].tolist())

    sims = cosine_similarity(Q, R)

    best = {'threshold': None, 'accuracy': -1, 'preds': None}
    for thr in thresholds:
        preds = []
        for i in range(sims.shape[0]):
            row = sims[i]
            j = row.argmax()
            score = row[j]
            if score >= thr:
                preds.append(resolved_ids[j])
            else:
                preds.append(None)
        true = new['Matches_With_Query_ID'].fillna(-1).astype(int).tolist()
        pred = [(-1 if p is None else int(p)) for p in preds]
        acc = sum(1 for a,b in zip(true, pred) if a==b) / len(true)
        if acc > best['accuracy']:
            best.update({'threshold': thr, 'accuracy': acc, 'preds': pred})
    return best, sims


def main():
    resolved_df = pd.read_csv('resolved_queries.csv')
    new_df = pd.read_csv('new_queries.csv')

    # preprocess texts
    resolved_df['proc'] = resolved_df['Pre_Resolved_Query'].apply(preprocess)
    new_df['proc_query'] = new_df['Variation_Query'].apply(preprocess)

    resolved = list(zip(resolved_df['Query_ID'].astype(str), resolved_df['proc']))

    # Evaluate fuzzy methods
    methods = {
        'ratio': fuzz.ratio,
        'partial_ratio': fuzz.partial_ratio,
        'token_sort_ratio': fuzz.token_sort_ratio,
        'token_set_ratio': fuzz.token_set_ratio
    }

    fuzzy_results = {}
    for name, scorer in methods.items():
        print(f"Evaluating fuzzy method: {name}")
        best = evaluate_fuzzy(resolved, new_df, scorer)
        fuzzy_results[name] = best
        print(f"  Best threshold: {best['threshold']}, accuracy: {best['accuracy']:.3f}")

    # pick best fuzzy
    best_name = max(fuzzy_results.items(), key=lambda kv: kv[1]['accuracy'])[0]
    best_info = fuzzy_results[best_name]
    print(f"\nBest fuzzy method: {best_name} with threshold {best_info['threshold']} acc={best_info['accuracy']:.3f}\n")

    # Save fuzzy matches
    new_df['fuzzy_pred'] = best_info['preds']
    new_df.to_csv('new_queries_with_fuzzy_matches.csv', index=False)

    # TF-IDF evaluation
    print('Evaluating TF-IDF + cosine similarity...')
    tf_best, sims = evaluate_tfidf(resolved, new_df)
    print(f"  Best TF-IDF threshold: {tf_best['threshold']:.2f}, accuracy: {tf_best['accuracy']:.3f}")

    new_df['tfidf_pred'] = tf_best['preds']
    # also include best cosine score for each query
    best_scores = [sims[i].max() for i in range(sims.shape[0])]
    new_df['tfidf_best_score'] = best_scores

    new_df.to_csv('new_queries_with_tfidf_matches.csv', index=False)

    # Summary
    print('\nSummary:')
    print('Fuzzy best method:', best_name, 'threshold:', best_info['threshold'], 'accuracy:', best_info['accuracy'])
    print('TF-IDF best threshold:', tf_best['threshold'], 'accuracy:', tf_best['accuracy'])

    # show some example matches
    out = []
    for i, row in new_df.iterrows():
        out.append({
            'query': row['Variation_Query'],
            'ground_truth': row['Matches_With_Query_ID'],
            'fuzzy_pred': row['fuzzy_pred'],
            'tfidf_pred': row['tfidf_pred'],
            'tfidf_score': row['tfidf_best_score']
        })
    out_df = pd.DataFrame(out)
    out_df.to_csv('match_comparison_examples.csv', index=False)
    print('\nWrote outputs: new_queries_with_fuzzy_matches.csv, new_queries_with_tfidf_matches.csv, match_comparison_examples.csv')

if __name__ == '__main__':
    main()
