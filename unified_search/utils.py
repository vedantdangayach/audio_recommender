import numpy as np

def normalize_scores(results):
    # Assume each result has a 'score' key
    scores = np.array([r['score'] for r in results])
    if len(scores) == 0:
        return results
    min_score, max_score = scores.min(), scores.max()
    for r in results:
        if max_score > min_score:
            r['score'] = (r['score'] - min_score) / (max_score - min_score)
        else:
            r['score'] = 1.0  # all same
    return results

def format_results(results, engine, top_k):
    formatted = []
    for r in sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]:
        formatted.append({
            "filename": r['filename'],
            "score": round(r['score'], 3),
            "type": r.get('type', None),
            "source_engine": engine
        })
    return formatted
