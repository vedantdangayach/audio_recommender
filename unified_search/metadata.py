import pandas as pd

METADATA_PATH = "/Users/vedantdangayach/audio_recommender/metadata.csv"
_metadata_df = pd.read_csv(METADATA_PATH)

def get_metadata(filename):
    """Return metadata dict for a given filename."""
    row = _metadata_df[_metadata_df['filename'] == filename]
    return row.to_dict('records')[0] if not row.empty else {}

def filter_by_type(results, target_type):
    """Filter results by type."""
    return [r for r in results if get_metadata(r['filename']).get('type') == target_type]

def filter_by_duration(results, min_sec=None, max_sec=None):
    """Filter results by duration range (in seconds)."""
    def in_range(r):
        dur = get_metadata(r['filename']).get('duration')
        return (min_sec is None or dur >= min_sec) and (max_sec is None or dur <= max_sec)
    return [r for r in results if in_range(r)]

def apply_filters(results, filters):
    """Chain filters based on filters dict."""
    if not filters:
        return results
    if 'type' in filters:
        results = filter_by_type(results, filters['type'])
    if 'min_duration' in filters or 'max_duration' in filters:
        results = filter_by_duration(results, filters.get('min_duration'), filters.get('max_duration'))
    return results
