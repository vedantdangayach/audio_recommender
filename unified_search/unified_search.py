from .input_router import route_input
from .metadata import apply_filters
from .utils import normalize_scores, format_results
from .config import DEFAULT_TOP_K

def search_audio(input, top_k=DEFAULT_TOP_K, filters=None):
    try:
        # Validate input
        if not isinstance(input, str):
            return {"error": "Input must be a string (text or file path)"}
        if isinstance(top_k, int) and top_k <= 0:
            return {"error": "top_k must be a positive integer"}
        engine, results = route_input(input, top_k=top_k)

        # Check for error dicts
        if results and isinstance(results, list) and "error" in results[0]:
            return results  # or return results[0] if you want a single error dict

        # Apply filters
        filtered = apply_filters(results, filters)
        # Normalize scores
        normalized = normalize_scores(filtered)
        # Format results
        formatted = format_results(normalized, engine, top_k)
        return formatted
    except FileNotFoundError as e:
        return {"error": f"File not found: {e.filename}"}
    except Exception as e:
        return {"error": str(e)}
