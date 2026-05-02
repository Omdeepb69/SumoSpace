def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a string using a 4-chars-per-token heuristic.
    Includes a 20% safety margin to prevent accidental overflow.
    """
    if not text:
        return 0
    
    char_count = len(text)
    # 4 chars per token + 20% safety
    estimated = int((char_count / 4) * 1.2)
    return max(1, estimated)

def truncate_by_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget (approximate)."""
    if not text or max_tokens <= 0:
        return ""
        
    estimated_current = estimate_tokens(text)
    if estimated_current <= max_tokens:
        return text
        
    # Heuristic: 4 chars per token (conservative)
    # We want max_tokens * 4 * 0.8 to be safe
    max_chars = int(max_tokens * 4 * 0.8)
    return text[:max_chars]
