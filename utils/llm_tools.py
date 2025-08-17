import os

OPENAI_API_KEY = os.environ.get("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")

def ask_llm(prompt: str) -> str:
    """Ask OpenAI if key is present, otherwise return a simple heuristic response."""
    if not OPENAI_API_KEY:
        # Basic deterministic fallback — do not call external APIs
        return "LLM not configured. Fallback: try parsing with pandas/duckdb. Prompt was: " + (prompt[:300])

    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"LLM call failed: {e}"
