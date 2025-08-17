import os

OPENAI_API_KEY = os.environ.get("YOUR_OPENAI_API_KEY")

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

