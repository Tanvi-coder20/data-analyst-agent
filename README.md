
"""
Data Analyst Agent (PyCharm)

This repo implements a FastAPI endpoint that accepts a POST with questions.txt and optional files
and returns JSON answers and base64-encoded plot images. It includes:
 - CSV / JSON / Parquet loading (via pandas / duckdb)
 - HTML scraping example (Wikipedia)
 - Plot generation and compression (guarantees under-size images using Pillow + resizing/quality)
 - Optional OpenAI integration (use env OPENAI_API_KEY)

Run locally (PyCharm terminal):
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8000

Test with curl:
  curl -X POST "http://127.0.0.1:8000/api/" -F "questions.txt=@sample_questions.txt" -F "data.csv=@sample.csv"

Deployment:
 - Use ngrok for quick public URL: ngrok http 8000
 - Or deploy to Render / Railway / Fly

License: MIT
"""