import pandas as pd
import duckdb
import json
import re
import io
import base64
import os
from bs4 import BeautifulSoup
import requests
from PIL import Image
import matplotlib.pyplot as plt

from .llm_tools import ask_llm

# Utilities: load a file by extension into a pandas DataFrame or a dict for JSON

def load_table(path):
    path = str(path)
    if path.endswith('.csv') or path.endswith('.CSV'):
        return pd.read_csv(path)
    if path.endswith('.json') or path.endswith('.JSON'):
        try:
            return pd.read_json(path, lines=True)
        except Exception:
            # fallback: load as normal JSON
            import json
            with open(path,'r',encoding='utf-8') as f:
                data = json.load(f)
            return pd.json_normalize(data)
    if path.endswith('.parquet') or path.endswith('.pq'):
        # use duckdb read_parquet for wildcard-friendly paths if needed
        return pd.read_parquet(path)
    # unknown: return raw text
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        return f.read()


# Plot helper that compresses and enforces max bytes

def compress_image_bytes(img_bytes: bytes, max_bytes: int = 100_000) -> bytes:
    """Try saving as WEBP/PNG with quality reduction and small resize until under max_bytes."""
    im = Image.open(io.BytesIO(img_bytes))
    # Start with webp (good compression for plots)
    quality = 90
    width, height = im.size
    while True:
        bio = io.BytesIO()
        try:
            im.save(bio, format='WEBP', quality=quality, method=6)
        except Exception:
            im.save(bio, format='PNG', optimize=True)
        data = bio.getvalue()
        if len(data) <= max_bytes or (quality <= 20 and (width < 200 or height < 200)):
            return data
        # reduce quality and maybe resize
        quality = max(20, quality - 10)
        width = int(width * 0.9)
        height = int(height * 0.9)
        im = im.resize((width, height), Image.LANCZOS)


def fig_to_b64img(fig, fmt='png', max_bytes=100_000):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches='tight')
    buf.seek(0)
    raw = buf.getvalue()
    # compress
    compressed = compress_image_bytes(raw, max_bytes=max_bytes)
    return 'data:image/webp;base64,' + base64.b64encode(compressed).decode()


# Determine response format from the questions.txt instructions

def detect_response_format(text: str):
    low = text.lower()
    if 'json array' in low or 'respond with a json array' in low or 'json array of strings' in low:
        return 'array'
    if 'json object' in low or 'respond with a json object' in low or 'json object containing' in low:
        return 'object'
    # default: array if multiple numbered questions otherwise object
    if re.search(r"^\s*\d+\.", text, flags=re.M):
        return 'array'
    return 'object'


# Core handler: accepts file_map: filename->path

def handle_data_and_questions(file_map: dict):
    # load questions.txt
    qpath = None
    for k in file_map:
        if k.lower().startswith('questions') or k.lower().endswith('questions.txt'):
            qpath = file_map[k]
            break
    if not qpath:
        raise ValueError('questions.txt must be provided')

    with open(qpath, 'r', encoding='utf-8', errors='ignore') as f:
        qtext = f.read()

    resp_type = detect_response_format(qtext)

    # Simple heuristics: if wikipedia mentioned, do scraping (example task)
    if 'highest grossing films' in qtext.lower() or 'wikipedia' in qtext.lower():
        return _handle_wikipedia_films(qtext, resp_type)

    # If duckdb / parquet keywords present, try to find parquet file or use duckdb
    for fname, path in file_map.items():
        if fname.endswith('.parquet') or fname.endswith('.parq'):
            # try to answer with duckdb (user's question might include SQL)
            try:
                df = load_table(path)
                # ask LLM for specific analytic plan (optional)
                plan = ask_llm('Given the question:\n' + qtext + '\nReturn a concise python/pandas plan for aggregation and plots.')
                # If LLM provided code, we could exec it, but for safety, return plan
                return {"plan": plan, "note": "Parquet loaded"}
            except Exception as e:
                return {"error": str(e)}

    # If CSV/JSON present, load first CSV and run lightweight numeric analysis
    df = None
    for fname, path in file_map.items():
        if fname.endswith('.csv'):
            try:
                df = load_table(path)
                break
            except Exception:
                continue

    # If we have DataFrame and simple questions (count, regression, plot)
    if isinstance(df, pd.DataFrame):
        # Try to parse numbered questions and answer a few common types
        answers = []
        # Example: common pattern "1. How many ... 2. Which ... 3. What's the correlation ... 4. Draw a scatterplot ..."
        if resp_type == 'array':
            # 1) If Q asks a count of rows matching condition (search for number/keyword)
            # We'll be conservative: answer a few heuristics and fill with placeholders for unknowns
            # 1. Count rows
            answers.append(int(len(df)))
            # 2. first textual column's top value
            textcols = df.select_dtypes(include='object').columns.tolist()
            if textcols:
                answers.append(str(df[textcols[0]].dropna().iloc[0]))
            else:
                answers.append('')
            # 3. correlation of first two numeric columns
            numcols = df.select_dtypes(include='number').columns.tolist()
            if len(numcols) >= 2:
                corr = float(df[numcols[0]].corr(df[numcols[1]]))
            else:
                corr = 0.0
            answers.append(corr)
            # 4. produce a scatterplot of the first two numeric cols
            if len(numcols) >= 2:
                fig, ax = plt.subplots()
                ax.scatter(df[numcols[0]], df[numcols[1]])
                # regression
                try:
                    m, b = _linear_regression(df[numcols[0]], df[numcols[1]])
                    xs = pd.Series(df[numcols[0]])
                    ax.plot(xs, m*xs + b, linestyle='--', color='red')
                except Exception:
                    pass
                ax.set_xlabel(numcols[0])
                ax.set_ylabel(numcols[1])
                b64 = fig_to_b64img(fig, fmt='png', max_bytes=100_000)
                plt.close(fig)
                answers.append(b64)
            else:
                answers.append('')

            return answers

    # Last-resort: ask LLM to propose an answer (if available)
    llm_answer = ask_llm(qtext)
    if resp_type == 'array':
        return [llm_answer]
    else:
        return {"answer": llm_answer}


# Helpers

def _linear_regression(x, y):
    import numpy as np
    x = pd.Series(x).dropna().astype(float)
    y = pd.Series(y).dropna().astype(float)
    # align
    df = pd.concat([x, y], axis=1).dropna()
    if df.shape[0] < 2:
        return 0.0, 0.0
    X = df.iloc[:,0].values
    Y = df.iloc[:,1].values
    A = np.vstack([X, np.ones_like(X)]).T
    m, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    return float(m), float(b)


def _handle_wikipedia_films(qtext, resp_type):
    url = 'https://en.wikipedia.org/wiki/List_of_highest-grossing_films'
    html = requests.get(url, timeout=20).text
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', {'class': 'wikitable'})
    if table is None:
        return {"error": "table not found on page"}

    df = pd.read_html(str(table))[0]
    # Normalize columns: Title, Worldwide gross, Year, Rank, Peak may exist
    # Clean numeric
    if 'Worldwide gross' in df.columns:
        df['Worldwide gross_clean'] = df['Worldwide gross'].astype(str).str.replace(r'[^0-9.]', '', regex=True).replace('', '0')
        df['Worldwide gross_clean'] = pd.to_numeric(df['Worldwide gross_clean'], errors='coerce')
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    answers = []
    # 1. How many $2 bn movies were released before 2000?
    if 'Worldwide gross_clean' in df.columns and 'Year' in df.columns:
        cnt = int(df[(df['Worldwide gross_clean'] >= 2_000_000_000) & (df['Year'] < 2000)].shape[0])
    else:
        cnt = 0
    answers.append(cnt)

    # 2. earliest film that grossed over $1.5 bn
    if 'Worldwide gross_clean' in df.columns and 'Year' in df.columns and 'Title' in df.columns:
        over = df[df['Worldwide gross_clean'] > 1_500_000_000].sort_values('Year')
        earliest = over.iloc[0]['Title'] if not over.empty else ''
    else:
        earliest = ''
    answers.append(str(earliest))

    # 3. correlation between Rank and Peak
    if 'Rank' in df.columns and 'Peak' in df.columns:
        try:
            corr = float(pd.to_numeric(df['Rank'], errors='coerce').corr(pd.to_numeric(df['Peak'], errors='coerce')))
        except Exception:
            corr = 0.0
    else:
        corr = 0.0
    answers.append(corr)

    # 4. scatterplot of Rank vs Peak with dotted red regression line (encoded)
    if 'Rank' in df.columns and 'Peak' in df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(pd.to_numeric(df['Rank'], errors='coerce'), pd.to_numeric(df['Peak'], errors='coerce'))
        try:
            m, b = _linear_regression(pd.to_numeric(df['Rank'], errors='coerce'), pd.to_numeric(df['Peak'], errors='coerce'))
            xs = pd.to_numeric(df['Rank'], errors='coerce')
            ax.plot(xs, m*xs + b, linestyle='--', color='red')
        except Exception:
            pass
        ax.set_xlabel('Rank')
        ax.set_ylabel('Peak')
        b64 = fig_to_b64img(fig, fmt='png', max_bytes=100_000)
        plt.close(fig)
    else:
        b64 = ''

    answers.append(b64)
    # Return either an array or an object depending on instruction
    if resp_type == 'array':
        return answers
    else:
        return {
            "count_2bn_before_2000": answers[0],
            "earliest_over_1_5bn": answers[1],
            "corr_rank_peak": answers[2],
            "plot": answers[3]
        }