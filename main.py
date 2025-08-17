from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import tempfile, os, asyncio
from utils.file_utils import save_uploads_to_tmp
from utils.analysis import handle_data_and_questions

app = FastAPI(title="Data Analyst Agent")

@app.post("/api/")
async def analyze_api(files: list[UploadFile] = File(...)):
    # Save uploads to temporary files and produce a filename->path map
    file_map = await save_uploads_to_tmp(files)

    # Run analysis synchronously but with asyncio timeout guard
    try:
        loop = asyncio.get_event_loop()
        # 170 seconds to leave margin under the 3-minute / 5-minute requirements
        result = await asyncio.wait_for(loop.run_in_executor(None, handle_data_and_questions, file_map), timeout=170)
    except asyncio.TimeoutError:
        return JSONResponse({"error": "analysis timed out"}, status_code=504)
    except Exception as e:
        return JSONResponse({"error": str(e)})

    # result should already be JSON-serializable (list or dict); return as-is
    return JSONResponse(result)