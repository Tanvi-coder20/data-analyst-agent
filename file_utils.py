import tempfile
import shutil
from typing import List, Dict
from fastapi import UploadFile

async def save_uploads_to_tmp(files: List[UploadFile]) -> Dict[str, str]:
    file_map = {}
    for f in files:
        suffix = "" if "." not in f.filename else f".{f.filename.split('.')[-1]}"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        contents = await f.read()
        tmp.write(contents)
        tmp.flush()
        tmp.close()
        file_map[f.filename] = tmp.name
    return file_map
