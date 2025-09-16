# backend/app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional
from utils_extract import extract_text_from_file
from utils_nlp import analyze_resume_text


app = FastAPI(title="AI Resume Analyzer API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/analyze')
async def analyze(
    resume: UploadFile = File(...),
    job_description: Optional[UploadFile] = None,
    use_gpt: bool = Form(False)
):
    """Accepts a resume file and an optional job description file. Returns analysis JSON."""
    try:
        resume_bytes = await resume.read()
        resume_text = extract_text_from_file(resume.filename, resume_bytes)

        jd_text = None
        if job_description is not None:
            jd_bytes = await job_description.read()
            jd_text = extract_text_from_file(job_description.filename, jd_bytes)

        result = analyze_resume_text(resume_text, jd_text, use_gpt=use_gpt)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)