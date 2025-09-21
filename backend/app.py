# backend/app.py
import logging
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional, Dict, Any
from utils_extract import extract_text_from_file
from utils_nlp import analyze_resume_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("AIResumeAnalyzer")

app = FastAPI(title="AI Resume Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def sanitize_filename(filename: str) -> str:
    """Sanitize the filename to prevent directory traversal and other attacks."""
    return os.path.basename(filename)

def allowed_file(filename: str) -> bool:
    """Allow only certain file extensions."""
    allowed_extensions = {"pdf", "doc", "docx", "txt"}
    ext = filename.rsplit('.', 1)[-1].lower()
    return ext in allowed_extensions

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

@app.post('/analyze')
async def analyze(
    resume: UploadFile = File(...),
    job_description: Optional[UploadFile] = None,
    use_gpt: bool = Form(False)
) -> JSONResponse:
    """
    Accepts a resume file and an optional job description file. Returns analysis JSON.
    Args:
        resume (UploadFile): The resume file to analyze.
        job_description (Optional[UploadFile]): Optional job description file.
        use_gpt (bool): Whether to use GPT-based analysis.
    Returns:
        JSONResponse: Analysis result or error message.
    """
    try:
        # Validate file type
        if not allowed_file(resume.filename):
            logger.warning(f"Rejected resume file type: {resume.filename}")
            raise HTTPException(status_code=400, detail="Unsupported resume file type.")
        resume_filename = sanitize_filename(resume.filename)
        resume_bytes = await resume.read()
        resume_text = extract_text_from_file(resume_filename, resume_bytes)

        jd_text = None
        if job_description is not None:
            if not allowed_file(job_description.filename):
                logger.warning(f"Rejected job description file type: {job_description.filename}")
                raise HTTPException(status_code=400, detail="Unsupported job description file type.")
            jd_filename = sanitize_filename(job_description.filename)
            jd_bytes = await job_description.read()
            jd_text = extract_text_from_file(jd_filename, jd_bytes)

        logger.info(f"Analyzing resume: {resume_filename} | Job description: {job_description.filename if job_description else 'None'} | use_gpt: {use_gpt}")
        result: Dict[str, Any] = analyze_resume_text(resume_text, jd_text, use_gpt=use_gpt)
        return JSONResponse(content={"success": True, "result": result})
    except HTTPException as he:
        logger.error(f"HTTPException: {he.detail}")
        return JSONResponse(status_code=he.status_code, content={"success": False, "error": he.detail})
    except Exception as e:
        logger.exception("Unhandled exception during analysis")
        return JSONResponse(status_code=500, content={"success": False, "error": "Internal server error."})

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)