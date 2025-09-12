# backend/utils_extract.py
import io
import pdfplumber
import docx
import re




def extract_text_from_file(filename: str, file_bytes: bytes) -> str:
"""Detects type from filename and extracts text. Returns a cleaned plain-text string."""
lower = filename.lower()
if lower.endswith('.pdf'):
return extract_text_from_pdf_bytes(file_bytes)
elif lower.endswith('.docx') or lower.endswith('.doc'):
return extract_text_from_docx_bytes(file_bytes)
else:
# assume plain text
return file_bytes.decode('utf-8', errors='ignore')




def extract_text_from_pdf_bytes(b: bytes) -> str:
out = []
with pdfplumber.open(io.BytesIO(b)) as pdf:
for page in pdf.pages:
text = page.extract_text()
if text:
out.append(text)
return clean_text('\n'.join(out))




def extract_text_from_docx_bytes(b: bytes) -> str:
with io.BytesIO(b) as bio:
doc = docx.Document(bio)
paragraphs = [p.text for p in doc.paragraphs]
return clean_text('\n'.join(paragraphs))




def clean_text(s: str) -> str:
s = s.replace('\r', '\n')
# collapse multiple newlines
s = re.sub(r'\n{2,}', '\n', s)
# strip leading/trailing spaces each line
s = '\n'.join([line.strip() for line in s.splitlines() if line.strip()])
return s