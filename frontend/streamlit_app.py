import streamlit as st
import requests
import base64

API_URL = "http://localhost:8000"  # Change to your deployed FastAPI backend URL

st.set_page_config(page_title='AI Resume Analyzer', layout='centered')
st.title('📄 AI Resume Analyzer')
st.markdown(
    "Upload your resume (PDF/DOCX/TXT) and optionally a job description. "
    "The app will analyze it, estimate an ATS-style score, and suggest improvements."
)

# Upload form
with st.form('analyze_form'):
    resume_file = st.file_uploader(
        'Upload Resume', 
        type=['pdf', 'docx', 'txt'], 
        help='Upload your resume file'
    )
    jd_file = st.file_uploader(
        'Upload Job Description (optional)', 
        type=['pdf', 'docx', 'txt'], 
        help='Upload job posting or description'
    )
    use_gpt = st.checkbox(
        'Use GPT for advanced rewrite suggestions (requires OpenAI API key on backend)', 
        value=False
    )
    submitted = st.form_submit_button('Analyze Resume')

# Handle form submission
if submitted:
    if not resume_file:
        st.error("⚠️ Please upload a resume file first.")
    else:
        with st.spinner("Analyzing resume..."):
            files = {
                "resume": (resume_file.name, resume_file.getvalue())
            }
            if jd_file:
                files["job_description"] = (jd_file.name, jd_file.getvalue())

            data = {"use_gpt": str(use_gpt)}

            try:
                resp = requests.post(f"{API_URL}/analyze", files=files, data=data, timeout=60)
                if resp.status_code == 200:
                    result = resp.json()

                    # ✅ ATS Score
                    st.success(f"✅ ATS Score: {result.get('score', 'N/A')}%")

                    # Keywords
                    st.subheader("🔑 Keywords")
                    st.write("**Resume Keywords:**", result.get("resume_keywords", []))
                    st.write("**Job Description Keywords:**", result.get("jd_keywords", []))

                    # Coverage
                    st.subheader("📊 Keyword Coverage & Sections")
                    st.write(f"Coverage: {result.get('coverage', 0) * 100:.1f}%")
                    st.json(result.get("section_presence", {}))

                    # Bullets
                    st.subheader("📌 Top Resume Bullets")
                    for b in result.get("top_bullets", []):
                        st.write("-", b)

                    # Suggestions
                    st.subheader("✍️ Rewrite Suggestions")
                    for r in result.get("rewrites", []):
                        st.markdown(f"**Original:** {r.get('original')}")
                        st.markdown(f"**Suggested:** {r.get('suggested')}")
                        if r.get("suggested_gpt"):
                            st.markdown(f"**GPT Suggestion:** {r.get('suggested_gpt')}")
                        st.divider()

                    # Other metrics
                    st.subheader("📈 Other Metrics")
                    st.write("Average sentence length:", result.get("avg_sentence_length"))
                    st.write("Embedding similarity:", result.get("embedding_similarity"))

                    # Downloadable JSON report
                    import json
                    summary_json = json.dumps(result, indent=2)
                    b64 = base64.b64encode(summary_json.encode()).decode()
                    href = f"data:application/json;base64,{b64}"
                    st.markdown(f"[⬇️ Download Analysis JSON]({href})", unsafe_allow_html=True)

                else:
                    st.error(f"❌ Backend error: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"🚨 Failed to connect to backend: {e}")
