import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os
from dotenv import load_dotenv
from streamlit_lottie import st_lottie
import json

# --- Theme & Configuration ---
# Color palette
PRIMARY_COLOR = "#4361EE"
SECONDARY_COLOR = "#3A0CA3"
ACCENT_COLOR = "#F72585"
BG_COLOR = "#F8F9FA"
TEXT_COLOR = "#212529"


# --- Helper Functions ---
def load_dotenv_config():
    """Load environment variables and configure APIs."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        return True
    return False


def get_gemini_response(input_prompt, pdf_content, job_desc):
    """Generates content using the Gemini model."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([input_prompt, pdf_content, job_desc])
        if response.prompt_feedback.block_reason:
            st.error(f"🚫 Content generation blocked: {response.prompt_feedback.block_reason_message}")
            return None
        return response.text
    except Exception as e:
        st.error(f"🔴 Error interacting with Gemini API: {e}")
        return None


def extract_pdf_text(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    if uploaded_file is not None:
        try:
            document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text_parts = [page.get_text() for page in document]
            return " ".join(text_parts)
        except Exception as e:
            st.error(f"🔴 Error processing PDF file: {e}")
            return None
    return None


def speak_text(text):
    """Generates audio from text using gTTS and saves to a temporary file."""
    if not text:
        return None
    try:
        tts = gTTS(text=text, lang='en')
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)
        audio_path = temp_audio.name
        temp_audio.close()
        return audio_path
    except Exception as e:
        st.error(f"🔴 Error generating audio: {e}")
        return None


def load_lottie_animation(filepath):
    """Loads a Lottie animation JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"🔴 Error loading animation: {e}")
        return None


def cleanup_temp_file(file_path):
    """Safely delete temporary files."""
    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except Exception:
            pass


# --- Prompts Dictionary ---
def get_analysis_prompts(custom_query=""):
    return {
        "resume_review": """
You are an experienced Hiring Manager and Resume Expert.
Analyze the provided resume against the job description.
Evaluate the following:
1. *Strengths & Weaknesses:* Highlight key qualifications and areas for improvement relevant to the job.
2. *Content & Impact:* Assess the clarity, conciseness, and impact of the language used. Does it quantify achievements?
3. *Structure & Formatting:* Comment on the layout, readability, and overall organization. Is it professional?
4. *Grammar & Professionalism:* Check for grammatical errors or unprofessional language.
Provide a concise professional summary based on this evaluation.
""",
        "skill_improvement": """
You are a Technical Recruiter and Career Advisor.
Based on the resume and the target job description:
1. Identify key skill gaps (both hard and soft skills).
2. Suggest specific areas for skill improvement.
3. Recommend relevant tools, programming languages, or platforms the candidate should consider learning or highlighting.
""",
        "missing_keywords": """
You are an expert ATS (Applicant Tracking System) scanner.
Compare the resume against the job description.
List the top 10-15 most important keywords and phrases from the job description that are MISSING or underrepresented in the resume. Focus on skills, technologies, and qualifications mentioned in the job description.
""",
        "match_ats": """
You are an advanced ATS simulator and Resume Analyst.
Carefully compare the resume against the job description and provide the following:
Before analysis, determine if the resume is for an intern or entry-level candidate. If yes, apply lenient scoring criteria based on early-career expectations (i.e., fewer years of experience, partial skills, learning potential). Interns can still have a high job match if they meet some key criteria.
Then, provide:
1. Specify about the job position suitability based on job match percentage
2. Job Match Percentage: Estimate the compatibility score (0-100%) based on skills, experience, and keywords. For intern or entry-level roles, consider learning potential and partial matches more positively. Explain your reasoning briefly.

3. ATS Friendliness: Determine if the resume is ATS-friendly. Consider:
   * Standard section headings (Experience, Education, Skills)
   * Clear and readable formatting (avoiding tables, columns, excessive graphics)
   * Standard font usage
   * Presence of keywords
   State clearly: "ATS-Friendly: Yes" or "ATS-Friendly: No" and briefly explain why.

4. Key Missing Keywords: List 5-7 crucial keywords from the job description missing in the resume.

5. Final Suggestions: Provide 2-3 concise, actionable tips for improvement based on the analysis.
        
""",
        "market_insights": """
You are a Market Research Analyst specializing in HR and compensation.
Based only on the provided Job Description:
1. *Potential Salary Range:* Estimate a likely salary range for this type of role in a general market (mention it's an estimate for india).
2. *Key Skills in Demand:* Identify 3-5 key skills mentioned in the job description that are currently high in demand in the industry.
3. *Potential Career Trajectory:* Briefly suggest 1-2 potential next steps or career growth opportunities typically following this role.
Disclaimer: These are general insights based on the job description text and not real-time, location-specific market data.
""",
        "career_path": """
Act as a Career Coach. Based on the skills and experience outlined in the resume:
1. Suggest 3-5 potential alternative or future career paths that align well with the candidate's profile.
2. For each path, briefly explain why it's a suitable suggestion, linking it to specific skills or experiences in the resume.
Consider both vertical and lateral moves.
""",
        "upskilling": """
You are a Learning and Development Advisor.
Based on the resume and the target job description:
1. Identify 2-3 key skill gaps or areas for development.
2. For each gap, suggest specific types of courses, certifications, or learning resources.
3. Mention relevant online learning platforms (like Coursera, Udemy, edX, LinkedIn Learning, Pluralsight, etc.) where such courses might be found. Provide specific course name examples if possible.
""",
        "linkedin": """
You are a LinkedIn Profile Optimization Expert and Professional Branding Coach.
Based on the provided resume and targeting the job description, write a compelling and engaging LinkedIn 'About' section (summary) for the candidate.
The summary should:
1. Be concise (around 3-5 short paragraphs).
2. Say how can the linkedIn profile can be optimized
3. Highlight key skills, experiences, and achievements relevant to the target role.
4. Be written in the first person.
5. Be professional and engaging.
6. Include relevant keywords for discoverability.
""",
        "custom": custom_query
    }


# --- Main Application ---
def main():
    # Page Configuration
    st.set_page_config(
        page_title="JobFit AI",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Check API Key Configuration
    if not load_dotenv_config():
        st.error("🔴 GEMINI_API_KEY not found. Please set it in your environment variables or .env file.")
        st.stop()

    # Custom CSS
    st.markdown(f"""
    <style>
        :root {{
            --primary-color: {PRIMARY_COLOR};
            --secondary-color: {SECONDARY_COLOR};
            --accent-color: {ACCENT_COLOR};
            --background-color: {BG_COLOR};
            --text-color: {TEXT_COLOR};
        }}

        .stApp {{
            background-color: var(--background-color);
            color: var(--text-color);
        }}

        h1, h2, h3 {{
            color: var(--primary-color);
        }}

        .step-container {{
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }}

        .step-active {{
            background-color: rgba(67, 97, 238, 0.1);
            border-left: 5px solid var(--primary-color);
        }}

        .step-completed {{
            opacity: 0.7;
        }}

        .step-heading {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .upload-area {{
            border: 2px dashed rgba(67, 97, 238, 0.3);
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
        }}

        .upload-area:hover {{
            border-color: var(--primary-color);
        }}

        .result-card {{
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }}

        .audio-player {{
            background-color: rgba(67, 97, 238, 0.05);
            border-radius: 10px;
            padding: 1rem;
        }}

        /* Button styling */
        .stButton button {{
            background-color: var(--primary-color);
            color: white;
            border-radius: 5px;
            transition: all 0.2s ease;
        }}

        .stButton button:hover {{
            background-color: var(--secondary-color);
            transform: translateY(-2px);
        }}

        /* Mobile responsiveness */
        @media screen and (max-width: 640px) {{
            .responsive-cols {{
                flex-direction: column;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1

    if 'resume_content' not in st.session_state:
        st.session_state.resume_content = None

    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""

    if 'analysis_option' not in st.session_state:
        st.session_state.analysis_option = None

    if 'custom_query' not in st.session_state:
        st.session_state.custom_query = ""

    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None

    if 'audio_path' not in st.session_state:
        st.session_state.audio_path = None

    # Navigation functions
    def go_to_step(step):
        st.session_state.step = step

    # Load animations
    header_animation = load_lottie_animation("ani2.json")

    # --- Header ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if header_animation:
            st_lottie(header_animation, speed=1, height=180, key="header_animation")

    st.markdown("<h1 style='text-align: center;'>JobFit AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Optimize your resume with AI-powered insights</p>",
                unsafe_allow_html=True)

    # --- Progress Indicator ---
    steps = ["Upload", "Analysis Options", "Results"]
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps), 1):
        with col:
            if i < st.session_state.step:
                st.markdown(
                    f"<div style='text-align:center;'><span style='color:{PRIMARY_COLOR}; font-size:1.5rem;'>✓</span><br><span style='color:{PRIMARY_COLOR};'>{step}</span></div>",
                    unsafe_allow_html=True)
            elif i == st.session_state.step:
                st.markdown(
                    f"<div style='text-align:center;'><span style='color:{PRIMARY_COLOR}; font-size:1.5rem; font-weight:bold;'>{i}</span><br><span style='color:{PRIMARY_COLOR}; font-weight:bold;'>{step}</span></div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='text-align:center;'><span style='color:gray; font-size:1.5rem;'>{i}</span><br><span style='color:gray;'>{step}</span></div>",
                    unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Step 1: Upload Documents ---
    if st.session_state.step == 1:
        st.markdown("<h2>Step 1: Upload Your Documents</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                "<div class='step-heading'><span style='font-size:1.5rem;'>📄</span><h3>Upload Resume</h3></div>",
                unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Select your resume (PDF format)",
                type=["pdf"],
                help="Upload your resume in PDF format"
            )

            if uploaded_file:
                st.session_state.resume_content = extract_pdf_text(uploaded_file)
                if st.session_state.resume_content:
                    st.success("✅ Resume uploaded successfully!")
                else:
                    st.error("❌ Failed to process the PDF. Please try again.")

        with col2:
            st.markdown(
                "<div class='step-heading'><span style='font-size:1.5rem;'>📋</span><h3>Job Description</h3></div>",
                unsafe_allow_html=True)
            job_description = st.text_area(
                "Paste the job description",
                value=st.session_state.job_description,
                height=250,
                placeholder="Paste the full job description here..."
            )

            st.session_state.job_description = job_description

        if st.button("Continue to Analysis Options", use_container_width=True):
            if not st.session_state.resume_content:
                st.error("Please upload your resume to continue.")
            elif not st.session_state.job_description.strip():
                st.error("Please paste the job description to continue.")
            else:
                go_to_step(2)

    # --- Step 2: Analysis Options ---
    elif st.session_state.step == 2:
        st.markdown("<h2>Step 2: Choose Analysis Type</h2>", unsafe_allow_html=True)

        # Define analysis options with icons and descriptions
        analysis_options = {
            "resume_review": {
                "icon": "📝",
                "title": "Resume Review",
                "desc": "Get feedback on strengths, weaknesses, and overall impact"
            },
            "skill_improvement": {
                "icon": "💡",
                "title": "Skill Improvement",
                "desc": "Find areas to improve your skills for this job"
            },
            "missing_keywords": {
                "icon": "🔍",
                "title": "Missing Keywords",
                "desc": "Identify keywords from the job ad missing in your resume"
            },
            "match_ats": {
                "icon": "📊",
                "title": "ATS Match Check",
                "desc": "Check resume compatibility with ATS systems"
            },
            "market_insights": {
                "icon": "📈",
                "title": "Market Insights",
                "desc": "Get salary ranges and market trends for this role"
            },
            "career_path": {
                "icon": "🧭",
                "title": "Career Path Options",
                "desc": "Explore potential career paths based on your resume"
            },
            "upskilling": {
                "icon": "📚",
                "title": "Upskilling Resources",
                "desc": "Find learning resources for skill gaps"
            },
            "linkedin": {
                "icon": "🔗",
                "title": "LinkedIn",
                "desc": "Generate an optimized LinkedIn profile summary"
            },
            "custom": {
                "icon": "❓",
                "title": "Custom Question",
                "desc": "Ask a specific question about your job fit"
            }
        }

        # Create a more visually appealing selection mechanism with cards
        cols = st.columns(3)
        for i, (option_key, option_data) in enumerate(analysis_options.items()):
            with cols[i % 3]:
                card_style = """
                    padding: 1rem;
                    border-radius: 10px;
                    margin-bottom: 1rem;
                    border: 2px solid transparent;
                    cursor: pointer;
                    height: 140px;
                    transition: all 0.2s ease;
                """

                # Highlight selected option
                if st.session_state.analysis_option == option_key:
                    card_style += f"border-color: {PRIMARY_COLOR}; background-color: rgba(67, 97, 238, 0.1);"
                else:
                    card_style += "border-color: #e0e0e0; background-color: white;"

                st.markdown(f"""
                <div style='{card_style}' onclick=''>
                    <div style='font-size: 1.8rem; margin-bottom: 0.5rem;'>{option_data['icon']}</div>
                    <div style='font-weight: bold; margin-bottom: 0.3rem;'>{option_data['title']}</div>
                    <div style='font-size: 0.8rem; color: #666;'>{option_data['desc']}</div>
                </div>
                """, unsafe_allow_html=True)

                # We need a button because the div onclick doesn't work in Streamlit
                if st.button(f"Select {option_data['title']}", key=f"btn_{option_key}", use_container_width=True):
                    st.session_state.analysis_option = option_key

        # Custom question input if selected
        if st.session_state.analysis_option == "custom":
            st.session_state.custom_query = st.text_area(
                "Enter your custom question:",
                value=st.session_state.custom_query,
                height=100,
                placeholder="Type your specific question here..."
            )

        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back", use_container_width=True):
                go_to_step(1)

        with col2:
            if st.button("Analyze Now →", use_container_width=True, type="primary"):
                if not st.session_state.analysis_option:
                    st.error("Please select an analysis type.")
                elif st.session_state.analysis_option == "custom" and not st.session_state.custom_query.strip():
                    st.error("Please enter your custom question.")
                else:
                    # Perform analysis
                    with st.spinner("Analyzing your resume... This may take a moment."):
                        # Get the appropriate prompt
                        prompts = get_analysis_prompts(st.session_state.custom_query)
                        selected_prompt = prompts[st.session_state.analysis_option]

                        # Generate response
                        st.session_state.analysis_result = get_gemini_response(
                            selected_prompt,
                            st.session_state.resume_content,
                            st.session_state.job_description
                        )

                        # Generate audio summary (first 500 chars)
                        if st.session_state.analysis_result:
                            if st.session_state.audio_path:
                                cleanup_temp_file(st.session_state.audio_path)

                            st.session_state.audio_path = speak_text(
                                st.session_state.analysis_result[:500]
                            )

                            go_to_step(3)
                        else:
                            st.error("Failed to generate analysis. Please try again.")

    # --- Step 3: Results ---
    elif st.session_state.step == 3:
        st.markdown("<h2>Analysis Results</h2>", unsafe_allow_html=True)

        if st.session_state.analysis_result:
            # Show summary info for ATS check
            if st.session_state.analysis_option == "match_ats":
                if "ATS-Friendly: Yes" in st.session_state.analysis_result:
                    st.success("✅ Your resume appears to be ATS-friendly!")
                elif "ATS-Friendly: No" in st.session_state.analysis_result:
                    st.error("❌ Your resume may have ATS compatibility issues.")

            # Results card
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown(st.session_state.analysis_result)
            st.markdown("</div>", unsafe_allow_html=True)

            # Audio summary
            if st.session_state.audio_path:
                st.markdown("<h3>Audio Summary</h3>", unsafe_allow_html=True)
                st.markdown("<div class='audio-player'>", unsafe_allow_html=True)
                st.audio(st.session_state.audio_path, format="audio/mp3")
                st.caption("Audio summary of the first portion of the analysis")
                st.markdown("</div>", unsafe_allow_html=True)

            # Actions row
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("← Back to Analysis Options", use_container_width=True):
                    go_to_step(2)

            with col2:
                if st.button("Start New Analysis", use_container_width=True):
                    # Clean up
                    if st.session_state.audio_path:
                        cleanup_temp_file(st.session_state.audio_path)
                        st.session_state.audio_path = None

                    # Reset state but keep documents
                    st.session_state.analysis_option = None
                    st.session_state.analysis_result = None
                    st.session_state.custom_query = ""
                    go_to_step(1)

            with col3:
                st.download_button(
                    label="Download Analysis",
                    data=st.session_state.analysis_result,
                    file_name="jobfit_analysis.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.error("No analysis results found. Please go back and try again.")
            if st.button("← Back to Start", use_container_width=True):
                go_to_step(1)

    # --- Footer ---
    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("Built with Streamlit & Google Gemini • JobFit Analyzer")
    with col2:
        st.caption("v2.0")


# Run the app
if __name__ == "__main__":
    main()
