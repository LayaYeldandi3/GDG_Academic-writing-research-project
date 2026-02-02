import streamlit as st
import requests
import arxiv
import pandas as pd
from io import StringIO
from groq import Groq
from fpdf import FPDF
from google import genai  
import re
import PyPDF2

# ==========================================
# 1. CONFIGURATION & CREDENTIALS
# ==========================================
# 🔴 REPLACE WITH YOUR ACTUAL KEYS
# GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE" 
# GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE" 
# USER_EMAIL = "YOUR EMAIL HERE (for OpenAlex API)" 
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
USER_EMAIL = os.getenv("USER_EMAIL")


# Initialize Clients
groq_client = Groq(api_key=GROQ_API_KEY)

# 🟢 NEW GEMINI CLIENT INITIALIZATION
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Error initializing Gemini Client: {e}")

# ==========================================
# 2. MODULE 1: DATA MANAGEMENT
# ==========================================
def fetch_papers(query, limit=10): 
    search = arxiv.Search(query=query, max_results=limit, sort_by=arxiv.SortCriterion.Relevance)
    results = []
    for r in search.results():
        results.append({
            "title": r.title,
            "abstract": r.summary.replace("\n", " "),
            "url": r.entry_id,
            "pdf_url": r.pdf_url,
            "doi": r.doi,
            "date": r.published.strftime("%Y")
        })
    return results

def enrich_metadata(title):
    base_url = "https://api.openalex.org/works"
    params = {"filter": f"title.search:{title}", "mailto": USER_EMAIL}
    try:
        res = requests.get(base_url, params=params).json()
        if res['results']:
            work = res['results'][0]
            return {
                "citations": work.get("cited_by_count", 0),
                "concepts": [c['display_name'] for c in work.get("concepts", [])[:3]]
            }
    except:
        pass
    return {"citations": 0, "concepts": []}

def clean_and_deduplicate(papers):
    clean_list = []
    seen_titles = set()
    for p in papers:
        if p['title'] in seen_titles: continue
        seen_titles.add(p['title'])
        if not p['abstract'] or len(p['abstract']) < 50: continue
        metadata = enrich_metadata(p['title'])
        p['citations'] = metadata['citations']
        p['concepts'] = metadata['concepts']
        clean_list.append(p)
    return clean_list[:5]

# ==========================================
# 3. MODULE 2: RESEARCHER (GROQ)
# ==========================================
def parse_markdown_sections(text):
    sections = {
        "summary": "Summary not generated.",
        "methodology": "Methodology not generated.",
        "analysis": "Analysis not generated.",
        "hypothesis": "Hypothesis not generated."
    }
    
    s_match = re.search(r"SUMMARY:\s*(.*?)\s*(?=METHODOLOGY:|ANALYSIS:|HYPOTHESIS:|$)", text, re.DOTALL | re.IGNORECASE)
    m_match = re.search(r"METHODOLOGY:\s*(.*?)\s*(?=ANALYSIS:|HYPOTHESIS:|$)", text, re.DOTALL | re.IGNORECASE)
    a_match = re.search(r"ANALYSIS:\s*(.*?)\s*(?=HYPOTHESIS:|$)", text, re.DOTALL | re.IGNORECASE)
    h_match = re.search(r"HYPOTHESIS:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    
    if s_match: 
        sections['summary'] = s_match.group(1).strip()
    if m_match: 
        sections['methodology'] = m_match.group(1).strip()
    if a_match: 
        sections['analysis'] = a_match.group(1).strip()
    if h_match: 
        sections['hypothesis'] = h_match.group(1).strip()
            
    return sections

def agent_logic_processor(paper_list):
    processed_kb = []
    progress_bar = st.progress(0)
    
    for i, paper in enumerate(paper_list):
        prompt = f"""
        Act as a Ph.D. Researcher. Analyze this abstract:
        TITLE: {paper['title']}
        ABSTRACT: {paper['abstract']}
        
        Strictly output the analysis using these 4 Headers.
        
        SUMMARY:
        [Write a VERY DETAILED 200-word technical summary. Do not be brief.]
        
        METHODOLOGY:
        [Explain the methods in at least 15 lines of text. Be specific about algorithms/data.]
        
        ANALYSIS:
        [List 5 Key Themes and 1 Critical Research Gap.]
        
        HYPOTHESIS:
        [Propose 1 novel hypothesis in bold text.]
        """
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.7 
            )
            raw_text = chat_completion.choices[0].message.content
            parsed = parse_markdown_sections(raw_text)
            paper.update(parsed)
            
        except Exception as e:
            paper["summary"] = f"Error: {str(e)}"
        
        processed_kb.append(paper)
        progress_bar.progress((i + 1) / len(paper_list))
    return processed_kb

def global_hypothesis_generator(paper_list, topic):
    context = ""
    for p in paper_list:
        context += f"Paper: {p['title']}\nGap: {p.get('analysis', '')}\nHypothesis: {p.get('hypothesis', '')}\n\n"
    
    prompt = f"""
    Act as a Principal Investigator. Topic: "{topic}".
    Findings: {context}
    Task: Formulate 3 Ranked Global Hypotheses. Each must have a Title, Statement, and Rationale.
    Output in Markdown format with Big Headings.
    """
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.7
        )
        return response.choices[0].message.content
    except:
        return "Global Hypothesis Generation Failed."

# ==========================================
# 4. MODULE 3: DATA ANALYST (GROQ)
# ==========================================
def generate_synthetic_data(topic):
    prompt = f"""
    Act as a Data Generator. Create a realistic CSV dataset for: "{topic}".
    Requirements: 20 rows, 4 columns (mix categorical/numeric), realistic values.
    Output ONLY CSV text.
    """
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.5
        )
        return response.choices[0].message.content
    except:
        return None

def data_analyst_agent(df):
    report = {}
    df_clean = df.copy()
    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
             df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        else:
             df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown")
    
    report['cleaned_data'] = df_clean
    report['statistics'] = df_clean.describe()
    
    stats_text = report['statistics'].to_string()
    prompt = f"""
    Act as a Lead Data Scientist. Analyze this statistics summary: {stats_text}
    
    Provide a DETAILED report with 3 sections:
    1. Data Quality Assessment (10 lines)
    2. Statistical Patterns & Outliers (10 lines)
    3. Research Implications (5 lines)
    """
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.5 
        )
        report['ai_insight'] = response.choices[0].message.content
    except:
        report['ai_insight'] = "N/A"
    return report

# ==========================================
# 5. MODULE 4: WRITER AGENT (UPDATED - NEW GEMINI SDK)
# ==========================================
def writer_agent_universal(topic, literature_data, global_hypothesis, analyst_insight):
    context = f"TOPIC: {topic}\n\nLITERATURE:\n"
    for p in literature_data:
        context += f"Title: {p['title']}\nSummary: {p['summary']}\nMethod: {p['methodology']}\nGap: {p['analysis']}\n\n"
    context += f"HYPOTHESES:\n{global_hypothesis}\n\nDATA INSIGHTS:\n{analyst_insight}\n"
    
    prompt = "Write a full academic Research Paper. Sections: Title, Abstract, Intro, Lit Review, Methodology, Results, Conclusion. No Markdown."
    
    try:
        # 🟢 NEW: Using google.genai syntax with gemini-2.5-flash
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"{context}\n\n{prompt}"
        )
        return response.text, "Gemini-2.5-Flash"
    except Exception as e:
        return f"Error: {str(e)}", "None"

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Agentic AI Research Report', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)

def generate_pdf_from_text(text_content):
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font('Arial', '', 11)
    clean_text = text_content.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, clean_text)
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 6. MODULE 5: DRAFT EDITOR AGENT (UPDATED - NEW GEMINI SDK)
# ==========================================
def read_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def editor_agent(draft_text, instruction="Improve flow and academic tone"):
    prompt = f"""
    Act as a Senior Academic Editor. 
    User Instruction: "{instruction}"
    
    Draft Content:
    {draft_text}
    
    Task:
    1. Critique: Briefly list 3 strengths and 3 weaknesses.
    2. Improvements: Provide a list of specific actionable changes.
    3. Rewrite: Rewrite the draft applying these improvements. 
    
    Output in Markdown. Use bold headers.
    """
    try:
        # 🟢 NEW: Using google.genai syntax with gemini-2.5-flash
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Editing Error (Gemini): {e}"

# ==========================================
# 6. MAIN UI
# ==========================================
def main():
    st.set_page_config(page_title="Agentic Research AI", layout="wide")
    
    # 🎨 CSS INJECTION
    st.markdown("""
        <style>
        .stApp { background-color: #E8F5E9 !important; }
        header[data-testid="stHeader"] { background-color: transparent !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(255,255,255,0.5);
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            color: #2e7d32;
            border: none;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            color: #1b5e20 !important;
            font-weight: bold;
            box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
        }
        div[data-baseweb="tab-panel"] {
            background: rgba(255, 255, 255, 0.4);
            backdrop-filter: blur(12px);
            border-radius: 0 15px 15px 15px;
            padding: 25px !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            border: 1px solid rgba(255, 255, 255, 0.5);
        }
        .glass-title {
            background: rgba(255, 255, 255, 0.4);
            backdrop-filter: blur(12px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.5);
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }
        h1, h2, h3, p, div, span { color: #1B5E20 !important; font-family: 'Helvetica Neue', sans-serif; }
        .stTextInput input, .stTextArea textarea {
            background-color: rgba(255, 255, 255, 0.8) !important;
            border: 1px solid #a5d6a7;
            color: #1b5e20;
            border-radius: 10px;
        }
        .stButton button {
            background: linear-gradient(135deg, #2e7d32, #1b5e20) !important;
            color: white !important;
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }
        .stButton button * { color: white !important; }
        </style>
    """, unsafe_allow_html=True)

    # --- TITLE SECTION ---
    st.markdown("""
    <div class="glass-title">
        <h1 style="margin:0; font-size: 3rem;">✨ Scholar Bot</h1>
        <p style="font-size: 1.2rem; opacity: 0.8; margin-top: 10px;">
            A Multi-Agent Research System
        </p>
    </div>
    """, unsafe_allow_html=True)

    if 'topic' not in st.session_state: 
        st.session_state.topic = ""
    if 'final_kb' not in st.session_state: 
        st.session_state.final_kb = []
    if 'global_hyp' not in st.session_state: 
        st.session_state.global_hyp = ""
    if 'analyst_result' not in st.session_state: 
        st.session_state.analyst_result = None
    if 'current_df' not in st.session_state: 
        st.session_state.current_df = None
    if 'editor_response' not in st.session_state: 
        st.session_state.editor_response = ""

    # TABS
    tab_research, tab_hypothesis, tab_analyst, tab_writer, tab_editor = st.tabs([
        "1. ScholarScout", 
        "2. Global Hypotheses", 
        "3. Analytica", 
        "4. ManuScriptor",
        "5. Draft Editor"
    ])

    # --- TAB 1: RESEARCHER ---
    with tab_research:
        st.header("📚 Literature Search")
        st.session_state.topic = st.text_input(
            "Research Topic:", 
            value=st.session_state.topic,
            placeholder="Enter your research topic..."
        )
        
        if st.button("Start Research Agents"):
            with st.status("Research in progress...", expanded=True):
                raw = fetch_papers(st.session_state.topic, limit=8)
                clean = clean_and_deduplicate(raw)
                st.session_state.final_kb = agent_logic_processor(clean)
                st.session_state.global_hyp = global_hypothesis_generator(st.session_state.final_kb, st.session_state.topic)
            st.success("Research Complete!")

        if st.session_state.final_kb:
            st.divider()
            st.subheader("📊 Research Dashboard")
            st.dataframe(pd.DataFrame(st.session_state.final_kb)[['title', 'date', 'citations']], use_container_width=True)
            st.subheader("📝 Detailed Paper Analysis")
            for i, p in enumerate(st.session_state.final_kb):
                with st.container(border=True):
                    c1, c2 = st.columns([0.85, 0.15])
                    with c1:
                        st.markdown(f"**{i+1}. {p['title']}**")
                        st.caption(f"📅 {p['date']} | 🔗 Citations: {p['citations']} | 🏷️ {', '.join(p['concepts'])}") 
                    with c2:
                        if p.get('pdf_url'):
                            st.link_button("🔗 Full Paper", p['pdf_url'])
                        elif p.get('url'):
                            st.link_button("🔗 Link", p['url'])
                    t1, t2, t3, t4 = st.tabs(["Summary", "Methodology", "Analysis", "Hypothesis"])
                    with t1: 
                        st.write(p.get('summary', 'Pending'))
                    with t2: 
                        st.write(p.get('methodology', 'Pending'))
                    with t3: 
                        st.write(p.get('analysis', 'Pending'))
                    with t4: 
                        st.info(p.get('hypothesis', 'Pending'))

    # --- TAB 2: GLOBAL HYPOTHESIS ---
    with tab_hypothesis:
        st.header("💡 Top Ranked Global Hypotheses")
        if st.session_state.global_hyp:
            st.markdown(st.session_state.global_hyp)
        else:
            st.info("Run the Research Agent in Tab 1 to generate hypotheses.")

    # --- TAB 3: DATA ANALYST ---
    with tab_analyst:
        st.header("📈 Analytica")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            st.session_state.current_df = pd.read_csv(uploaded_file)
        elif st.button("Generate Synthetic Data"):
            with st.spinner("Generating synthetic data..."):
                csv_str = generate_synthetic_data(st.session_state.topic)
                if csv_str: 
                    st.session_state.current_df = pd.read_csv(StringIO(csv_str))
                    st.success("Synthetic Data Created!")

        if st.session_state.current_df is not None:
            st.dataframe(st.session_state.current_df.head())
            if st.button("Run Analyst Agent"):
                with st.spinner("Analyzing dataset..."):
                    st.session_state.analyst_result = data_analyst_agent(st.session_state.current_df)
                st.success("Analysis Complete!")
                
        if st.session_state.analyst_result:
            with st.container(border=True):
                res = st.session_state.analyst_result
                st.write("### 🧠 Deep Statistical Narrative")
                st.success(res['ai_insight'])
                st.write("### 📉 Visualization")
                numeric_cols = res['cleaned_data'].select_dtypes(include=['number'])
                if not numeric_cols.empty:
                    st.bar_chart(numeric_cols)
                else:
                    st.warning("No numeric data columns found for visualization.")

    # --- TAB 4: WRITER AGENT ---
    with tab_writer:
        st.header("✍️ ManuScriptor")
        if st.session_state.final_kb:
            if st.button("Generate PDF Report"):
                with st.spinner("Writing paper..."):
                    insight = st.session_state.analyst_result['ai_insight'] if st.session_state.analyst_result else "No Data Analysis Performed."
                    full_text, model_used = writer_agent_universal(
                        st.session_state.topic, 
                        st.session_state.final_kb, 
                        st.session_state.global_hyp, 
                        insight
                    )
                    pdf_bytes = generate_pdf_from_text(full_text)
                    st.download_button(
                        label="⬇️ Download Final PDF",
                        data=pdf_bytes,
                        file_name="Final_Research_Paper.pdf",
                        mime="application/pdf"
                    )
        else:
            st.warning("Please complete the Research Phase (Tab 1) first.")

    # --- TAB 5: DRAFT EDITOR (GEMINI) ---
    with tab_editor:
        st.header("🖊️ Draft Editor & Improver")
        st.write("Upload an existing draft or paste your text for AI critique and improvements.")
        
        # Input selection
        input_method = st.radio("Choose Input Method:", ["Paste Text", "Upload File (PDF/TXT)"], horizontal=True)
        
        draft_content = ""
        
        if input_method == "Paste Text":
            draft_content = st.text_area("Paste your draft here:", height=300)
        else:
            uploaded_draft = st.file_uploader("Upload your paper", type=["pdf", "txt", "md"])
            if uploaded_draft:
                if uploaded_draft.type == "application/pdf":
                    draft_content = read_pdf(uploaded_draft)
                else:
                    stringio = StringIO(uploaded_draft.getvalue().decode("utf-8"))
                    draft_content = stringio.read()
                
                if draft_content:
                    st.info("File loaded successfully.")
                    with st.expander("Preview Content"):
                        st.text(draft_content[:1000] + "...")
        
        # Instructions
        edit_instruction = st.text_input("Editing Instructions (e.g., 'Make it more formal', 'Fix grammar', 'Improve clarity')", value="")
        
        if st.button("Analyze & Improve Draft"):
            if draft_content:
                with st.spinner("reviewing your work..."):
                    st.session_state.editor_response = editor_agent(draft_content, edit_instruction)
                st.success("Editing Complete!")
            else:
                st.warning("Please provide some text to edit.")
        
        # Display Results
        if st.session_state.editor_response:
            st.markdown("### 📝 Editor Feedback & Rewrite")
            st.markdown(st.session_state.editor_response)
            
            # Download revised text
            st.download_button(
                label="⬇️ Download Feedback",
                data=st.session_state.editor_response,
                file_name="Editor_Feedback.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main()
