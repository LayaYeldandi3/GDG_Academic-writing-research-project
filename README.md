# 🎓 Scholar Bot: Multi-Agent Academic Research System

**Scholar Bot** is an autonomous, multi-agent AI framework designed to streamline the academic research lifecycle. Built with **Streamlit**, it orchestrates a collaborative workflow between specialized AI agents to handle literature review, hypothesis generation, statistical data analysis, and academic drafting.

This project leverages a **Hybrid Inference Strategy**: it utilizes **Groq (Llama-3.3-70b)** for high-speed logical reasoning and data parsing, and **Google Gemini (2.5-Flash)** for long-context creative synthesis and drafting.

---

## 🚀 Key Features & Modules

The application is divided into five autonomous modules (Agents), each responsible for a specific stage of the research process:

### 1. 📚 ScholarScout (The Researcher)
* **Real-Time Data:** Connects to the **arXiv API** to fetch the latest academic papers based on user queries.
* **Metadata Enrichment:** Integrates with **OpenAlex API** to retrieve citation counts and related concepts, prioritizing high-impact research.
* **Intelligent Parsing:** Uses **Groq (Llama-3.3)** to read abstracts and extract structured data: *Technical Summary, Methodology, Key Findings, and Research Gaps*.

### 2. 💡 Global Hypothesis Generator
* **Synthesis:** Aggregates the "Research Gaps" identified by ScholarScout across multiple papers.
* **Reasoning:** Formulates ranked, testable global hypotheses that address these gaps, ensuring the research has a solid theoretical foundation.

### 3. 📈 Analytica (The Data Analyst)
* **Flexible Input:** Accepts user-uploaded CSVs or generates realistic **synthetic datasets** on the fly for testing.
* **Automated Cleaning:** Uses **Pandas** to detect and impute missing values (mean/mode strategies) automatically.
* **Statistical Narrative:** Generates a "Deep Statistical Narrative" using **Groq**, interpreting descriptive statistics to find outliers, trends, and research implications.
* **Visualization:** Auto-generates charts based on numeric data distributions.

### 4. ✍️ ManuScriptor (The Writer)
* **Long-Context Drafting:** Leveraging the massive context window of **Gemini-2.5-Flash**, this agent ingests the entire Knowledge Base (summaries, methodologies, data insights) in a single pass.
* **Academic Output:** Writes a cohesive research paper (Title, Abstract, Introduction, Literature Review, Methodology, Results, Conclusion).
* **PDF Export:** Uses **FPDF** to compile the generated text into a downloadable PDF report.

### 5. 🖊️ Draft Editor
* **Refinement:** A dedicated tool for reviewing existing work. Users can upload PDFs or paste text.
* **Critique & Rewrite:** Uses **Gemini-2.5-Flash** to analyze tone, flow, and grammar, providing specific actionable improvements and rewritten sections based on user prompts.

---

## 🛠️ Tech Stack

* **Frontend:** `Streamlit` (UI, Session State management)
* **AI Inference (Logic & Speed):** `Groq API` (Model: `llama-3.3-70b-versatile`)
* **AI Inference (Context & Writing):** `Google GenAI SDK` (Model: `gemini-2.5-flash`)
* **Academic Data Sources:** `arXiv API` (Paper fetching), `OpenAlex API` (Citation metrics)
* **Data Processing:** `Pandas` (Dataframes), `PyPDF2` (PDF Parsing), `Regular Expressions`
* **Output Generation:** `FPDF` (PDF Creation)

---

## ⚙️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/your-username/scholar-bot.git](https://github.com/your-username/scholar-bot.git)
   cd scholar-bot
   
2. **Install Dependencies**
   ```bash
   pip install streamlit requests arxiv pandas groq fpdf google-genai PyPDF2
   
3. **Configure API Keys** : Open the app.py file and replace the placeholder keys with your actual credentials:
   GROQ_API_KEY = "your_groq_key_here"
   GEMINI_API_KEY = "your_gemini_key_here"

4. **Run the Application**
   ```bash
   streamlit run app.py

🧠 **Architecture Flow**

   **Input:** User defines a research topic.
   
   **Search:** System queries arXiv -> Fetches Top N Papers.
   
   **Process:** Llama-3.3 reads abstracts -> Outputs JSON-structured Knowledge Base.
   
   **Analyze:** System processes Data (CSV) -> Generates Stats & Insights.
   
   **Synthesize:** Gemini-2.5 takes [Knowledge Base + Hypothesis + Data Insights] -> Generates Full Paper.
   
   **Export:** Result is compiled into a PDF.
