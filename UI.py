"""
Front end for the Hyderabad Institute of Technology RAG assistant.

"""

from __future__ import annotations

import html
import textwrap
import time

import streamlit as st

from app import ask_question 
from pathlib import Path
from add_document import add_document 



# CONFIGURATION



CONTACT = {
    "name": "Tanishk",
    "focus": "Python · Retrieval-augmented generation · Applied LLM systems",
    "linkedin": "https://www.linkedin.com/in/tanishk-agarwal-78a38b262/", 
    "github": "https://github.com/Tanishk-2004", 
    "email": "agarwaltanishk73@gmail.com",  
    "repo": "https://github.com/Tanishk-2004/University-RAG",  
}


PIPELINE = {
    "documents": "20 policy documents",
    "pages": "~180 pages",
    "areas": "20 policy areas",
    "retrieval": "MMR + multi-query",
    "model": "GPT-OSS-120B (Groq)",
    "store": "Chroma, persistent",
}


EXAMPLES = [
    ("Attendance & condonation", "What is the minimum attendance requirement, and how does condonation work?"),
    ("Grading & examinations", "How are internal and external marks combined into the final grade?"),
    ("Fees & refunds", "What is the refund policy if a student withdraws after paying tuition fees?"),
    ("Internships & NOC", "Do I need a No Objection Certificate for an internship during the semester?"),
    ("Library borrowing", "How many physical books can a 4th-year B.Tech student borrow from the central library simultaneously?"),
    ("Campus speed limit", "What is the absolute maximum speed limit for vehicles driven within the HIT campus boundary?"),
]

QUESTION_KEY = "hit_question"
RESULT_KEY = "hit_result"
ERROR_KEY = "hit_error"
NOTICE_KEY = "hit_notice"

st.set_page_config(
    page_title="HIT Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="locked",
)



# DESIGN SYSTEM



THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,300..600&display=swap');

:root {
  --paper: #F1F3F6;
  --card: #FFFFFF;
  --ink: #0E1A2B;
  --ink-hover: #1B2E47;
  --slate: #55677D;
  --slate-soft: #7D8CA0;
  --rule: #DBE1EA;
  --rule-soft: #E9EDF3;
  --verdigris: #1E6C58;
  --verdigris-soft: #EBF3F0;
  --oxblood: #8B2E3C;
  --oxblood-soft: #F8EEEF;
  --sans: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --mono: 'IBM Plex Mono', ui-monospace, 'SFMono-Regular', Menlo, monospace;
  --serif: 'Newsreader', Georgia, 'Times New Roman', serif;
}

/* ---------- app shell ---------- */
.stApp, [data-testid="stAppViewContainer"] { background: var(--paper); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDeployButton"], [data-testid="stAppDeployButton"] { display: none; }
[data-testid="stToolbar"], #MainMenu { display: none !important; }
footer { visibility: hidden; }
.block-container, .stMainBlockContainer, [data-testid="stMainBlockContainer"] {
  max-width: 1040px;
  padding-top: 3rem;
  padding-bottom: 4.5rem;
}

/* ---------- spinner ---------- */
[data-testid="stSpinner"] {
    color: var(--ink) !important;
}

[data-testid="stSpinner"] svg {
    color: var(--ink) !important;
    fill: var(--ink) !important;
}

/* ---------- base type ---------- */
.stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp textarea,
.stApp [data-testid="stMarkdownContainer"] {
  font-family: var(--sans);
  color: var(--ink);
}
.stApp h1, .stApp h2, .stApp h3 { font-family: var(--serif); color: var(--ink); font-weight: 500; }
.stApp a { color: var(--verdigris); text-decoration: none; border-bottom: 1px solid rgba(30,108,88,.3); }
.stApp a:hover { border-bottom-color: var(--verdigris); }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: var(--card); border-right: 1px solid var(--rule); }
[data-testid="stSidebar"] > div { padding-top: 1.25rem; }
[data-testid="stSidebarNav"] { padding-top: .25rem; }
[data-testid="stSidebarNav"] ul { gap: 2px; }
[data-testid="stSidebarNav"] a { border-radius: 6px; border-bottom: 0; padding-top: .4rem; padding-bottom: .4rem; }
[data-testid="stSidebarNav"] a span { font-family: var(--sans); font-size: .875rem; color: var(--slate); }
[data-testid="stSidebarNav"] a:hover { background: var(--rule-soft); }
[data-testid="stSidebarNav"] a[aria-current="page"] {
  background: var(--oxblood-soft);
  box-shadow: inset 2px 0 0 var(--oxblood);
}
[data-testid="stSidebarNav"] a[aria-current="page"] span { color: var(--ink); font-weight: 600; }

/* ---------- buttons ---------- */
.stButton > button, .stFormSubmitButton > button {
  font-family: var(--sans);
  font-size: .9rem;
  font-weight: 500;
  border-radius: 6px;
  border: 1px solid var(--rule);
  background: var(--card);
  color: var(--ink);
  padding: .55rem 1.1rem;
  transition: background .15s ease, border-color .15s ease, color .15s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
  border-color: var(--ink);
  background: var(--card);
  color: var(--ink);
}
.stFormSubmitButton > button, .stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background: #f5d4d4;
  border: 1px solid #752C36; 
  color: #0F172A; 
  padding: .6rem 1.6rem;
  letter-spacing: .01em;
  border-radius: 6px; 
  font-weight: 500;
}
.stFormSubmitButton > button:hover, .stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {
  background: #f4cccc;
  border-color: #752C36;
  color: #0F172A;
}
.stButton > button:focus-visible, .stFormSubmitButton > button:focus-visible {
  outline: 2px solid var(--verdigris);
  outline-offset: 2px;
}

/* ---------- inputs ---------- */
[data-baseweb="textarea"], [data-baseweb="input"], [data-baseweb="base-input"] {
  background: var(--card);
  border-radius: 8px;
  border-color: var(--rule);
}
[data-baseweb="textarea"]:focus-within, [data-baseweb="input"]:focus-within {
  border-color: var(--verdigris);
  box-shadow: 0 0 0 3px var(--verdigris-soft);
}
.stApp textarea, .stApp input {
  background: var(--card) !important;
  color: var(--ink) !important;
  font-size: 1rem !important;
  line-height: 1.6 !important;
}
.stApp textarea::placeholder { color: var(--slate-soft) !important; }
[data-testid="InputInstructions"] { display: none; }

/* ---------- expanders and code ---------- */
[data-testid="stExpander"] {
  border: 1px solid var(--rule);
  border-radius: 8px;
  background: var(--card);
  overflow: hidden;
}
[data-testid="stExpander"] summary p {
  font-family: var(--mono);
  font-size: .74rem;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--slate);
}
[data-testid="stExpander"] summary:hover p { color: var(--ink); }
.stApp code { font-family: var(--mono); font-size: .82rem; }

/* ---------- answer card ---------- */
.hit-marker { display: none; }
.stElementContainer:has(> .stMarkdown .hit-marker),
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] .hit-marker) { display: none; }
[data-testid="stVerticalBlockBorderWrapper"]:has(.hit-marker) {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
}
.stVerticalBlock:has(> .stElementContainer > .stMarkdown .hit-marker),
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > [data-testid="stMarkdown"] .hit-marker) {
  background: var(--card);
  border: 1px solid var(--rule);
  border-left: 3px solid var(--oxblood);
  border-radius: 10px;
  padding: 1.6rem 1.8rem 1.4rem;
  box-shadow: 0 1px 2px rgba(14,26,43,.05);
}
/* Answer body only: the generated markdown has unclassed <p>/<li>, while the
   card's own eyebrow and meta lines carry .hit-* classes and keep their styling. */
.stVerticalBlock:has(> .stElementContainer > .stMarkdown .hit-marker) p:not([class]),
.stVerticalBlock:has(> .stElementContainer > .stMarkdown .hit-marker) li:not([class]),
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > [data-testid="stMarkdown"] .hit-marker) p:not([class]),
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > [data-testid="stMarkdown"] .hit-marker) li:not([class]) {
  font-family: var(--serif);
  font-size: 1.14rem;
  line-height: 1.72;
  color: var(--ink);
}

/* ---------- typographic components ---------- */
.hit-eyebrow {
  font-family: var(--mono);
  font-size: .72rem;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--slate-soft);
  margin: 0 0 .9rem;
}
.hit-display {
  font-family: var(--serif);
  font-size: clamp(1.95rem, 4.2vw, 2.85rem);
  font-weight: 400;
  line-height: 1.14;
  letter-spacing: -.015em;
  color: var(--ink);
  margin: 0 0 1.1rem;
  max-width: 22ch;
}
.hit-display.hit-wide { max-width: 30ch; }
.hit-lede {
  font-size: 1.05rem;
  line-height: 1.68;
  color: var(--slate);
  max-width: 62ch;
  margin: 0;
}
.hit-h2 {
  font-family: var(--serif);
  font-size: 1.45rem;
  font-weight: 500;
  color: var(--ink);
  margin: 0 0 .35rem;
}
.hit-sub { font-size: .95rem; line-height: 1.65; color: var(--slate); max-width: 62ch; margin: 0 0 1.1rem; }
.hit-hint { font-family: var(--mono); font-size: .74rem; color: var(--slate-soft); margin: .5rem 0 0; }
.hit-section { margin-top: 3.25rem; }
.hit-section-tight { margin-top: 2.25rem; }
.hit-rule { height: 1px; background: var(--rule); border: 0; margin: 2.5rem 0 0; }
.hit-accent-rule { width: 46px; height: 2px; background: var(--oxblood); margin: 0 0 1.5rem; }

/* ---------- record strip ---------- */
.hit-record {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1px;
  background: var(--rule);
  border: 1px solid var(--rule);
  border-radius: 10px;
  overflow: hidden;
  margin-top: 2rem;
}
.hit-record div { background: var(--card); padding: 1rem 1.2rem; }
.hit-record dt {
  font-family: var(--mono);
  font-size: .68rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--slate-soft);
  margin: 0 0 .35rem;
}
.hit-record dd { font-family: var(--sans); font-size: 1.02rem; font-weight: 500; color: var(--ink); margin: 0; }

/* ---------- ledger (ordered steps) ---------- */
.hit-ledger { border-top: 1px solid var(--rule); margin-top: 1.25rem; }
.hit-step {
  display: grid;
  grid-template-columns: 2.4rem 1fr;
  gap: 1rem;
  padding: .95rem .25rem;
  border-bottom: 1px solid var(--rule-soft);
}
.hit-step span {
  font-family: var(--mono);
  font-size: .78rem;
  color: var(--oxblood);
  padding-top: .15rem;
}
.hit-step p { margin: 0; font-size: .95rem; line-height: 1.6; color: var(--slate); }
.hit-step strong { color: var(--ink); font-weight: 600; }
.hit-stage {
  font-family: var(--mono);
  font-size: .7rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--verdigris);
  margin: 0;
}

/* ---------- stack rows ---------- */
.hit-stack { border-top: 1px solid var(--rule); margin-top: 1.25rem; }
.hit-stack-row {
  display: grid;
  grid-template-columns: minmax(120px, 200px) 1fr;
  gap: 1rem;
  padding: .8rem .25rem;
  border-bottom: 1px solid var(--rule-soft);
  align-items: baseline;
}
.hit-stack-row dt {
  font-family: var(--mono);
  font-size: .72rem;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--slate-soft);
}
.hit-stack-row dd { margin: 0; font-size: .95rem; color: var(--ink); }

/* ---------- figures ---------- */
.hit-figures {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  margin-top: 1.5rem;
}
.hit-figure { background: var(--card); border: 1px solid var(--rule); border-radius: 10px; padding: 1.3rem 1.4rem; }
.hit-figure b {
  display: block;
  font-family: var(--mono);
  font-size: 1.8rem;
  font-weight: 500;
  color: var(--verdigris);
  letter-spacing: -.02em;
}
.hit-figure p { margin: .5rem 0 0; font-size: .87rem; line-height: 1.55; color: var(--slate); }

/* ---------- notices ---------- */
.hit-notice {
  border: 1px solid var(--rule);
  border-left: 3px solid var(--slate-soft);
  border-radius: 8px;
  background: var(--card);
  padding: 1rem 1.2rem;
  font-size: .95rem;
  line-height: 1.6;
  color: var(--slate);
}
.hit-notice.hit-error { border-left-color: var(--oxblood); background: var(--oxblood-soft); color: var(--ink); }
.hit-notice strong { color: var(--ink); }

/* ---------- answer meta + sources ---------- */
.hit-meta {
  font-family: var(--mono);
  font-size: .72rem;
  letter-spacing: .04em;
  color: var(--slate-soft);
  margin: .9rem 0 0;
}
.hit-question-echo {
  font-family: var(--mono);
  font-size: .74rem;
  letter-spacing: .06em;
  text-transform: uppercase;
  color: var(--slate-soft);
  margin: 0 0 .1rem;
}
.hit-source { border-top: 1px solid var(--rule-soft); padding: .9rem 0 .2rem; }
.hit-source:first-child { border-top: 0; }
.hit-source p { font-family: var(--mono); font-size: .74rem; color: var(--oxblood); margin: 0 0 .4rem; }
.hit-source pre {
  font-family: var(--mono);
  font-size: .78rem;
  line-height: 1.6;
  color: var(--slate);
  white-space: pre-wrap;
  margin: 0;
  background: transparent;
}

/* ---------- contact ---------- */
.hit-contact { border-top: 1px solid var(--rule); margin-top: 1.5rem; }
.hit-contact-row {
  display: grid;
  grid-template-columns: minmax(110px, 160px) 1fr;
  gap: 1rem;
  padding: .95rem .25rem;
  border-bottom: 1px solid var(--rule-soft);
  align-items: baseline;
}
.hit-contact-row dt {
  font-family: var(--mono);
  font-size: .72rem;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--slate-soft);
}
.hit-contact-row dd { margin: 0; font-size: .98rem; }
.hit-empty { font-family: var(--mono); font-size: .8rem; color: var(--slate-soft); }

/* ---------- footer ---------- */
.hit-footer {
  font-size: .82rem;
  line-height: 1.6;
  color: var(--slate-soft);
  margin-top: 1.25rem;
  max-width: 66ch;
}

/* ---------- sidebar panel ---------- */
.hit-panel { border-top: 1px solid var(--rule); padding-top: 1rem; }
.hit-panel dl { margin: 0; }
.hit-panel dt {
  font-family: var(--mono);
  font-size: .66rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--slate-soft);
  margin-top: .75rem;
}
.hit-panel dd { margin: .15rem 0 0; font-size: .84rem; color: var(--ink); }
.hit-wordmark { font-family: var(--serif); font-size: 1.05rem; color: var(--ink); margin: 0 0 .2rem; }

@media (max-width: 640px) {
  .block-container, .stMainBlockContainer { padding-top: 1.75rem; }
  .hit-display { max-width: none; }
  .hit-step { grid-template-columns: 1.9rem 1fr; gap: .75rem; }
  .hit-stack-row, .hit-contact-row { grid-template-columns: 1fr; gap: .2rem; }
}

@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
</style>
"""


def markup(block: str) -> None:
    """Render a raw HTML block (dedented, no blank lines inside)."""
    st.markdown(textwrap.dedent(block).strip(), unsafe_allow_html=True)


def section(title: str, subtitle: str = "", tight: bool = False) -> None:
    spacing = "hit-section-tight" if tight else "hit-section"
    sub = f'<p class="hit-sub">{subtitle}</p>' if subtitle else ""
    markup(f'<div class="{spacing}"><h2 class="hit-h2">{title}</h2>{sub}</div>')


def footer_note() -> None:
    markup(
        """
        <hr class="hit-rule" />
        <p class="hit-footer">Hyderabad Institute of Technology is a fictional institution.
        The knowledge base was written for this project, so it demonstrates the retrieval
        pipeline rather than any real university's regulations.</p>
        """
    )


def sidebar_panel() -> None:
    """Project identity and configuration, shown under the navigation."""
    with st.sidebar:
        markup(
            f"""
            <div class="hit-panel">
            <p class="hit-wordmark">HIT Knowledge Assistant</p>
            <p class="hit-eyebrow" style="margin:0">Retrieval-augmented generation</p>
            <dl>
            <dt>Knowledge base</dt><dd>{PIPELINE['documents']} · {PIPELINE['pages']}</dd>
            <dt>Retrieval</dt><dd>{PIPELINE['retrieval']}</dd>
            <dt>Vector store</dt><dd>{PIPELINE['store']}</dd>
            <dt>Generation</dt><dd>{PIPELINE['model']}</dd>
            </dl>
            </div>
            """
        )



# PAGE 1 — ABOUT PROJECT


def page_about() -> None:
    markup(
        f"""
        <p class="hit-eyebrow">Portfolio project · Python · LangChain</p>
        <h1 class="hit-display">Ask a university's rulebook a question, get the answer it actually contains.</h1>
        <p class="hit-lede">A language model on its own has never read your institution's handbooks.
        This assistant searches documents of university policy first, then answers from
        what it found — and says so plainly when the answer is not in there.</p>
        <div class="hit-record">
        <div><dt>Source documents</dt><dd>20 PDFs</dd></div>
        <div><dt>Indexed</dt><dd>~180 pages</dd></div>
        <div><dt>Policy areas</dt><dd>20 topics</dd></div>
        <div><dt>Benchmark</dt><dd>50 questions</dd></div>
        </div>
        """
    )

    section(
        "The problem it solves",
        "Rules that students actually need — attendance thresholds, refund windows, "
        "internship approvals — live in long PDFs that nobody reads end to end. A general "
        "model asked about them will guess, confidently. Retrieval-augmented generation "
        "removes the guessing: the model only sees passages pulled from the real documents, "
        "so every answer traces back to something in the knowledge base.",
    )

    section("How a question is answered", tight=False)
    markup(
        """
        <p class="hit-stage">Stage 1 — Indexing, once per knowledge base</p>
        <div class="hit-ledger">
        <div class="hit-step"><span>01</span><p><strong>Load and clean.</strong>
        Every PDF is read page by page and stripped of whitespace and formatting artefacts.</p></div>
        <div class="hit-step"><span>02</span><p><strong>Split into overlapping chunks.</strong>
        Overlap keeps a sentence's context intact when it lands on a chunk boundary.</p></div>
        <div class="hit-step"><span>03</span><p><strong>Embed and store.</strong>
        Each chunk becomes a vector with all-MiniLM-L6-v2 and is written to a persistent
        Chroma database, so indexing never has to run again.</p></div>
        </div>
        """
    )
    markup(
        """
        <div class="hit-section-tight">
        <p class="hit-stage">Stage 2 — Answering, on every question</p>
        <div class="hit-ledger">
        <div class="hit-step"><span>04</span><p><strong>Rewrite the question.</strong>
        A multi-query step generates several phrasings, so retrieval does not hinge on the
        exact words a visitor happened to use.</p></div>
        <div class="hit-step"><span>05</span><p><strong>Retrieve with MMR.</strong>
        Maximum Marginal Relevance favours passages that are relevant <em>and</em> different
        from each other, which keeps three copies of the same paragraph out of the context.</p></div>
        <div class="hit-step"><span>06</span><p><strong>Answer from the retrieved text.</strong>
        The passages and the question go to the language model together. If the documents do
        not cover it, the assistant returns a fallback instead of inventing a rule.</p></div>
        </div>
        </div>
        """
    )

    section("What is under it", tight=False)
    markup(
        f"""
        <dl class="hit-stack">
        <div class="hit-stack-row"><dt>Orchestration</dt><dd>LangChain</dd></div>
        <div class="hit-stack-row"><dt>Documents</dt><dd>PyPDFLoader with RecursiveCharacterTextSplitter</dd></div>
        <div class="hit-stack-row"><dt>Embeddings</dt><dd>all-MiniLM-L6-v2 (Sentence Transformers)</dd></div>
        <div class="hit-stack-row"><dt>Vector store</dt><dd>Chroma, persisted to disk</dd></div>
        <div class="hit-stack-row"><dt>Retrieval</dt><dd>{PIPELINE['retrieval']}</dd></div>
        <div class="hit-stack-row"><dt>Generation</dt><dd>{PIPELINE['model']}</dd></div>
        <div class="hit-stack-row"><dt>Evaluation</dt><dd>BERTScore and LLM-as-a-judge</dd></div>
        <div class="hit-stack-row"><dt>Interface</dt><dd>Streamlit</dd></div>
        </dl>
        """
    )

    section(
        "Measured, not assumed",
        "Fifty questions with hand-written reference answers were run through the pipeline "
        "and scored two ways — an embedding-based similarity metric and a second model "
        "judging each answer against its reference.",
    )
    markup(
        """
        <div class="hit-figures">
        <div class="hit-figure"><b>0.9035</b><p>BERTScore F1 against the reference answers</p></div>
        <div class="hit-figure"><b>90%</b><p>Answers judged fully correct, 93% counting partials</p></div>
        <div class="hit-figure"><b>9.92</b><p>Faithfulness out of 10 — how well answers stay inside the retrieved text</p></div>
        </div>
        """
    )

    with st.expander("Full evaluation detail"):
        markup(
            """
            <dl class="hit-stack" style="margin-top:0">
            <div class="hit-stack-row"><dt>BERTScore</dt><dd>Precision 0.8768 · Recall 0.9322 · F1 0.9035</dd></div>
            <div class="hit-stack-row"><dt>Judge averages</dt><dd>Correctness 9.60 · Completeness 9.38 · Faithfulness 9.92 · Overall 9.63</dd></div>
            <div class="hit-stack-row"><dt>Outcomes</dt><dd>45 fully correct · 3 partially correct · 2 returned no answer</dd></div>
            <div class="hit-stack-row"><dt>Caveat</dt><dd>The judge is itself a language model, so read these as indicative rather than authoritative.</dd></div>
            </dl>
            """
        )

    with st.expander("Policy areas in the knowledge base"):
        markup(
            """
            <dl class="hit-stack" style="margin-top:0">
            <div class="hit-stack-row"><dt>Getting in</dt><dd>Admissions and cutoffs · tuition fees and refunds · scholarships and financial aid</dd></div>
            <div class="hit-stack-row"><dt>Academics</dt><dd>Credit system and registration · examinations and grading · attendance and condonation · probation and detention</dd></div>
            <div class="hit-stack-row"><dt>Campus life</dt><dd>Hostel and mess · library rules and fines · code of conduct · anti-ragging and student safety · sports and clubs · IT services and Wi-Fi</dd></div>
            <div class="hit-stack-row"><dt>Beyond the degree</dt><dd>Placements and training cell · internships and NOC · research and final-year projects · entrepreneurship and incubation · alumni and mentorship</dd></div>
            <div class="hit-stack-row"><dt>Process</dt><dd>Grievance redressal · convocation and graduation</dd></div>
            </dl>
            """
        )

    footer_note()



# PAGE 2 — LIVE DEMO


def normalise_response(raw):
    """
    Presentation-only adapter for whatever ask_question() returns.

    Accepts a plain string, an (answer, sources) pair, or a mapping with an
    answer key and optional source documents. Nothing here changes the RAG
    behaviour — it only decides how the result is displayed.
    """
    answer, sources = "", []

    if isinstance(raw, str):
        answer = raw
    elif isinstance(raw, dict):
        for key in ("answer", "result", "output_text", "response", "text"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                answer = value
                break
        else:
            answer = str(raw)
        for key in ("sources", "source_documents", "context", "documents", "citations"):
            value = raw.get(key)
            if isinstance(value, (list, tuple)) and value:
                sources = list(value)
                break
    elif isinstance(raw, (tuple, list)) and len(raw) == 2 and isinstance(raw[0], str):
        answer, maybe_sources = raw
        if isinstance(maybe_sources, (list, tuple)):
            sources = list(maybe_sources)
    else:
        answer = str(raw)

    return answer.strip(), sources


def describe_source(item, index: int):
    """Turn one retrieved item into (label, snippet) for display."""
    text = getattr(item, "page_content", None)
    meta = getattr(item, "metadata", None)

    if text is None and isinstance(item, dict):
        text = item.get("page_content") or item.get("content") or item.get("text") or ""
        meta = item.get("metadata") or item
    if text is None:
        text = str(item)
    if not isinstance(meta, dict):
        meta = {}

    name = meta.get("source") or meta.get("file_path") or meta.get("title") or "Knowledge base"
    name = str(name).replace("\\", "/").split("/")[-1]
    page = meta.get("page")
    label = f"{index:02d} · {name}"
    if page is not None:
        try:
            label += f" · p.{int(page) + 1}"
        except (TypeError, ValueError):
            label += f" · p.{page}"

    snippet = " ".join(str(text).split())
    if len(snippet) > 700:
        snippet = snippet[:700].rstrip() + " …"
    return label, snippet


def use_example(question: str) -> None:
    """Callback: load an example into the input and clear the previous result."""
    st.session_state[QUESTION_KEY] = question
    reset_result()


def reset_result() -> None:
    st.session_state[RESULT_KEY] = None
    st.session_state[ERROR_KEY] = None
    st.session_state[NOTICE_KEY] = None


def clear_all() -> None:
    st.session_state[QUESTION_KEY] = ""
    reset_result()


def render_result() -> None:
    """Draw whichever state the demo is currently in."""
    notice = st.session_state.get(NOTICE_KEY)
    error = st.session_state.get(ERROR_KEY)
    result = st.session_state.get(RESULT_KEY)

    if notice:
        markup(f'<div class="hit-section-tight"><div class="hit-notice">{html.escape(notice)}</div></div>')
        return

    if error:
        markup(
            '<div class="hit-section-tight"><div class="hit-notice hit-error">'
            "<strong>The pipeline could not answer that.</strong><br>"
            "The question was not answered, and nothing was generated in its place. "
            "Check that the Chroma database exists and that GROQ_API_KEY is set, then try again."
            "</div></div>"
        )
        with st.expander("Technical detail"):
            st.code(error, language="text")
        return

    if not result:
        markup(
            '<div class="hit-section-tight"><div class="hit-notice">'
            "Nothing asked yet. Pick one of the examples above or write your own question — "
            "answers come only from the indexed policy documents."
            "</div></div>"
        )
        return

    answer = result.get("answer", "")
    if not answer:
        markup(
            '<div class="hit-section-tight"><div class="hit-notice">'
            "<strong>The pipeline returned an empty answer.</strong><br>"
            "This usually means retrieval found nothing usable for that phrasing. Try asking it differently."
            "</div></div>"
        )
        return

    markup('<div class="hit-section-tight"></div>')
    with st.container(border=True):
    
        markup('<span class="hit-marker"></span>')
        markup(f'<p class="hit-question-echo">Answer </p>')
        st.markdown(answer)
        markup(
            f'<p class="hit-meta">{result["elapsed"]:.1f}s · retrieval {PIPELINE["retrieval"]} '
            f'· {PIPELINE["model"]}</p>'
        )

    sources = result.get("sources") or []
    if sources:
        with st.expander(f"Retrieved passages ({len(sources)})"):
            rows = []
            for i, item in enumerate(sources, start=1):
                label, snippet = describe_source(item, i)
                rows.append(
                    f'<div class="hit-source"><p>{html.escape(label)}</p>'
                    f"<pre>{html.escape(snippet)}</pre></div>"
                )
            markup("".join(rows))

    st.button("Ask another question", on_click=clear_all)


def page_demo() -> None:
    st.session_state.setdefault(QUESTION_KEY, "")

    markup(
        """
        <p class="hit-eyebrow">Live demo</p>
        <div class="hit-accent-rule"></div>
        <h1 class="hit-display hit-wide">Put a question to the knowledge base.</h1>
        <p class="hit-lede">Ask about attendance, grading, fees, hostel rules, placements or any of the
        other 20 policy areas. The assistant retrieves the relevant passages first and answers from
        them — if the documents do not cover your question, it will tell you rather than guess.</p>
        """
    )

    section("Try one of these", tight=True)
    columns = st.columns(2)
    for i, (label, question) in enumerate(EXAMPLES):
        with columns[i % 2]:
            st.button(
                label,
                key=f"example_{i}",
                use_container_width=True,
                on_click=use_example,
                args=(question,),
            )

    markup('<div class="hit-section-tight"></div>')
    with st.form("hit_ask_form", clear_on_submit=False, border=False):
        st.text_area(
            "Your question",
            key=QUESTION_KEY,
            height=110,
            placeholder="What is the minimum attendance required to sit for an examination?",
            label_visibility="collapsed",
        )
        markup('<p class="hit-hint">Ctrl / ⌘ + Enter also submits</p>')
        submit_col, _ = st.columns([1, 3])
        with submit_col:
            submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

    if submitted:
        handle_submit()

    render_result()
    section(
        "Add to knowledge base",
        "Upload a PDF to expand the existing knowledge base.",
        tight=True,
    )

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if st.button("Add Document", type="primary"):
            project_root = Path(__file__).resolve().parent
            knowledge_base = project_root / "knowledge_base"
            knowledge_base.mkdir(exist_ok=True)

            file_path = knowledge_base / uploaded_file.name

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Adding document to the knowledge base..."):
                try:
                    Result= add_document(file_path)

                    if Result["status"]=="duplicate":
                        st.warning("The document is already present. ")

                    else:
                        st.success(
                            f"{uploaded_file.name} added successfully. "
                            f"{Result['chunks']} chunks indexed. "
                        )
                except Exception as exc:
                    st.error(f"Could not add document: {type(exc).__name__}: {exc}")
    footer_note()


def handle_submit() -> None:
    """Validate the input, call the RAG pipeline, and store the outcome."""
    reset_result()
    question = (st.session_state.get(QUESTION_KEY) or "").strip()

    if not question:
        st.session_state[NOTICE_KEY] = "Write a question first — then Ask searches the knowledge base."
        return

    started = time.perf_counter()
    with st.spinner("Searching the knowledge base and drafting an answer…"):
        try:
            raw = ask_question(question)  # Call goes to app.py
        except Exception as exc:  
            st.session_state[ERROR_KEY] = f"{type(exc).__name__}: {exc}"
            return

    answer, sources = normalise_response(raw)
    st.session_state[RESULT_KEY] = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "elapsed": time.perf_counter() - started,
    }


 
# PAGE 3 — CONTACT


def contact_row(label: str, value: str, href: str = "", display: str = "", hint: str = "") -> str:
    target = href or value
    link_text = display or label
    inner = f'<a href="{html.escape(target)}">{html.escape(link_text)}</a>'

    return f'<div class="hit-contact-row"><dt>{label}</dt><dd>{inner}</dd></div>'


def page_contact() -> None:
    markup(
        f"""
        <p class="hit-eyebrow">Contact</p>
        <div class="hit-accent-rule"></div>
        <h1 class="hit-display hit-wide">{html.escape(CONTACT['name'])}</h1>
        <p class="hit-lede">I build retrieval and evaluation pipelines in Python. This project is one of
        them: a knowledge base of institutional documents, a persistent vector store, and a 50-question
        benchmark used to check that the answers hold up. Happy to talk about the retrieval design,
        the evaluation setup, or anything else on this site.</p>
        <p class="hit-eyebrow" style="margin-top:2.5rem">{html.escape(CONTACT['focus'])}</p>
        """
    )

    email = CONTACT["email"]
    markup(
        "<dl class='hit-contact'>"
        + contact_row("LinkedIn", CONTACT["linkedin"], display="View Profile")
        + contact_row("GitHub", CONTACT["github"], display="View Profile")
        + contact_row("Email", email, href=f"mailto:{email}" if email else "", display=email)
        + contact_row("This project", CONTACT["repo"], display="University-RAG")
        + "</dl>"
    )


# NAVIGATION


def make_page(view, title: str, icon: str):
    """Material icons need a recent Streamlit; fall back to no icon if unsupported."""
    return st.Page(view, title=title)


st.markdown(THEME, unsafe_allow_html=True)
sidebar_panel()

pg = st.navigation(
    [
        make_page(page_about, "About Project", ":material/article:"),
        make_page(page_demo, "Live Demo", ":material/forum:"),
        make_page(page_contact, "Contact", ":material/alternate_email:"),
    ]
)
pg.run()
