"""
Front end for the Hyderabad Institute of Technology RAG assistant.

"""

from __future__ import annotations

import html
import re
import textwrap
import time

import streamlit as st

from app import ask_question 
from pathlib import Path
from add_document import add_document 
from delete_document import delete_document



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

# UI-only state added for the loading indicator and the upload / delete resets.
PENDING_KEY = "hit_pending"
UPLOAD_ROUND_KEY = "hit_upload_round"
UPLOAD_FLASH_KEY = "hit_upload_flash"
DELETE_ROUND_KEY = "hit_delete_round"
DELETE_FLASH_KEY = "hit_delete_flash"

DELETE_PLACEHOLDER = "Select PDF to remove"

# Upload cap, checked before the file is written to knowledge_base/. The caption
# under the uploader is derived from this same value, so the number shown and the
# number enforced can never drift apart. Set to None to drop the check and the
# caption together (Streamlit's own server.maxUploadSize still applies).
MAX_UPLOAD_MB = 10

st.set_page_config(
    page_title="HIT Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="locked",
)


def knowledge_base_dir() -> Path:
    """Folder the PDFs live in — same path the add / delete flows already use."""
    return Path(__file__).resolve().parent / "knowledge_base"


def pdf_sort_key(path: Path):
    """
    Deterministic ascending order for display.

    Splitting on digit runs means 2_fees sorts before 10_hostel instead of after
    it, which plain lexicographic sorting gets wrong. Presentation only — nothing
    about how PDFs are stored, indexed or deleted depends on this.
    """
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def list_knowledge_base_pdfs() -> list[Path]:
    """Whatever PDFs exist right now. No fixed count is assumed anywhere."""
    directory = knowledge_base_dir()
    if not directory.exists():
        return []
    files = list(directory.glob("*.pdf"))
    try:
        return sorted(files, key=pdf_sort_key)
    except TypeError:
        return sorted(files)


def document_count_label() -> str:
    count = len(list_knowledge_base_pdfs())
    return f"{count} PDF document" if count == 1 else f"{count} PDF documents"


def upload_caption() -> str:
    """Caption under the uploader — always the limit that is actually enforced."""
    if MAX_UPLOAD_MB is None:
        return "PDF only"
    return f"PDF only · up to {MAX_UPLOAD_MB} MB"


def oversized(uploaded_file) -> bool:
    """True when the file is past the cap above. Unknown size never blocks."""
    if MAX_UPLOAD_MB is None:
        return False
    size = getattr(uploaded_file, "size", None)
    if not isinstance(size, int):
        return False
    return size > MAX_UPLOAD_MB * 1024 * 1024



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
  --amber: #B47C1E;
  --amber-soft: #FBF4E7;
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

/* ---------- sidebar reachability (kept visible on every screen size) ---------- */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  z-index: 999;
}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button,
[data-testid="stExpandSidebarButton"] button,
[data-testid="stSidebarCollapseButton"] button {
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 6px;
  color: var(--ink);
}
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="collapsedControl"] button:hover,
[data-testid="stExpandSidebarButton"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover { border-color: var(--ink); }

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
.stButton > button:disabled, .stButton > button:disabled:hover {
  background: var(--rule-soft);
  border-color: var(--rule);
  color: var(--slate-soft);
  cursor: not-allowed;
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

/* ---------- select ---------- */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background: var(--card);
  border-color: var(--rule);
  border-radius: 8px;
  font-family: var(--sans);
  font-size: .95rem;
  color: var(--ink);
  min-height: 2.6rem;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover { border-color: var(--slate-soft); }
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
  border-color: var(--verdigris);
  box-shadow: 0 0 0 3px var(--verdigris-soft);
}
[data-testid="stSelectbox"] [data-baseweb="select"] div[value] { overflow-wrap: anywhere; }
/* Chevron and clear control, kept in the same grey family as the rest of the chrome. */
[data-testid="stSelectbox"] [data-baseweb="select"] svg { color: var(--slate-soft); fill: var(--slate-soft); }
[data-testid="stSelectbox"] [data-baseweb="select"]:hover svg { color: var(--ink); fill: var(--ink); }
div[data-baseweb="popover"] [role="listbox"],
div[data-baseweb="popover"] ul {
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 8px;
  box-shadow: 0 6px 18px rgba(14,26,43,.08);
  padding: .25rem;
}
div[data-baseweb="popover"] li {
  font-family: var(--sans);
  font-size: .92rem;
  color: var(--ink);
  border-radius: 6px;
  overflow-wrap: anywhere;
}
div[data-baseweb="popover"] li:hover { background: var(--rule-soft); }
div[data-baseweb="popover"] li[aria-selected="true"] { background: var(--oxblood-soft); color: var(--ink); }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
  gap: .3rem;
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 10px;
  padding: .35rem;
  margin-top: 2.25rem;
  margin-bottom: .5rem;
  overflow-x: auto;
  scrollbar-width: none;
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
.stTabs [data-baseweb="tab"] {
  height: auto;
  min-height: 0;
  padding: .6rem 1.2rem;
  border-radius: 7px;
  background: transparent;
  color: var(--slate);
  white-space: nowrap;
  transition: background .15s ease;
}
.stTabs [data-baseweb="tab"] p {
  font-family: var(--sans);
  font-size: .96rem;
  font-weight: 500;
  margin: 0;
  color: var(--slate);
}
.stTabs [data-baseweb="tab"]:hover { background: var(--rule-soft); }
.stTabs [data-baseweb="tab"]:hover p { color: var(--ink); }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
  background: var(--oxblood-soft);
  box-shadow: inset 0 -2px 0 var(--oxblood);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] p { color: var(--ink); font-weight: 600; }
.stTabs [data-baseweb="tab"]:focus-visible { outline: 2px solid var(--verdigris); outline-offset: 2px; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: .25rem; }

/* ---------- file uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
  background: var(--card);
  border: 1px dashed var(--rule);
  border-radius: 10px;
  padding: 1.1rem 1.2rem;
  gap: .9rem;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--verdigris); }
[data-testid="stFileUploaderDropzoneInstructions"] span {
  font-family: var(--sans);
  font-size: .95rem;
  color: var(--ink);
}
/* The stock caption reads "Limit 200MB per file • PDF". It is hidden in both the
   markups Streamlit has used for it; the real limit is printed under the widget
   from MAX_UPLOAD_MB instead, so the number shown is the number enforced. */
[data-testid="stFileUploaderDropzoneInstructions"] small { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] div > span:last-child:not(:first-child) {
  display: none !important;
}
[data-testid="stFileUploaderDropzone"] button {
  font-family: var(--sans);
  font-size: .85rem;
  font-weight: 500;
  border: 1px solid var(--rule);
  border-radius: 6px;
  background: var(--card);
  color: var(--ink);
  white-space: nowrap;
}
[data-testid="stFileUploaderDropzone"] button:hover { border-color: var(--ink); color: var(--ink); }
[data-testid="stFileUploaderFile"] {
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 8px;
  padding: .6rem .8rem;
  margin-top: .6rem;
  min-width: 0;
}
[data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileName"] {
  font-family: var(--sans);
  font-size: .9rem;
  color: var(--ink);
  overflow-wrap: anywhere;
}
[data-testid="stFileUploaderFile"] small { font-family: var(--mono); font-size: .72rem; color: var(--slate-soft); }

/* ---------- alerts ---------- */
[data-testid="stAlertContainer"] {
  border: 1px solid var(--rule);
  border-left: 3px solid var(--slate-soft);
  border-radius: 8px;
  background: var(--card);
  font-family: var(--sans);
  font-size: .93rem;
  line-height: 1.6;
  color: var(--ink);
}
[data-testid="stAlertContainer"] p { font-size: .93rem; color: var(--ink); overflow-wrap: anywhere; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
  border-left-color: var(--verdigris);
  background: var(--verdigris-soft);
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {
  border-left-color: var(--amber);
  background: var(--amber-soft);
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {
  border-left-color: var(--oxblood);
  background: var(--oxblood-soft);
}

/* ---------- expanders and code ---------- */
[data-testid="stExpander"] {
  border: 1px solid var(--rule);
  border-radius: 8px;
  background: var(--card);
  overflow: hidden;
}
[data-testid="stExpander"] details { background: var(--card); border: 0; }
[data-testid="stExpander"] summary {
  background: var(--rule-soft);
  border-bottom: 1px solid var(--rule);
  padding: .7rem 1rem;
  transition: background .15s ease;
}
[data-testid="stExpander"] summary p {
  font-family: var(--mono);
  font-size: .74rem;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--slate);
}
[data-testid="stExpander"] summary:hover { background: var(--verdigris-soft); }
[data-testid="stExpander"] summary:hover p { color: var(--ink); }
[data-testid="stExpander"] summary:focus-visible { outline: 2px solid var(--verdigris); outline-offset: -2px; }
[data-testid="stExpander"] details[open] > summary { background: var(--card); }
/* Chevron on the right, matched to the label rather than left at the Streamlit blue. */
[data-testid="stExpander"] summary svg { color: var(--slate-soft); fill: var(--slate-soft); }
[data-testid="stExpander"] summary:hover svg { color: var(--ink); fill: var(--ink); }
[data-testid="stExpanderDetails"] { background: var(--card); padding: 1rem 1.1rem 1.1rem; }
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

/* ---------- loading state ---------- */
.hit-loading {
  display: flex;
  align-items: flex-start;
  gap: .9rem;
  border: 1px solid var(--rule);
  border-left: 3px solid var(--verdigris);
  border-radius: 8px;
  background: var(--card);
  padding: 1rem 1.2rem;
}
.hit-loading-spin {
  flex: 0 0 auto;
  width: 18px;
  height: 18px;
  margin-top: .15rem;
  border-radius: 50%;
  border: 2px solid var(--rule);
  border-top-color: var(--verdigris);
  animation: hit-spin .8s linear infinite;
}
@keyframes hit-spin { to { transform: rotate(360deg); } }
.hit-loading div { min-width: 0; }
.hit-loading p { margin: 0; font-size: .95rem; line-height: 1.6; color: var(--ink); }
.hit-loading .hit-loading-note {
  font-family: var(--mono);
  font-size: .74rem;
  color: var(--slate-soft);
  margin-top: .35rem;
}

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

/* ---------- pipeline flow ---------- */
.hit-flow {
  display: flex;
  flex-wrap: wrap;
  align-items: stretch;
  gap: .5rem;
  margin-top: 1.5rem;
}
.hit-flow-node {
  flex: 1 1 150px;
  min-width: 0;
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 8px;
  padding: .95rem 1rem;
}
.hit-flow-node span {
  display: block;
  font-family: var(--mono);
  font-size: .66rem;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--slate-soft);
  margin: 0 0 .3rem;
}
.hit-flow-node b { display: block; font-family: var(--sans); font-size: .97rem; font-weight: 600; color: var(--ink); }
.hit-flow-node p { margin: .35rem 0 0; font-size: .83rem; line-height: 1.5; color: var(--slate); }
.hit-flow-node.hit-flow-key { border-color: var(--verdigris); background: var(--verdigris-soft); }
.hit-flow-node.hit-flow-key span { color: var(--verdigris); }
.hit-flow-arrow {
  flex: 0 0 auto;
  align-self: center;
  font-family: var(--mono);
  font-size: .8rem;
  color: var(--slate-soft);
}

/* ---------- paired cards ---------- */
.hit-duo {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 1rem;
  margin-top: 1.5rem;
}
.hit-card {
  background: var(--card);
  border: 1px solid var(--rule);
  border-radius: 10px;
  padding: 1.3rem 1.4rem;
}
.hit-card .hit-eyebrow { margin-bottom: .55rem; }
.hit-card h3 { font-family: var(--serif); font-size: 1.14rem; font-weight: 500; margin: 0 0 .4rem; color: var(--ink); }
.hit-card p { margin: 0; font-size: .9rem; line-height: 1.62; color: var(--slate); }

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
.hit-notice.hit-key { border-left-color: var(--verdigris); background: var(--verdigris-soft); color: var(--ink); }
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
  overflow-wrap: anywhere;
  max-width: 100%;
  margin: 0;
  background: transparent;
}

/* ---------- contact ---------- */
.hit-name {
  font-size: clamp(2.6rem, 8vw, 4.4rem);
  line-height: 1.02;
  letter-spacing: -.025em;
  max-width: none;
  margin: 0 0 1.4rem;
}
.hit-links { border-top: 1px solid var(--rule); margin: 2.25rem 0 0; }
.hit-link-row { border-bottom: 1px solid var(--rule-soft); }
.hit-link-row a {
  display: grid;
  grid-template-columns: minmax(110px, 150px) 1fr auto;
  gap: 1rem;
  align-items: baseline;
  padding: 1.05rem .35rem;
  border-bottom: 0;
  color: var(--ink);
  transition: background .18s ease, padding-left .18s ease;
}
.hit-link-row a:hover { background: var(--rule-soft); padding-left: .75rem; }
.hit-link-row a:focus-visible { outline: 2px solid var(--verdigris); outline-offset: -2px; }
.hit-link-label {
  font-family: var(--mono);
  font-size: .72rem;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--slate-soft);
}
.hit-link-value { font-size: 1rem; color: var(--ink); overflow-wrap: anywhere; }
.hit-link-row a:hover .hit-link-value { color: var(--verdigris); }
.hit-link-arrow {
  font-family: var(--mono);
  font-size: .95rem;
  color: var(--slate-soft);
  transition: transform .18s ease, color .18s ease;
}
.hit-link-row a:hover .hit-link-arrow { color: var(--verdigris); transform: translateX(4px); }
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

/* ---------- overflow guards ---------- */
.stApp img, .stApp table { max-width: 100%; }
[data-testid="stMarkdownContainer"] pre { max-width: 100%; overflow-x: auto; }

@media (max-width: 860px) {
  [data-testid="stSidebar"] { min-width: 20rem; }
  .hit-flow-node { flex: 1 1 210px; }
}

@media (max-width: 640px) {
  .block-container, .stMainBlockContainer { padding-top: 1.75rem; }
  .hit-display { max-width: none; }
  .hit-step { grid-template-columns: 1.9rem 1fr; gap: .75rem; }
  .hit-stack-row { grid-template-columns: 1fr; gap: .2rem; }
  [data-testid="stSidebar"] { min-width: 82vw !important; max-width: 92vw !important; }
  .stTabs [data-baseweb="tab-list"] { margin-top: 1.5rem; }
  .stTabs [data-baseweb="tab"] { padding: .55rem .9rem; }
  .stTabs [data-baseweb="tab"] p { font-size: .9rem; }
  .stButton > button p, .stFormSubmitButton > button p { white-space: normal; overflow-wrap: anywhere; }
  [data-testid="stFileUploaderDropzone"] { flex-direction: column; align-items: flex-start; }
  [data-testid="stFileUploaderDropzone"] button { width: 100%; }
  .hit-record { grid-template-columns: 1fr 1fr; }
  .hit-figures { grid-template-columns: 1fr; }
  .hit-flow { flex-direction: column; }
  .hit-flow-arrow { transform: rotate(90deg); }
  .hit-duo { grid-template-columns: 1fr; }
  .hit-link-row a { grid-template-columns: 1fr auto; gap: .15rem 1rem; }
  .hit-link-label { grid-column: 1 / -1; }
}

@media (max-width: 420px) {
  .hit-record { grid-template-columns: 1fr; }
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
            <dt>Knowledge base</dt><dd>{document_count_label()}</dd>
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
        <div class="hit-accent-rule"></div>
        <h1 class="hit-display">Ask a university's rulebook a question, get the answer it actually contains.</h1>
        <p class="hit-lede">A language model on its own has never read your institution's handbooks.
        This assistant searches documents of university policy first, then answers from
        what it found — and says so plainly when the answer is not in there.</p>
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

    section(
        "From a PDF to an answer",
        "Five stages. The first three run once per document, the last two on every question.",
    )
    markup(
        """
        <div class="hit-flow">
        <div class="hit-flow-node">
        <span>Input</span><b>Documents</b>
        <p>Policy PDFs, uploaded from inside the app.</p>
        </div>
        <span class="hit-flow-arrow">→</span>
        <div class="hit-flow-node"> 
        <span>Check</span><b>SHA-256</b>
        <p>A fingerprint of the file, compared before anything is indexed.</p>
        </div>
        <span class="hit-flow-arrow">→</span>
        <div class="hit-flow-node">
        <span>Store</span><b>Knowledge base</b>
        <p>Chunks and embeddings, kept in Chroma on disk.</p>
        </div>
        <span class="hit-flow-arrow">→</span>
        <div class="hit-flow-node">
        <span>Search</span><b>Retrieval</b>
        <p>MMR and multi-query pull the passages that matter.</p>
        </div>
        <span class="hit-flow-arrow">→</span>
        <div class="hit-flow-node">
        <span>Output</span><b>Answer</b>
        <p>Written from those passages, or not at all.</p>
        </div>
        </div>
        """
    )

    section("How a question is answered", tight=False)
    markup(
        """
        <p class="hit-stage">Stage 1 — Indexing, once per document</p>
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

    section(
        "The knowledge base is not fixed",
        "It is edited from the app itself, on the Manage PDFs tab of the live demo. "
        "Both operations work against the existing index — there is no rebuild step and "
        "nothing to restart.",
    )
    markup(
        """
        <div class="hit-duo">
        <div class="hit-card">
        <p class="hit-eyebrow">Add</p>
        <h3>Upload a PDF</h3>
        <p>A new policy document is chunked, embedded and written into the Chroma store
        alongside everything already there. It is searchable on the next question.</p>
        </div>
        <div class="hit-card">
        <p class="hit-eyebrow">Delete</p>
        <h3>Remove a PDF</h3>
        <p>Pick a document from the list and it leaves the knowledge base along with every
        chunk it contributed, so retrieval can no longer reach it.</p>
        </div>
        </div>
        """
    )
    markup(
        """
        <div class="hit-section-tight">
        <div class="hit-notice hit-key">
        <strong>The same document cannot be added twice.</strong> Before a file is indexed it is
        hashed with SHA-256 — a short fingerprint of its contents. If that fingerprint is already
        in the store, the upload is reported as a duplicate and nothing is written. Renaming a PDF
        does not get it past the check, because the hash is of what is inside the file, not what it
        is called.
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
        <div class="hit-stack-row"><dt>Indexing</dt><dd>Incremental — documents added and removed without a rebuild</dd></div>
        <div class="hit-stack-row"><dt>Deduplication</dt><dd>SHA-256 content hash</dd></div>
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

    markup('<div class="hit-section-tight"></div>')
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
    st.session_state[PENDING_KEY] = None


def clear_all() -> None:
    st.session_state[QUESTION_KEY] = ""
    reset_result()


def render_loading() -> None:
    """The searching state. Stays on screen until the backend call returns."""
    markup(
        """
        <div class="hit-section-tight">
        <div class="hit-loading">
        <div class="hit-loading-spin"></div>
        <div>
        <p>Searching the knowledge base and drafting an answer…</p>
        <p class="hit-loading-note">Retrieval and generation are still running — the answer appears here when they finish.</p>
        </div>
        </div>
        </div>
        """
    )


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
        markup('<p class="hit-question-echo">Answer</p>')
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


# --- Tab 1 — Ask Questions -------------------------------------------------


def render_ask_tab() -> None:
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

    # One slot holds either the loading state or the outcome. Writing the
    # loading state into it clears the previous answer straight away, and the
    # slot is only rewritten once the backend call below has returned.
    result_slot = st.empty()

    pending = st.session_state.get(PENDING_KEY)
    if pending:
        with result_slot.container():
            render_loading()
        run_question(pending)

    with result_slot.container():
        render_result()


def handle_submit() -> None:
    """Validate the input and queue the question for this same script run."""
    reset_result()
    question = (st.session_state.get(QUESTION_KEY) or "").strip()

    if not question:
        st.session_state[NOTICE_KEY] = "Write a question first — then Ask searches the knowledge base."
        return

    st.session_state[PENDING_KEY] = question


def run_question(question: str) -> None:
    """Call the RAG pipeline and store the outcome. Backend call is unchanged."""
    started = time.perf_counter()
    try:
        raw = ask_question(question)  # Call goes to app.py
    except Exception as exc:  
        st.session_state[ERROR_KEY] = f"{type(exc).__name__}: {exc}"
        st.session_state[PENDING_KEY] = None
        return

    answer, sources = normalise_response(raw)
    st.session_state[RESULT_KEY] = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "elapsed": time.perf_counter() - started,
    }
    st.session_state[PENDING_KEY] = None


# --- Tab 2 — Manage PDFs ---------------------------------------------------


def render_manage_tab() -> None:
    section(
        "Add to knowledge base",
        "Upload a PDF to expand the existing knowledge base. Duplicates are caught by "
        "their SHA-256 hash, so the same document cannot be indexed twice.",
        tight=True,
    )

    # The key changes after a successful add, which is what empties the widget.
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key=f"hit_pdf_upload_{st.session_state[UPLOAD_ROUND_KEY]}",
        label_visibility="collapsed",
    )
    

    upload_flash = st.session_state.pop(UPLOAD_FLASH_KEY, None)
    if upload_flash:
        st.success(upload_flash)

    if uploaded_file is not None:
        if oversized(uploaded_file):
            # Checked before anything is written to disk or handed to add_document().
            st.error(
                f"{uploaded_file.name} is over the {MAX_UPLOAD_MB} MB limit. "
                "Upload a smaller PDF."
            )
        elif st.button("Add Document", type="primary"):
            project_root = Path(__file__).resolve().parent
            knowledge_base = project_root / "knowledge_base"
            knowledge_base.mkdir(exist_ok=True)

            file_path = knowledge_base / uploaded_file.name

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            added_message = None
            with st.spinner("Adding document to the knowledge base..."):
                try:
                    Result= add_document(file_path)

                    if Result["status"]=="duplicate":
                        st.warning("The document is already present. ")

                    else:
                        added_message = (
                            f"{uploaded_file.name} added successfully. "
                            f"{Result['chunks']} chunks indexed. "
                        )
                except Exception as exc:
                    st.error(f"Could not add document: {type(exc).__name__}: {exc}")

            # Only the successful path resets the uploader.
            if added_message:
                st.session_state[UPLOAD_FLASH_KEY] = added_message
                st.session_state[UPLOAD_ROUND_KEY] += 1
                st.rerun()

    section(
        "Manage knowledge base",
        "Select a document to remove it from the knowledge base. Listed in ascending order.",
        tight=True,
    )

    pdf_files = list_knowledge_base_pdfs()

    delete_flash = st.session_state.pop(DELETE_FLASH_KEY, None)

    if pdf_files:
        selected_pdf = st.selectbox(
            "Select PDF to remove",
            [None] + pdf_files,
            index=0,
            format_func=lambda path: DELETE_PLACEHOLDER if path is None else path.name,
            key=f"hit_pdf_delete_{st.session_state[DELETE_ROUND_KEY]}",
            label_visibility="collapsed",
        )

        if delete_flash:
            st.success(delete_flash)

        if st.button("Remove Document", disabled=selected_pdf is None) and selected_pdf is not None:
            deleted_message = None
            with st.spinner("Removing document from the knowledge base..."):
                try:
                    result = delete_document(selected_pdf)

                    if result["status"] == "deleted":
                        deleted_message = (
                            f"{selected_pdf.name} removed successfully. "
                            f"{result['chunks']} chunks deleted."
                        )
                    elif result["status"] == "not_found":
                        st.warning(
                            "The document was not found in the vector database."
                        )
                    else:
                        st.error("Could not remove the document.")

                except Exception as exc:
                    st.error(
                        f"Could not remove document: "
                        f"{type(exc).__name__}: {exc}"
                    )

            # Only the successful path resets the selection to the placeholder.
            if deleted_message:
                st.session_state[DELETE_FLASH_KEY] = deleted_message
                st.session_state[DELETE_ROUND_KEY] += 1
                st.rerun()
    else:
        if delete_flash:
            st.success(delete_flash)
        st.info("No PDF documents are currently in the knowledge base.")


def page_demo() -> None:
    st.session_state.setdefault(QUESTION_KEY, "")
    st.session_state.setdefault(PENDING_KEY, None)
    st.session_state.setdefault(UPLOAD_ROUND_KEY, 0)
    st.session_state.setdefault(DELETE_ROUND_KEY, 0)

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

    ask_tab, manage_tab = st.tabs(["💬 Ask Questions", "📄 Manage PDFs"])

    with ask_tab:
        render_ask_tab()

    with manage_tab:
        render_manage_tab()

    footer_note()


 
# PAGE 3 — CONTACT


def link_row(label: str, href: str, display: str) -> str:
    """One line in the contact list: label, value, arrow — the whole row is the link."""
    return (
        '<div class="hit-link-row">'
        f'<a href="{html.escape(href)}">'
        f'<span class="hit-link-label">{html.escape(label)}</span>'
        f'<span class="hit-link-value">{html.escape(display)}</span>'
        '<span class="hit-link-arrow">↗</span>'
        "</a></div>"
    )


def page_contact() -> None:
    markup(
        f"""
        <p class="hit-eyebrow">The person behind the project</p>
        <div class="hit-accent-rule"></div>
        <h1 class="hit-display hit-name">{html.escape(CONTACT['name'])}</h1>
        <p class="hit-lede">I build retrieval and evaluation pipelines in Python. This project is one of
        them: institutional documents in a persistent vector store, and a 50-question benchmark to check
        that the answers hold up. Happy to talk about the retrieval design, the evaluation, or anything
        else on this site.</p>
        <p class="hit-eyebrow" style="margin:2rem 0 0">{html.escape(CONTACT['focus'])}</p>
        """
    )

    email = CONTACT["email"]
    rows = [
        link_row("LinkedIn", CONTACT["linkedin"], "Tanishk Agarwal"),
        link_row("GitHub", CONTACT["github"], "Tanishk-2004"),
        link_row("This project", CONTACT["repo"], "University-RAG"),
    ]
    if email:
        rows.append(link_row("Email", f"mailto:{email}", email))

    markup('<div class="hit-links">' + "".join(rows) + "</div>")

    markup(
        """
        <p class="hit-footer">Built with Streamlit, LangChain and Chroma. The institution is
        fictional; the retrieval pipeline behind it is not.</p>
        """
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