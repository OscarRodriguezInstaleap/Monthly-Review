
import streamlit as st, pandas as pd, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from components.sources import load_from_public_sheets
DEFAULT_SHEETS_URL = "https://docs.google.com/spreadsheets/d/1ACX9OWNB0vHs8EpxeHxgByuPjDP9VC0E3k9b61V-i1I/edit?usp=sharing"
if "unified_df" not in st.session_state or st.session_state["unified_df"] is None or st.session_state["unified_df"].empty:
    try:
        st.session_state["unified_df"] = load_from_public_sheets(DEFAULT_SHEETS_URL, ("MRR","CIHS","Transactions"))
    except Exception as e:
        st.error(f"No se pudo cargar Google Sheets por defecto: {e}")
        st.stop()
css_file = ROOT / "assets" / "style.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)

from components.filters import filters_ui
from components.cihs import cihs_section

st.title("CIHS — Adopción"); st.caption("Engagement por funcionalidad y clientes.")

unified = st.session_state.get("unified_df")
filtered, _ = filters_ui(unified)
cihs_section(filtered)
