
import pandas as pd
import streamlit as st
from components.data import build_unified_long

def _extract_sheet_id(url_or_id: str) -> str:
    if "/d/" in url_or_id:
        return url_or_id.split("/d/")[1].split("/")[0]
    return url_or_id

@st.cache_data(show_spinner=True)
def load_from_public_sheets(url_or_id: str, sheet_names=("MRR", "CIHS", "Transactions")):
    sheet_id = _extract_sheet_id(url_or_id)
    base = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet="
    tables = {}
    for name in sheet_names:
        url = base + name
        tables[name] = pd.read_csv(url)
    return build_unified_long(tables)
