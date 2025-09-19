# components/sources.py (v5)

import pandas as pd
import streamlit as st
from components.data import build_unified_long

def _extract_sheet_id(url_or_id: str) -> str:
    """Acepta URL completa o solo el ID de Google Sheets y devuelve el ID."""
    if "/d/" in url_or_id:
        return url_or_id.split("/d/")[1].split("/")[0]
    return url_or_id.strip()

@st.cache_data(show_spinner=True, ttl=60)
def load_from_public_sheets(url_or_id: str, sheet_names=("MRR", "CIHS", "Transactions")):
    """
    Lee varias hojas públicas de Google Sheets como CSV y devuelve el dataframe unificado (long).
    - La hoja debe tener acceso: 'Cualquiera con el enlace' (viewer).
    - Usa cache con TTL=60s. Puedes forzar recarga con load_from_public_sheets.clear()
      (ya lo hace el botón 'Refrescar datos' en app.py).
    """
    sheet_id = _extract_sheet_id(url_or_id)
    base = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet="
    tables = {}
    for name in sheet_names:
        url = base + name
        # Si tu cabecera empieza en otra fila, ajusta read_csv con skiprows.
        tables[name] = pd.read_csv(url)
    return build_unified_long(tables)
