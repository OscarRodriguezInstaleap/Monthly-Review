
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
from components.forecast import make_forecast, plot_series_with_forecast

st.title("Transacciones"); st.caption("Histórico y pronósticos de transacciones.")

unified = st.session_state.get("unified_df")
filtered, _ = filters_ui(unified)

grouping = st.multiselect("Agrupar por", options=["zona","type","nuevo"], default=["zona","type"], key="p2g")
horizon = st.slider("Horizonte de pronóstico (meses)", 1, 12, 6, key="p2h")
aggregate_first = st.radio("Método", ["Agregue y pronostique", "Pronostique por cliente y agregue"], index=0, horizontal=True, key="p2m")
aggregate_first = (aggregate_first == "Agregue y pronostique")

if (filtered["metric"]=="Transactions").any():
    show_tx_fc = st.checkbox("Mostrar pronóstico de Transacciones", value=True)
    hist_tx, fcst_tx = make_forecast(filtered, metric="Transactions", horizon=horizon if show_tx_fc else 1, aggregate_first=aggregate_first, by=grouping if grouping else None)
    if not show_tx_fc: import pandas as pd; fcst_tx = pd.DataFrame()
    st.plotly_chart(plot_series_with_forecast(hist_tx, fcst_tx, "Transacciones"), use_container_width=True)
else:
    st.caption("No se encontraron datos de Transacciones bajo los filtros actuales.")
