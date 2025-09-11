
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

st.title("Revenue & Forecasts"); st.caption("Análisis detallado de MRR y pronósticos.")

unified = st.session_state.get("unified_df")
filtered, selected_clients = filters_ui(unified)

grouping = st.multiselect("Agrupar por", options=["zona","type","nuevo"], default=["zona","type"], key="p1g")
horizon = st.slider("Horizonte de pronóstico (meses)", 1, 12, 6, key="p1h")
aggregate_first = st.radio("Método", ["Agregue y pronostique", "Pronostique por cliente y agregue"], index=0, horizontal=True, key="p1m")
aggregate_first = (aggregate_first == "Agregue y pronostique")

hist_df, fcst_df = make_forecast(filtered, metric="MRR", horizon=horizon, aggregate_first=aggregate_first, by=grouping if grouping else None)
st.plotly_chart(plot_series_with_forecast(hist_df, fcst_df, "MRR Histórico + Pronóstico"), use_container_width=True)

if selected_clients:
    st.subheader("Pronóstico por cliente(s)")
    max_clients = 12; show_clients = selected_clients[:max_clients]
    if len(selected_clients) > max_clients: st.caption(f"Mostrando los primeros {max_clients} clientes seleccionados.")
    for cli in show_clients:
        cli_df = filtered[filtered["cliente"]==cli]
        h, f = make_forecast(cli_df, metric="MRR", horizon=horizon, aggregate_first=True, by=[])
        st.plotly_chart(plot_series_with_forecast(h, f, f"MRR - {cli}"), use_container_width=True)
