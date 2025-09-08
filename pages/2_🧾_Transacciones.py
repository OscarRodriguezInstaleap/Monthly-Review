import streamlit as st
import pandas as pd
from components.data import build_unified_long
from components.filters import filters_ui
from components.forecast import make_forecast, plot_series_with_forecast

st.title("Transacciones")
st.caption("Histórico y pronósticos de transacciones.")

uploaded = st.sidebar.file_uploader("Sube tu archivo (.csv o .xlsx)", type=["csv","xlsx","xls"], key="p2u")
horizon = st.sidebar.slider("Horizonte (meses)", 1, 12, 6, key="p2h")
aggregate_first = st.sidebar.radio("Método", ["Agregue y pronostique", "Pronostique por cliente y agregue"], index=0, key="p2m")
aggregate_first = (aggregate_first == "Agregue y pronostique")
grouping = st.sidebar.multiselect("Agrupar por", options=["zona","type","nuevo"], default=["zona","type"], key="p2g")

if uploaded is None:
    st.info("Sube un archivo para comenzar."); st.stop()
xls = pd.ExcelFile(uploaded) if uploaded.name.lower().endswith(('.xlsx','.xls')) else None
tables = {}
if xls is None:
    tables["MRR"] = pd.read_csv(uploaded)
else:
    for sheet in xls.sheet_names:
        tables[sheet] = pd.read_excel(uploaded, sheet_name=sheet)
unified = build_unified_long(tables)
filtered, _ = filters_ui(unified)

if (filtered["metric"]=="Transactions").any():
    show_tx_fc = st.checkbox("Mostrar pronóstico de Transacciones", value=True)
    hist_tx, fcst_tx = make_forecast(filtered, metric="Transactions", horizon=horizon if show_tx_fc else 1, aggregate_first=aggregate_first, by=grouping if grouping else None)
    if not show_tx_fc: fcst_tx = pd.DataFrame()
    st.plotly_chart(plot_series_with_forecast(hist_tx, fcst_tx, "Transacciones"), use_container_width=True)
else:
    st.caption("No se encontraron datos de Transacciones bajo los filtros actuales.")
