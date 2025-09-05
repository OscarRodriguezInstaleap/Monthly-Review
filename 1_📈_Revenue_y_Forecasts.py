import streamlit as st
import pandas as pd
from components.data import build_unified_long
from components.filters import filters_ui
from components.forecast import make_forecast, plot_series_with_forecast

st.set_page_config(page_title="Revenue & Forecasts", layout="wide")

st.title("Revenue & Forecasts")
st.caption("Análisis detallado de MRR y pronósticos.")

uploaded = st.sidebar.file_uploader("Sube tu archivo (.csv o .xlsx)", type=["csv","xlsx","xls"], key="p1u")
horizon = st.sidebar.slider("Horizonte (meses)", 1, 12, 6, key="p1h")
aggregate_first = st.sidebar.radio("Método", ["Agregue y pronostique", "Pronostique por cliente y agregue"], index=0, key="p1m")
aggregate_first = (aggregate_first == "Agregue y pronostique")
grouping = st.sidebar.multiselect("Agrupar por", options=["zona","type","nuevo"], default=["zona","type"], key="p1g")

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
filtered, selected_clients = filters_ui(unified)

hist_df, fcst_df = make_forecast(filtered, metric="MRR", horizon=horizon, aggregate_first=aggregate_first, by=grouping if grouping else None)
st.plotly_chart(plot_series_with_forecast(hist_df, fcst_df, "MRR Histórico + Pronóstico"), use_container_width=True)

if selected_clients:
    st.subheader("Pronóstico por cliente(s)")
    max_clients = 12; show_clients = selected_clients[:max_clients]
    if len(selected_clients) > max_clients:
        st.caption(f"Mostrando los primeros {max_clients} clientes seleccionados.")
    for cli in show_clients:
        cli_df = filtered[filtered["cliente"]==cli]
        h, f = make_forecast(cli_df, metric="MRR", horizon=horizon, aggregate_first=True, by=[])
        st.plotly_chart(plot_series_with_forecast(h, f, f"MRR - {cli}"), use_container_width=True)
