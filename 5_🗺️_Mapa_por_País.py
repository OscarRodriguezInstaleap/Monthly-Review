import streamlit as st
import pandas as pd
from components.data import build_unified_long
from components.filters import filters_ui
from components.map import map_by_country
from components.forecast import make_forecast, plot_series_with_forecast

st.title("Mapa por País")
st.caption("Zoom / pan + clic para ver KPIs y forecast por país.")

uploaded = st.sidebar.file_uploader("Sube tu archivo (.csv o .xlsx)", type=["csv","xlsx","xls"], key="p5u")
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
map_by_country(filtered, make_forecast, plot_series_with_forecast)
