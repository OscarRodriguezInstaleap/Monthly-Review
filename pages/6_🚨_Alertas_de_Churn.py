import streamlit as st
import pandas as pd
from components.data import build_unified_long
from components.filters import filters_ui
from components.alerts import churn_alerts

st.title("Alertas de Churn")
st.caption("Señales tempranas combinando MRR, Transacciones y CIHS.")

uploaded = st.sidebar.file_uploader("Sube tu archivo (.csv o .xlsx)", type=["csv","xlsx","xls"], key="p6u")
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
alerts = churn_alerts(filtered)
if alerts.empty:
    st.caption("Sin alertas según las reglas actuales.")
else:
    st.dataframe(alerts.sort_values(["riesgo","ultimo_mrr"], ascending=[True, False]), use_container_width=True)
