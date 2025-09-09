import streamlit as st
import pandas as pd
import numpy as np
from components.data import build_unified_long
from components.filters import filters_ui
from components.forecast import make_forecast, plot_series_with_forecast
from components.cohort import yoy_cohort_stacked
from components.cihs import cihs_section
from components.alerts import churn_alerts
from components.map import map_by_country

from pathlib import Path
ASSETS = Path(__file__).resolve().parent / "assets"
css_file = ASSETS / "style.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)
else:
    st.warning("No se encontró assets/style.css (revisa que la carpeta exista en el repo y esté en la raíz).")

st.set_page_config(page_title="Instaleap BI — Revenue & Health", layout="wide")
st.markdown('<style>' + open('assets/style.css').read() + '</style>', unsafe_allow_html=True)

st.sidebar.title("Configuración")
uploaded = st.sidebar.file_uploader("Sube tu archivo (.csv o .xlsx)", type=["csv","xlsx","xls"])
horizon = st.sidebar.slider("Horizonte de pronóstico (meses)", 1, 12, 6)
aggregate_first = st.sidebar.radio("Método de consolidación", ["Agregue y pronostique", "Pronostique por cliente y agregue"], index=0)
aggregate_first = (aggregate_first == "Agregue y pronostique")
grouping = st.sidebar.multiselect("Agrupar resultados por", options=["zona","type","nuevo"], default=["zona","type"])

st.title("📊 Instaleap — Revenue & Health Forecaster")
st.caption("MRR · CIHS · Transacciones · Cohortes · País · Alertas de churn")

if uploaded is None:
    st.info("Sube un archivo para comenzar. Para Excel, usa hojas llamadas 'MRR', 'CIHS' y 'Transactions'.")
    st.stop()

# Load and normalize
xls = pd.ExcelFile(uploaded) if uploaded.name.lower().endswith(('.xlsx','.xls')) else None
tables = {}
if xls is None:
    tables["MRR"] = pd.read_csv(uploaded)
else:
    for sheet in xls.sheet_names:
        tables[sheet] = pd.read_excel(uploaded, sheet_name=sheet)

unified = build_unified_long(tables)
if unified.empty:
    st.error("No se pudieron leer datos. Verifica la estructura del archivo.")
    st.stop()

filtered, selected_clients = filters_ui(unified)

# KPIs overview
m = filtered[filtered["metric"]=="MRR"].copy()
if m.empty:
    st.info("No hay datos de MRR para KPIs.")
else:
    agg = m.groupby("period", as_index=False)["value"].sum().sort_values("period")
    last_p = agg["period"].max()
    periods = sorted(agg["period"].unique().tolist())
    idx = periods.index(last_p) if last_p in periods else -1
    prev_p = periods[idx-1] if idx-1>=0 else None
    yoy_p = str((pd.Period(last_p) - 12)) if last_p else None

    total_last = float(agg[agg["period"]==last_p]["value"].sum()) if last_p else 0.0
    total_prev = float(agg[agg["period"]==prev_p]["value"].sum()) if prev_p else 0.0
    total_yoy_prev = float(agg[agg["period"]==yoy_p]["value"].sum()) if yoy_p else 0.0

    mom = ((total_last - total_prev) / total_prev * 100.0) if total_prev>0 else None
    yoy = ((total_last - total_yoy_prev) / total_yoy_prev * 100.0) if total_yoy_prev>0 else None
    arr = total_last * 12.0

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric(f"MRR Total ({last_p})", f"${total_last:,.0f}")
    c2.metric("MoM", f"{mom:,.1f}%" if mom is not None else "—")
    c3.metric("YoY", f"{yoy:,.1f}%" if yoy is not None else "—")
    c4.metric("ARR", f"${arr:,.0f}")
    active_clients = int((m.groupby("cliente")["value"].sum()>0).sum())
    c5.metric("Clientes activos", active_clients)

st.markdown("### Pronóstico MRR consolidado")
hist_df, fcst_df = make_forecast(filtered, metric="MRR", horizon=horizon, aggregate_first=aggregate_first, by=grouping if grouping else None)
st.plotly_chart(plot_series_with_forecast(hist_df, fcst_df, "MRR Histórico + Pronóstico"), use_container_width=True)

st.markdown("### CIHS global")
cihs_section(unified=filtered)

st.markdown("### Cohortes principales")
yoy_cohort_stacked(filtered)

st.markdown("### Mapa por país")
map_by_country(unified=filtered, make_forecast=make_forecast, plot_series_with_forecast=plot_series_with_forecast)

st.markdown("### Alertas de Churn")
alerts = churn_alerts(filtered)
if alerts.empty:
    st.caption("Sin alertas según las reglas actuales.")
else:
    st.dataframe(alerts.sort_values(["riesgo","ultimo_mrr"], ascending=[True, False]), use_container_width=True)
