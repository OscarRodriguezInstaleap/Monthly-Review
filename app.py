import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
def _add_components_to_syspath():
    candidate_dirs = [ROOT]
    candidate_dirs += [p for p in ROOT.iterdir() if p.is_dir()]
    if ROOT.parent != ROOT:
        candidate_dirs.append(ROOT.parent)
        candidate_dirs += [p for p in ROOT.parent.iterdir() if p.is_dir()]
    for base in candidate_dirs:
        if (base / "components" / "__init__.py").exists():
            if str(base) not in sys.path:
                sys.path.insert(0, str(base))
            return True
    return False
if not _add_components_to_syspath():
    st.error("No se encontró la carpeta 'components' con __init__.py. Verifica estructura.")
    st.stop()

from components.data import build_unified_long
from components.filters import filters_ui
from components.forecast import make_forecast, plot_series_with_forecast
from components.cohort import yoy_cohort_stacked
from components.cihs import cihs_section
from components.alerts import churn_alerts
from components.map import map_by_country
from components.sources import load_from_public_sheets

DEFAULT_SHEETS_URL = "https://docs.google.com/spreadsheets/d/1ACX9OWNB0vHs8EpxeHxgByuPjDP9VC0E3k9b61V-i1I/edit?usp=sharing"

st.set_page_config(page_title="Instaleap BI — Revenue & Health", layout="wide")

css_file = ROOT / "assets" / "style.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)

st.sidebar.title("Configuración")
data_source = st.sidebar.selectbox("Fuente de datos", ["Google Sheets (público)", "Archivo"], index=0)

if data_source == "Google Sheets (público)":
    url_or_id = st.sidebar.text_input("URL o ID de Google Sheets", value=DEFAULT_SHEETS_URL, help="Debe tener acceso 'Cualquiera con el enlace'")
    sheet_list = st.sidebar.text_input("Nombres de hojas (separadas por coma)", value="MRR,CIHS,Transactions")
    if "unified_df" not in st.session_state and url_or_id:
        try:
            names = tuple([s.strip() for s in sheet_list.split(",") if s.strip()])
            st.session_state["unified_df"] = load_from_public_sheets(url_or_id, names)
        except Exception as e:
            st.error(f"Ocurrió un error al leer Google Sheets: {e}")
            st.stop()
if st.sidebar.button("Refrescar datos", use_container_width=True):
    try:
        load_from_public_sheets.clear()  # <- limpia caché de esa función
        names = tuple([s.strip() for s in sheet_list.split(",") if s.strip()])
        st.session_state["unified_df"] = load_from_public_sheets(url_or_id, names)
    except Exception as e:
        st.error(f"Ocurrió un error al leer Google Sheets: {e}")
        st.stop()
else:
    uploaded = st.sidebar.file_uploader("Sube tu archivo (.csv o .xlsx)", type=["csv","xlsx","xls"])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(('.xlsx', '.xls')):
                tables = pd.read_excel(uploaded, sheet_name=None)
            else:
                tables = {"MRR": pd.read_csv(uploaded)}
            st.session_state["unified_df"] = build_unified_long(tables)
        except Exception as e:
            st.error(f"Ocurrió un error al leer el archivo: {e}")
            st.stop()

horizon = st.sidebar.slider("Horizonte de pronóstico (meses)", 1, 12, 6)
aggregate_first = st.sidebar.radio("Método de consolidación", ["Agregue y pronostique", "Pronostique por cliente y agregue"], index=0)
aggregate_first = (aggregate_first == "Agregue y pronostique")
grouping = st.sidebar.multiselect("Agrupar resultados por", options=["zona","type","nuevo"], default=["zona","type"])

st.title("📊 Instaleap — Revenue & Health Forecaster")
st.caption("MRR · CIHS · Transacciones · Cohortes · País · Alertas de churn")

unified = st.session_state.get("unified_df")
if unified is None or unified.empty:
    st.info("Configura la fuente de datos y presiona **Refrescar datos**.")
    st.stop()

filtered, selected_clients = filters_ui(unified)

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
