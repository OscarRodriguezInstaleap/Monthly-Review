
import io
import sys
import math
import numpy as np
import re
from datetime import datetime
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Optional

# Optional plotting
import plotly.graph_objects as go
import plotly.express as px

# Forecasting
from prophet import Prophet

st.set_page_config(page_title="Revenue & Health Forecaster", layout="wide")

# -----------------------------
# Helpers
# -----------------------------

@st.cache_data(show_spinner=False)
def load_file(uploaded) -> Dict[str, pd.DataFrame]:
    """
    Load a CSV or Excel (with optional sheets) and return a dict of dataframes keyed by metric.
    Expected sheet names (case-insensitive): 'MRR', 'CIHS', 'Transactions'.
    If CSV, assume MRR.
    """
    if uploaded is None:
        return {}
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
        return {"MRR": df}
    else:
        # Excel: read all sheets
        xls = pd.ExcelFile(uploaded)
        sheets = {}
        for sheet in xls.sheet_names:
            key = sheet.strip().lower()
            # standardize to known keys
            if "mrr" in key:
                sheets["MRR"] = pd.read_excel(uploaded, sheet_name=sheet)
            elif "cihs" in key or "health" in key:
                sheets["CIHS"] = pd.read_excel(uploaded, sheet_name=sheet)
            elif "trans" in key or "tx" in key or "orders" in key:
                sheets["Transactions"] = pd.read_excel(uploaded, sheet_name=sheet)
            else:
                # if unknown, keep as-is using the sheet name
                sheets[sheet] = pd.read_excel(uploaded, sheet_name=sheet)
        return sheets


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower_map = {c: (str(c).strip().lower() if not isinstance(c, (pd.Timestamp, datetime)) else c) for c in df.columns}
    df.rename(columns=lower_map, inplace=True)

    mapping = {
        "type client": "type",
        "tipo cliente": "type",
        "tipo de cliente": "type",
        "nuevo": "nuevo",
        "zona": "zona",
        "cliente": "cliente",
        "client": "cliente",
        "razon social": "razon_social",
        "razón social": "razon_social",
        "razon_social": "razon_social",
    }
    for k,v in mapping.items():
        if k in df.columns:
            df.rename(columns={k:v}, inplace=True)

    return df



def detect_meta_and_time(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    def is_time_header(col):
        if isinstance(col, (pd.Timestamp, datetime)):
            return True
        s = str(col).strip()
        month_regex = re.compile(
            r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
            r"ene|feb|mar|abr|may|jun|jul|ago|sept|sep|oct|nov|dic)"
            r"[a-z]*[-_/ ]?\d{2,4}$", re.IGNORECASE
        )
        if month_regex.match(s):
            return True
        for fmt in ("%b-%y", "%b-%Y", "%B-%y", "%B-%Y"):
            try:
                pd.to_datetime(s, format=fmt)
                return True
            except Exception:
                pass
        try:
            dt = pd.to_datetime(s, errors="raise")
            if s.lower() not in {"type", "zona", "nuevo", "cliente", "razon social", "razón social", "razon_social"}:
                return True
        except Exception:
            return False
        return False

    meta_cols, ts_cols = [], []
    for c in df.columns:
        if is_time_header(c):
            ts_cols.append(c)
        else:
            meta_cols.append(c)
    return meta_cols, ts_cols


def to_long(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    df = standardize_columns(df)
    meta_cols, ts_cols = detect_meta_and_time(df)
    if not ts_cols:
        return pd.DataFrame(columns=meta_cols + ["date","value","metric"])

    # Melt to long
    long_df = df.melt(id_vars=meta_cols, value_vars=ts_cols,
                      var_name="month", value_name="value_raw")
    # Clean numbers
    # remove spaces and thousand separators; respect decimals with dot
    long_df["value"] = (
        long_df["value_raw"].astype(str)
                             .str.replace(",", "", regex=False)
                             .str.replace(" ", "", regex=False)
                             .replace({"": None, "nan": None, "None": None, "NULL": None, "null": None})
    )
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce").fillna(0.0)

    # Parse dates like 'Jan-19'
    # Some spreadsheets may use 'Jan-2019' -> try both
    def _parse_date(x):
        x = str(x)
        for fmt in ("%b-%y", "%b-%Y", "%B-%y", "%B-%Y"):
            try:
                return pd.to_datetime(x, format=fmt)
            except Exception:
                pass
        # fallback: let pandas guess
        try:
            return pd.to_datetime(x)
        except Exception:
            return pd.NaT

    long_df["date"] = long_df["month"].map(_parse_date)
    long_df["metric"] = metric_name
    return long_df

def build_unified_long(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for key, df in tables.items():
        # Normalize key to expected metric names
        key_norm = key.strip().lower()
        if key_norm in ("mrr","revenue"):
            parts.append(to_long(df, "MRR"))
        elif key_norm in ("cihs","health"):
            parts.append(to_long(df, "CIHS"))
        elif key_norm.startswith("trans") or key_norm in ("tx","orders","transactions"):
            parts.append(to_long(df, "Transactions"))
        else:
            # Treat unknown sheets as MRR by default
            parts.append(to_long(df, key))
    if parts:
        out = pd.concat(parts, ignore_index=True)
        # Standardize meta col names one more time
        out = standardize_columns(out)
        # infer missing meta columns
        for col in ["type","zona","nuevo","cliente","razon_social"]:
            if col not in out.columns:
                out[col] = None
        # clean strings
        for col in ["type","zona","nuevo","cliente","razon_social"]:
            out[col] = out[col].astype(str).str.strip()
        return out
    return pd.DataFrame(columns=["type","zona","nuevo","cliente","razon_social","date","value","metric"])

def last_obs_per(series_df: pd.DataFrame, by_cols: List[str], value_col: str="value") -> pd.DataFrame:
    # last observed value by group (based on max date)
    idx = series_df.groupby(by_cols)["date"].idxmax()
    return series_df.loc[idx, by_cols + ["date", value_col]].rename(columns={value_col:"last_value"})

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    with st.sidebar:
        st.header("Filtros")
        zones = sorted([z for z in df["zona"].dropna().unique() if z])
        types = sorted([t for t in df["type"].dropna().unique() if t])
        nuevos = sorted([n for n in df["nuevo"].dropna().unique() if n])

        sel_zones = st.multiselect("Zona", zones, default=zones)
        sel_types = st.multiselect("Tipo de cliente", types, default=types)
        sel_nuevo = st.multiselect("¿Nuevo?", nuevos, default=nuevos)

        df = df[df["zona"].isin(sel_zones)] if sel_zones else df
        df = df[df["type"].isin(sel_types)] if sel_types else df
        df = df[df["nuevo"].isin(sel_nuevo)] if sel_nuevo else df

        # Top-N by último MRR
        st.markdown("---")
        use_top = st.checkbox("Filtrar por Top-N MRR (último mes)", value=False)
        if use_top:
            n_top = st.slider("Top N", 3, 50, 10, step=1)
            # Compute latest MRR by cliente
            mrr_df = df[df["metric"]=="MRR"]
            if not mrr_df.empty:
                last = last_obs_per(mrr_df, ["cliente","zona","type"])
                top_clients = last.sort_values("last_value", ascending=False).head(n_top)["cliente"].tolist()
                df = df[df["cliente"].isin(top_clients)]
        return df

def kpis_overview(df: pd.DataFrame):
    # Focus on MRR for KPIs
    m = df[df["metric"]=="MRR"]
    if m.empty:
        st.info("No hay datos de MRR para KPI.")
        return
    # last and previous month
    max_date = m["date"].max()
    prev_date = (max_date - pd.offsets.MonthBegin(1))

    total_last = m[m["date"]==max_date]["value"].sum()
    total_prev = m[m["date"]==prev_date]["value"].sum()
    mom = (total_last - total_prev) / total_prev * 100 if total_prev != 0 else np.nan

    yoy_prev = (max_date - pd.DateOffset(years=1))
    total_yoy_prev = m[m["date"]==yoy_prev]["value"].sum()
    yoy = (total_last - total_yoy_prev) / total_yoy_prev * 100 if total_yoy_prev != 0 else np.nan

    # Top cliente
    last_by_client = m[m["date"]==max_date].groupby("cliente", as_index=False)["value"].sum()
    top_row = last_by_client.sort_values("value", ascending=False).head(1)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("MRR Total (último mes)", f"${total_last:,.0f}")
    c2.metric("Crecimiento MoM", f"{mom:,.1f}%" if not np.isnan(mom) else "—")
    c3.metric("Crecimiento YoY", f"{yoy:,.1f}%" if not np.isnan(yoy) else "—")
    c4.metric("Clientes activos", int((m.groupby('cliente')['value'].sum()>0).sum()))

    if not top_row.empty:
        st.caption(f"Top cliente por MRR (último mes): **{top_row.iloc[0]['cliente']}** (${top_row.iloc[0]['value']:,.0f})")

def fit_prophet(ts: pd.DataFrame, horizon_months: int=12, yearly_seasonality: bool=True) -> pd.DataFrame:
    """
    ts: dataframe with columns ['date','value']
    Returns forecast dataframe with ['date','yhat','yhat_lower','yhat_upper']
    """
    if ts["value"].sum() == 0 or ts["value"].nunique() <= 1:
        # flat or empty -> naive forecast
        last = ts["value"].iloc[-1] if not ts.empty else 0.0
        future_dates = pd.date_range(ts["date"].max()+pd.offsets.MonthBegin(1) if not ts.empty else pd.to_datetime("today"),
                                     periods=horizon_months, freq="MS")
        out = pd.DataFrame({"date": future_dates,
                            "yhat": last,
                            "yhat_lower": last,
                            "yhat_upper": last})
        return out

    mod = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_range=0.9,
    )
    dfp = ts.rename(columns={"date":"ds","value":"y"}).copy()
    mod.fit(dfp)
    future = mod.make_future_dataframe(periods=horizon_months, freq="MS")
    fcst = mod.predict(future)
    out = fcst[["ds","yhat","yhat_lower","yhat_upper"]].rename(columns={"ds":"date"})
    out = out[out["date"] > ts["date"].max()]  # only future
    return out

def make_forecast(df: pd.DataFrame, horizon: int, aggregate_first: bool, by: Optional[List[str]]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (history_df, forecast_df)
    history_df: original aggregated time series
    forecast_df: forecasted points
    """
    m = df[df["metric"]=="MRR"].copy()
    if m.empty:
        return pd.DataFrame(), pd.DataFrame()

    if by is None:
        by = []

    # aggregate by selection
    group_cols = by + ["date"]
    agg = m.groupby(group_cols, as_index=False)["value"].sum().sort_values("date")

    if aggregate_first or not by:
        # Forecast each group (or the single aggregate) independently
        results = []
        hist_parts = []
        for key, g in agg.groupby(by) if by else [([], agg)]:
            hist = g[["date","value"]].sort_values("date")
            fcst = fit_prophet(hist, horizon_months=horizon, yearly_seasonality=True)
            fcst["group_key"] = "|".join(map(str, key)) if by else "TOTAL"
            hist["group_key"] = fcst["group_key"].iloc[0]
            results.append(fcst)
            hist_parts.append(hist)
        hist_df = pd.concat(hist_parts, ignore_index=True)
        fcst_df = pd.concat(results, ignore_index=True)
        return hist_df, fcst_df

    else:
        # Forecast per cliente and then aggregate to requested grouping
        results = []
        for client, g in m.groupby("cliente"):
            series = g.groupby("date", as_index=False)["value"].sum().sort_values("date")
            fcst = fit_prophet(series, horizon_months=horizon, yearly_seasonality=True)
            fcst["cliente"] = client
            results.append(fcst)
        fcst_all = pd.concat(results, ignore_index=True)

        # Join back meta to aggregate by 'by'
        latest_meta = m.groupby("cliente", as_index=False).agg({"type":"last","zona":"last","nuevo":"last"})
        fcst_all = fcst_all.merge(latest_meta, on="cliente", how="left")

        fcst_group = fcst_all.groupby(by + ["date"], as_index=False)[["yhat","yhat_lower","yhat_upper"]].sum()
        # Build historical aggregated series as well
        hist_df = agg.rename(columns={"value":"y"}).rename(columns={"y":"value"}).copy()
        return hist_df, fcst_group

def churn_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple heuristic churn detector on MRR:
    - HARD CHURN: último mes = 0 y penúltimo > 0
    - ALTO RIESGO: suma últimos 3 meses cae >30% vs. 3 meses previos
    - MEDIO RIESGO: suma últimos 3 meses cae >15%
    """
    m = df[df["metric"]=="MRR"].copy()
    alerts = []
    for client, g in m.groupby("cliente"):
        g = g.sort_values("date")
        if g.empty:
            continue
        last = g.iloc[-1]["value"]
        prev = g.iloc[-2]["value"] if len(g) >= 2 else np.nan

        # 3-month windows
        if len(g) >= 6:
            last3 = g.iloc[-3:]["value"].sum()
            prev3 = g.iloc[-6:-3]["value"].sum()
        else:
            last3 = g.iloc[-min(3,len(g)):]["value"].sum()
            prev3 = g.iloc[:max(0,len(g)-min(3,len(g)))].tail(min(3,len(g)))["value"].sum()

        risk = None
        rule = None
        if (not np.isnan(prev)) and prev > 0 and last == 0:
            risk, rule = "HARD CHURN", "Último mes = 0 y penúltimo > 0"
        else:
            drop = (prev3 - last3) / prev3 * 100 if prev3 > 0 else 0.0
            if drop > 30:
                risk, rule = "ALTO", f"Caída {drop:.1f}% en suma 3 meses"
            elif drop > 15:
                risk, rule = "MEDIO", f"Caída {drop:.1f}% en suma 3 meses"

        if risk:
            alerts.append({
                "cliente": client,
                "zona": g["zona"].iloc[-1] if "zona" in g.columns else None,
                "type": g["type"].iloc[-1] if "type" in g.columns else None,
                "ultimo_mrr": last,
                "mrr_penultimo": prev if not np.isnan(prev) else None,
                "riesgo": risk,
                "regla": rule,
            })

    return pd.DataFrame(alerts)

def plot_series_with_forecast(hist_df: pd.DataFrame, fcst_df: pd.DataFrame, title: str):
    fig = go.Figure()
    # History
    if not hist_df.empty:
        # If multiple groups, sum for display (we also show breakdown table separately)
        agg_hist = hist_df.groupby("date", as_index=False)["value"].sum()
        fig.add_trace(go.Scatter(x=agg_hist["date"], y=agg_hist["value"], name="Histórico", mode="lines"))
    # Forecast
    if not fcst_df.empty:
        agg_fcst = fcst_df.groupby("date", as_index=False)[["yhat","yhat_lower","yhat_upper"]].sum()
        fig.add_trace(go.Scatter(x=agg_fcst["date"], y=agg_fcst["yhat"], name="Pronóstico", mode="lines"))
        fig.add_trace(go.Scatter(
            x=list(agg_fcst["date"])+list(agg_fcst["date"][::-1]),
            y=list(agg_fcst["yhat_upper"])+list(agg_fcst["yhat_lower"][::-1]),
            fill="toself",
            name="Intervalo",
            opacity=0.2,
            line=dict(width=0)
        ))
    fig.update_layout(title=title, xaxis_title="Fecha", yaxis_title="MRR", height=420, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

def correlation_section(unified: pd.DataFrame):
    st.subheader("Correlaciones entre métricas")
    # pivot by date and metric over selected scope (sum across clientes)
    if unified["metric"].nunique() <= 1:
        st.caption("Se requiere al menos 2 métricas (p.ej., MRR y Transactions o CIHS) para correlación.")
        return
    piv = unified.groupby(["metric","date"], as_index=False)["value"].sum()
    wide = piv.pivot(index="date", columns="metric", values="value").dropna(how="any")
    if wide.shape[1] < 2 or wide.empty:
        st.caption("No hay suficientes datos coincidentes para correlación.")
        return
    corr = wide.corr(method="pearson")
    st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)

# -----------------------------
# Sidebar: Upload & Settings
# -----------------------------

st.sidebar.title("Configuración")
uploaded = st.sidebar.file_uploader("Sube tu archivo (.csv o .xlsx)", type=["csv","xlsx","xls"])

horizon = st.sidebar.slider("Horizonte de pronóstico (meses)", 1, 12, 6)
aggregate_first = st.sidebar.radio("Método de consolidación", ["Agregue y pronostique", "Pronostique por cliente y agregue"], index=0)
aggregate_first = (aggregate_first == "Agregue y pronostique")

grouping = st.sidebar.multiselect(
    "Agrupar resultados por (para series consolidadas y tablas)",
    options=["zona","type","nuevo"],
    default=["zona","type"]
)

# -----------------------------
# Main
# -----------------------------

st.title("Revenue & Health Forecaster (MRR / CIHS / Transactions)")

if uploaded is None:
    st.info("Sube un archivo para comenzar. Para CSVs con estructura como la demo de MRR, basta con subir el archivo. Para Excel, usa hojas llamadas 'MRR', 'CIHS' y 'Transactions'.")
    st.stop()

tables = load_file(uploaded)
unified = build_unified_long(tables)

if unified.empty:
    st.error("No se pudieron leer datos. Verifica la estructura del archivo.")
    st.stop()

# Apply sidebar filters
filtered = apply_filters(unified)

# KPIs
st.subheader("Resumen")
kpis_overview(filtered)

# Forecasting
st.subheader("Pronóstico de MRR consolidado")
hist_df, fcst_df = make_forecast(filtered, horizon=horizon, aggregate_first=aggregate_first, by=grouping if grouping else None)
plot_series_with_forecast(hist_df, fcst_df, "MRR Histórico + Pronóstico")

# Detailed tables
with st.expander("Ver tablas detalladas"):
    st.write("Histórico (agregado según filtros/agrupación)")
    if not hist_df.empty:
        st.dataframe(hist_df, use_container_width=True)
        csv_hist = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar histórico (CSV)", csv_hist, "historico.csv", "text/csv")

    st.write("Pronóstico (agregado según filtros/agrupación)")
    if not fcst_df.empty:
        st.dataframe(fcst_df, use_container_width=True)
        csv_fcst = fcst_df.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar pronóstico (CSV)", csv_fcst, "pronostico.csv", "text/csv")

# Churn alerts
st.subheader("Alertas de Churn (heurísticas)")
alerts = churn_alerts(filtered)
if alerts.empty:
    st.caption("Sin alertas según las reglas actuales.")
else:
    st.dataframe(alerts.sort_values(["riesgo","ultimo_mrr"], ascending=[True, False]), use_container_width=True)
    st.download_button("Descargar alertas (CSV)", alerts.to_csv(index=False).encode("utf-8"), "alertas_churn.csv", "text/csv")

# Correlations
correlation_section(filtered)

st.caption("Sugerencia: alimenta el Excel con hojas separadas 'MRR', 'CIHS' y 'Transactions' para obtener correlaciones y alertas más inteligentes.")
