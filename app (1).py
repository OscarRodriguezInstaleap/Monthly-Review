
import io
import sys
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from prophet import Prophet
from typing import Dict, List, Tuple, Optional, Any
import re
from datetime import datetime

st.set_page_config(page_title="Revenue & Health Forecaster", layout="wide")

@st.cache_data(show_spinner=False)
def load_file(uploaded) -> Dict[str, pd.DataFrame]:
    if uploaded is None:
        return {}
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
        return {"MRR": df}
    else:
        xls = pd.ExcelFile(uploaded)
        sheets = {}
        for sheet in xls.sheet_names:
            key = sheet.strip().lower()
            df = pd.read_excel(uploaded, sheet_name=sheet)
            if "mrr" in key:
                sheets["MRR"] = df
            elif "cihs" in key or "health" in key:
                sheets["CIHS"] = df
            elif "trans" in key or "tx" in key or "orders" in key:
                sheets["Transactions"] = df
            else:
                sheets[sheet] = df
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
    long_df = df.melt(id_vars=meta_cols, value_vars=ts_cols, var_name="month", value_name="value_raw")
    long_df["value"] = (
        long_df["value_raw"].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
            .replace({"": None, "nan": None, "None": None, "NULL": None, "null": None})
    )
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce").fillna(0.0)
    def _parse_date(x):
        if isinstance(x, (pd.Timestamp, datetime)):
            return pd.to_datetime(x)
        x = str(x)
        for fmt in ("%b-%y", "%b-%Y", "%B-%y", "%B-%Y"):
            try:
                return pd.to_datetime(x, format=fmt)
            except Exception:
                pass
        try:
            return pd.to_datetime(x)
        except Exception:
            return pd.NaT
    long_df["date"] = long_df["month"].map(_parse_date)
    long_df["period"] = long_df["date"].dt.to_period("M")
    long_df["metric"] = metric_name
    return long_df

def build_unified_long(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for key, df in tables.items():
        key_norm = key.strip().lower()
        if key_norm in ("mrr","revenue"):
            parts.append(to_long(df, "MRR"))
        elif key_norm in ("cihs","health"):
            parts.append(to_long(df, "CIHS"))
        elif key_norm.startswith("trans") or key_norm in ("tx","orders","transactions"):
            parts.append(to_long(df, "Transactions"))
        else:
            parts.append(to_long(df, key))
    if parts:
        out = pd.concat(parts, ignore_index=True)
        out = standardize_columns(out)
        for col in ["type","zona","nuevo","cliente","razon_social"]:
            if col not in out.columns:
                out[col] = None
        for col in ["type","zona","nuevo","cliente","razon_social"]:
            out[col] = out[col].astype(str).str.strip()
        out["period"] = pd.PeriodIndex(out["period"]).astype(str)
        return out
    return pd.DataFrame(columns=["type","zona","nuevo","cliente","razon_social","date","period","value","metric"])

def filters_ui(df: pd.DataFrame):
    st.subheader("Filtros")
    with st.container():
        map_file = st.file_uploader("Mapa opcional de grupos (CSV con columnas: cliente, grupo)", type=["csv"], key="map_groups")
        map_df = None
        if map_file is not None:
            try:
                map_df = pd.read_csv(map_file)
                if set(map_df.columns.str.lower()) >= {"cliente","grupo"}:
                    map_df.columns = map_df.columns.str.lower()
                else:
                    st.warning("El CSV debe tener columnas 'cliente' y 'grupo'. Se ignorará el mapeo.")
                    map_df = None
            except Exception as e:
                st.warning(f"No se pudo leer el mapeo: {e}")
                map_df = None

        cols = st.columns(4)
        zones = sorted([z for z in df["zona"].dropna().unique() if z and z != "None"])
        sel_all_z = cols[0].checkbox("Seleccionar todo (Zona)", value=True, key="all_zona")
        sel_zones = cols[0].multiselect("Zona", zones, default=zones if sel_all_z else zones[:0])

        types = sorted([t for t in df["type"].dropna().unique() if t and t != "None"])
        sel_all_t = cols[1].checkbox("Seleccionar todo (Tipo)", value=True, key="all_type")
        sel_types = cols[1].multiselect("Tipo", types, default=types if sel_all_t else types[:0])

        nuevos = sorted([n for n in df["nuevo"].dropna().unique() if n and n != "None"])
        sel_all_n = cols[2].checkbox("Seleccionar todo (Nuevo)", value=True, key="all_nuevo")
        sel_nuevo = cols[2].multiselect("Nuevo", nuevos, default=nuevos if sel_all_n else nuevos[:0])

        clients = sorted([c for c in df["cliente"].dropna().unique() if c and c != "None"])
        sel_all_c = cols[3].checkbox("Seleccionar todo (Clientes)", value=False, key="all_clients")
        sel_clients = cols[3].multiselect("Clientes", clients, default=clients if sel_all_c else [])

        out = df.copy()
        if sel_zones: out = out[out["zona"].isin(sel_zones)]
        if sel_types: out = out[out["type"].isin(sel_types)]
        if sel_nuevo: out = out[out["nuevo"].isin(sel_nuevo)]
        if sel_clients: out = out[out["cliente"].isin(sel_clients)]

        st.markdown("---")
        use_top = st.checkbox("Filtrar por Top-N MRR (último mes)", value=False)
        top_clients = []
        if use_top:
            n_top = st.slider("Top N", 3, 50, 10, step=1)
            mrr_df = out[out["metric"]=="MRR"]
            if not mrr_df.empty:
                last_period = mrr_df["period"].max()
                last = mrr_df[mrr_df["period"]==last_period].groupby("cliente", as_index=False)["value"].sum()
                top_clients = last.sort_values("value", ascending=False).head(n_top)["cliente"].tolist()
                out = out[out["cliente"].isin(top_clients)]
        return out, sel_clients, map_df

def compute_kpis(df: pd.DataFrame):
    m = df[df["metric"]=="MRR"].copy()
    if m.empty:
        return None
    agg = m.groupby("period", as_index=False)["value"].sum().sort_values("period")
    last_p = agg["period"].max()
    periods = sorted(agg["period"].unique().tolist())
    idx = periods.index(last_p) if last_p in periods else -1
    prev_p = periods[idx-1] if idx-1>=0 else None
    try: yoy_p = str((pd.Period(last_p) - 12))
    except Exception: yoy_p = None

    total_last = float(agg[agg["period"]==last_p]["value"].sum()) if last_p else 0.0
    total_prev = float(agg[agg["period"]==prev_p]["value"].sum()) if prev_p else 0.0
    total_yoy_prev = float(agg[agg["period"]==yoy_p]["value"].sum()) if yoy_p else 0.0

    mom = ((total_last - total_prev) / total_prev * 100.0) if total_prev>0 else None
    yoy = ((total_last - total_yoy_prev) / total_yoy_prev * 100.0) if total_yoy_prev>0 else None
    arr = total_last * 12.0

    nrr = None; cohort_size = 0
    if yoy_p:
        base_clients = m[(m["period"]==yoy_p) & (m["value"]>0)]["cliente"].unique().tolist()
        cohort_size = len(base_clients)
        base = m[(m["period"]==yoy_p) & (m["cliente"].isin(base_clients))]["value"].sum()
        now_for = m[(m["period"]==last_p) & (m["cliente"].isin(base_clients))]["value"].sum()
        if base>0:
            nrr = float(now_for)/float(base)*100.0

    active_clients = int((m.groupby("cliente")["value"].sum()>0).sum())
    return {"last_period": last_p, "prev_period": prev_p, "yoy_period": yoy_p, "total_last": total_last,
            "mom": mom, "yoy": yoy, "arr": arr, "nrr_yoy": nrr, "nrr_cohort_size": cohort_size,
            "active_clients": active_clients}

def fit_prophet(ts: pd.DataFrame, horizon_months: int=12, yearly_seasonality: bool=True) -> pd.DataFrame:
    if ts.empty or ts["value"].sum() == 0 or ts["value"].nunique() <= 1:
        last = ts["value"].iloc[-1] if not ts.empty else 0.0
        future_dates = pd.date_range(ts["date"].max()+pd.offsets.MonthBegin(1) if not ts.empty else pd.to_datetime("today"),
                                     periods=horizon_months, freq="MS")
        out = pd.DataFrame({"date": future_dates, "yhat": last, "yhat_lower": last, "yhat_upper": last})
        out["period"] = out["date"].dt.to_period("M").astype(str)
        return out
    mod = Prophet(yearly_seasonality=yearly_seasonality, weekly_seasonality=False, daily_seasonality=False,
                  seasonality_mode="additive", changepoint_range=0.9)
    dfp = ts.rename(columns={"date":"ds","value":"y"}).copy()
    mod.fit(dfp)
    future = mod.make_future_dataframe(periods=horizon_months, freq="MS")
    fcst = mod.predict(future)
    out = fcst[["ds","yhat","yhat_lower","yhat_upper"]].rename(columns={"ds":"date"})
    out = out[out["date"] > ts["date"].max()]
    out["period"] = out["date"].dt.to_period("M").astype(str)
    return out

def make_forecast(df: pd.DataFrame, metric: str, horizon: int, aggregate_first: bool, by: Optional[List[str]]=None):
    m = df[df["metric"]==metric].copy()
    if m.empty:
        return pd.DataFrame(), pd.DataFrame()
    if by is None: by = []
    group_cols = by + ["date"]
    agg = m.groupby(group_cols, as_index=False)["value"].sum().sort_values("date")
    if aggregate_first or not by:
        results = []; hist_parts = []
        for key, g in agg.groupby(by) if by else [([], agg)]:
            hist = g[["date","value"]].sort_values("date")
            fcst = fit_prophet(hist, horizon_months=horizon, yearly_seasonality=True)
            group_key = "|".join(map(str, key)) if by else "TOTAL"
            fcst["group_key"] = group_key; hist["group_key"] = group_key
            results.append(fcst); hist_parts.append(hist)
        return pd.concat(hist_parts, ignore_index=True), pd.concat(results, ignore_index=True)
    else:
        results = []
        for client, g in m.groupby("cliente"):
            series = g.groupby("date", as_index=False)["value"].sum().sort_values("date")
            fcst = fit_prophet(series, horizon_months=horizon, yearly_seasonality=True)
            fcst["cliente"] = client
            results.append(fcst)
        fcst_all = pd.concat(results, ignore_index=True)
        latest_meta = m.groupby("cliente", as_index=False).agg({"type":"last","zona":"last","nuevo":"last"})
        fcst_all = fcst_all.merge(latest_meta, on="cliente", how="left")
        fcst_group = fcst_all.groupby(by + ["date","period"], as_index=False)[["yhat","yhat_lower","yhat_upper"]].sum()
        hist_df = agg.rename(columns={"value":"y"}).rename(columns={"y":"value"}).copy()
        return hist_df, fcst_group

def plot_series_with_forecast(hist_df: pd.DataFrame, fcst_df: pd.DataFrame, title: str):
    fig = go.Figure()
    if not hist_df.empty:
        agg_hist = hist_df.groupby("date", as_index=False)["value"].sum()
        fig.add_trace(go.Scatter(x=agg_hist["date"], y=agg_hist["value"], name="Histórico", mode="lines"))
    if not fcst_df.empty:
        agg_fcst = fcst_df.groupby("date", as_index=False)[["yhat","yhat_lower","yhat_upper"]].sum()
        fig.add_trace(go.Scatter(x=agg_fcst["date"], y=agg_fcst["yhat"], name="Pronóstico", mode="lines"))
        fig.add_trace(go.Scatter(x=list(agg_fcst["date"])+list(agg_fcst["date"][::-1]),
                                 y=list(agg_fcst["yhat_upper"])+list(agg_fcst["yhat_lower"][::-1]),
                                 fill="toself", name="Intervalo", opacity=0.2, line=dict(width=0)))
    fig.update_layout(title=title, xaxis_title="Fecha", yaxis_title="", height=420, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

def cihs_section(unified: pd.DataFrame):
    st.subheader("CIHS (Adopción de funcionalidades)")
    c = unified[unified["metric"]=="CIHS"].copy()
    if c.empty:
        st.caption("No se encontraron datos de CIHS."); return
    agg = c.groupby("date", as_index=False)["value"].mean().sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg["date"], y=agg["value"], mode="lines", name="CIHS promedio"))
    fig.update_layout(title="CIHS promedio (global)", xaxis_title="Fecha", yaxis_title="CIHS", height=360, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)
    last_p = c["period"].max()
    by_client_last = c[c["period"]==last_p].groupby("cliente", as_index=False)["value"].mean()
    topN = by_client_last.sort_values("value", ascending=False).head(10)
    botN = by_client_last.sort_values("value", ascending=True).head(10)
    col1, col2 = st.columns(2)
    col1.write("**Top 10 mayor adopción (último mes)**"); col1.dataframe(topN, use_container_width=True)
    col2.write("**Top 10 menor adopción (último mes)**"); col2.dataframe(botN, use_container_width=True)

def churn_alerts(df: pd.DataFrame) -> pd.DataFrame:
    m = df[df["metric"]=="MRR"].copy()
    tx = df[df["metric"]=="Transactions"].copy()
    c = df[df["metric"]=="CIHS"].copy()
    alerts = []
    for client, g in m.groupby("cliente"):
        g = g.sort_values("date")
        if g.empty: continue
        last = g.iloc[-1]["value"]
        prev = g.iloc[-2]["value"] if len(g)>=2 else np.nan
        if len(g) >= 6:
            last3 = g.iloc[-3:]["value"].sum(); prev3 = g.iloc[-6:-3]["value"].sum()
        else:
            last3 = g.iloc[-min(3,len(g)):]["value"].sum()
            prev3 = g.iloc[:max(0,len(g)-min(3,len(g)))].tail(min(3,len(g)))["value"].sum()
        risk = None; rule = None; signals = []
        if (not np.isnan(prev)) and prev > 0 and last == 0:
            risk, rule = "HARD CHURN", "Último mes = 0 y penúltimo > 0"
        else:
            drop = (prev3 - last3) / prev3 * 100 if prev3 > 0 else 0.0
            if drop > 30: risk, rule = "ALTO", f"MRR: caída {drop:.1f}% en 3m"
            elif drop > 15: risk, rule = "MEDIO", f"MRR: caída {drop:.1f}% en 3m"
        if not tx[tx["cliente"]==client].empty:
            txg = tx[tx["cliente"]==client].sort_values("date")
            if len(txg)>=6:
                tx_last3 = txg.iloc[-3:]["value"].sum(); tx_prev3 = txg.iloc[-6:-3]["value"].sum()
                tx_drop = (tx_prev3 - tx_last3)/tx_prev3*100 if tx_prev3>0 else 0.0
                if tx_drop>25: signals.append(f"TX caída {tx_drop:.0f}%")
        if not c[c["cliente"]==client].empty:
            cg = c[c["cliente"]==client].sort_values("date")
            if len(cg)>=3:
                c_last3 = cg.iloc[-3:]["value"].mean()
                c_prev3 = cg.iloc[-6:-3]["value"].mean() if len(cg)>=6 else cg.iloc[:max(0,len(cg)-3)].tail(3)["value"].mean()
                c_drop = c_prev3 - c_last3
                if c_drop>5: signals.append(f"CIHS baja {c_drop:.1f} pts")
        if risk or signals:
            alerts.append({"cliente": client, "zona": g["zona"].iloc[-1] if "zona" in g.columns else None,
                           "type": g["type"].iloc[-1] if "type" in g.columns else None,
                           "ultimo_mrr": last, "mrr_penultimo": prev if not np.isnan(prev) else None,
                           "riesgo": risk if risk else "SEÑALES", "evidencias": "; ".join(signals) if signals else rule})
    return pd.DataFrame(alerts)

def yoy_cohort_stacked(df: pd.DataFrame):
    st.subheader("MRR por cohorte (clientes activos en el mes seleccionado)")
    m = df[df["metric"]=="MRR"].copy()
    if m.empty:
        st.caption("No hay datos de MRR para gráfico de cohorte."); return
    all_periods = sorted(m["period"].dropna().unique().tolist())
    sel_p = st.selectbox("Selecciona el mes (cohorte)", options=all_periods, index=len(all_periods)-1, key="cohort_period")
    yoy_p = str((pd.Period(sel_p) - 12))
    cohort = sorted(m[(m["period"]==sel_p) & (m["value"]>0)]["cliente"].unique().tolist())
    now_df = m[(m["period"]==sel_p) & (m["cliente"].isin(cohort))].groupby("cliente", as_index=False)["value"].sum().rename(columns={"value":"now"})
    yoy_df = m[(m["period"]==yoy_p) & (m["cliente"].isin(cohort))].groupby("cliente", as_index=False)["value"].sum().rename(columns={"value":"yoy"})
    det = now_df.merge(yoy_df, on="cliente", how="outer").fillna(0.0)
    topN = st.slider("Top N clientes para apilar (por valor del mes seleccionado)", 5, 25, 12)
    det["rank"] = det["now"].rank(ascending=False, method="first")
    top_clients = det.sort_values("now", ascending=False).head(topN)["cliente"].tolist()
    det["cliente_bucket"] = det["cliente"].where(det["cliente"].isin(top_clients), other="Otros")
    stack = det.groupby("cliente_bucket", as_index=False)[["now","yoy"]].sum()
    fig = go.Figure()
    for _, row in stack.iterrows():
        fig.add_trace(go.Bar(x=["Año anterior", "Mes seleccionado"], y=[row["yoy"], row["now"]], name=str(row["cliente_bucket"])))
    fig.update_layout(barmode="stack", title=f"Cohorte {sel_p}: clientes activos vs mismo mes del año anterior",
                      xaxis_title="", yaxis_title="MRR", height=420, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Detalle por cliente y descarga"):
        st.dataframe(det.sort_values("now", ascending=False), use_container_width=True)
        st.download_button("Descargar detalle (CSV)", det.to_csv(index=False).encode("utf-8"), "cohorte_detalle.csv", "text/csv")

st.sidebar.title("Configuración")
uploaded = st.sidebar.file_uploader("Sube tu archivo (.csv o .xlsx)", type=["csv","xlsx","xls"])
horizon = st.sidebar.slider("Horizonte de pronóstico (meses)", 1, 12, 6)
aggregate_first = st.sidebar.radio("Método de consolidación", ["Agregue y pronostique", "Pronostique por cliente y agregue"], index=0)
aggregate_first = (aggregate_first == "Agregue y pronostique")
grouping = st.sidebar.multiselect("Agrupar resultados por", options=["zona","type","nuevo"], default=["zona","type"])

st.title("Revenue & Health Forecaster (MRR / CIHS / Transactions)")
if uploaded is None:
    st.info("Sube un archivo para comenzar. Para Excel, usa hojas llamadas 'MRR', 'CIHS' y 'Transactions'."); st.stop()
tables = load_file(uploaded)
unified = build_unified_long(tables)
if unified.empty:
    st.error("No se pudieron leer datos. Verifica la estructura del archivo."); st.stop()

filtered, selected_clients, map_df = filters_ui(unified)

st.subheader("Resumen")
k = compute_kpis(filtered)
if not k:
    st.info("No hay datos de MRR para KPIs.")
else:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric(f"MRR Total ({k['last_period']})", f"${k['total_last']:,.0f}")
    c2.metric("Crecimiento MoM", f"{k['mom']:,.1f}%" if k['mom'] is not None else "—")
    c3.metric("Crecimiento YoY", f"{k['yoy']:,.1f}%" if k['yoy'] is not None else "—")
    c4.metric("Clientes activos", k["active_clients"])
    c5,c6 = st.columns(2)
    c5.metric("ARR (run-rate)", f"${k['arr']:,.0f}")
    c6.metric("NRR YoY (cohorte)", f"{k['nrr_yoy']:,.1f}%" if k['nrr_yoy'] is not None else "—", help=f"Cohorte basada en {k['yoy_period']}")

st.subheader("Top 5 clientes por MRR (agrupado por grupo)")
m = filtered[filtered["metric"]=="MRR"].copy()
if not m.empty:
    last_p = m["period"].max()
    base = m[m["period"]==last_p].groupby(["cliente","zona"], as_index=False)["value"].sum()
    if map_df is not None:
        base = base.merge(map_df, on="cliente", how="left")
        base["grupo"] = base["grupo"].fillna(base["zona"])
    else:
        base["grupo"] = base["zona"]
    for gname, sub in base.groupby("grupo"):
        top5 = sub.sort_values("value", ascending=False).head(5)[["cliente","value"]]
        st.write(f"**{gname}**"); st.dataframe(top5.rename(columns={"value":"MRR"}), use_container_width=True)
else:
    st.caption("No hay datos de MRR para el último periodo.")

st.subheader("Pronóstico de MRR consolidado")
hist_df, fcst_df = make_forecast(filtered, metric="MRR", horizon=horizon, aggregate_first=aggregate_first, by=grouping if grouping else None)
plot_series_with_forecast(hist_df, fcst_df, "MRR Histórico + Pronóstico")
with st.expander("Tabla resumen de pronóstico (exportable)"):
    if not fcst_df.empty:
        agg_fc = fcst_df.groupby(["period"], as_index=False)[["yhat","yhat_lower","yhat_upper"]].sum()
        st.dataframe(agg_fc, use_container_width=True)
        st.download_button("Descargar pronóstico (CSV)", agg_fc.to_csv(index=False).encode("utf-8"), "pronostico_mrr.csv", "text/csv")
    else:
        st.caption("Sin datos de pronóstico.")

if selected_clients:
    st.subheader("Pronóstico por cliente(s) seleccionado(s)")
    max_clients = 12; show_clients = selected_clients[:max_clients]
    if len(selected_clients) > max_clients:
        st.caption(f"Mostrando los primeros {max_clients} clientes seleccionados.")
    for cli in show_clients:
        cli_df = filtered[filtered["cliente"]==cli]
        h, f = make_forecast(cli_df, metric="MRR", horizon=horizon, aggregate_first=True, by=[])
        plot_series_with_forecast(h, f, f"MRR - {cli} (Histórico + Pronóstico)")
        if not f.empty:
            with st.expander(f"Tabla pronóstico - {cli}"):
                st.dataframe(f[["date","period","yhat","yhat_lower","yhat_upper"]], use_container_width=True)

st.subheader("Transacciones globales")
if (filtered["metric"]=="Transactions").any():
    show_tx_fc = st.checkbox("Mostrar pronóstico de Transacciones", value=True)
    hist_tx, fcst_tx = make_forecast(filtered, metric="Transactions", horizon=horizon if show_tx_fc else 1, aggregate_first=aggregate_first, by=grouping if grouping else None)
    if not show_tx_fc: fcst_tx = pd.DataFrame()
    plot_series_with_forecast(hist_tx, fcst_tx, "Transacciones Histórico" + (" + Pronóstico" if show_tx_fc else ""))
    with st.expander("Tabla resumen de pronóstico de Transacciones (exportable)"):
        if not fcst_tx.empty:
            agg_tx = fcst_tx.groupby(["period"], as_index=False)[["yhat","yhat_lower","yhat_upper"]].sum()
            st.dataframe(agg_tx, use_container_width=True)
            st.download_button("Descargar pronóstico TX (CSV)", agg_tx.to_csv(index=False).encode("utf-8"), "pronostico_tx.csv", "text/csv")
else:
    st.caption("No se encontraron datos de Transacciones en el archivo cargado.")

cihs_section(unified=filtered)
yoy_cohort_stacked(filtered)

st.subheader("Alertas de Churn")
alerts = churn_alerts(filtered)
if alerts.empty:
    st.caption("Sin alertas según las reglas actuales.")
else:
    st.dataframe(alerts.sort_values(["riesgo","ultimo_mrr"], ascending=[True, False]), use_container_width=True)
    st.download_button("Descargar alertas (CSV)", alerts.to_csv(index=False).encode("utf-8"), "alertas_churn.csv", "text/csv")
