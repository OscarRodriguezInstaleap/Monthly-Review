import os
import re
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet

# =========================
# Config
# =========================
st.set_page_config(page_title="Instaleap — Forecast V2", layout="wide")
SHEET_ID = os.environ.get("SHEET_ID", "1ACX9OWNB0vHs8EpxeHxgByuPjDP9VC0E3k9b61V-i1I")
SHEETS = ("MRR", "Transactions")
DEFAULT_HORIZON = 12

# =========================
# Helpers de datos
# =========================
MONTH_RE = re.compile(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2}(?:\.\d+)?$')

def gviz_csv_url(sheet_id: str, sheet_name: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

def to_numeric_clean(s: pd.Series) -> pd.Series:
    """Convierte strings con $ , . y formatos ES/EN a float seguro."""
    s = s.astype(str).str.strip()
    s = s.replace({'': '0', 'nan': '0', 'None':'0'})
    s = s.str.replace(r'[\$\s]', '', regex=True)
    # Formato español: 12.345,67  o 12.345
    es_mask = s.str.match(r'^\d{1,3}(?:\.\d{3})+(?:,\d+)?$')
    s.loc[es_mask] = s.loc[es_mask].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    # Formato EN miles: 12,345.67 -> quitar comas
    s = s.str.replace(',', '', regex=False)
    return pd.to_numeric(s, errors='coerce').fillna(0.0)

def pick_best_month_columns(df: pd.DataFrame):
    """Cuando hay duplicados (Jan-25, Jan-25.1, ...), elige para cada base la columna de mayor suma."""
    month_cols = [c for c in df.columns if MONTH_RE.match(str(c))]
    groups = {}
    for c in month_cols:
        base = c.split('.')[0]
        groups.setdefault(base, []).append(c)

    selected = {}
    for base, cols in groups.items():
        best = max(cols, key=lambda col: to_numeric_clean(df[col]).sum())
        selected[base] = best

    order = sorted(selected.keys(), key=lambda b: pd.to_datetime(b, format='%b-%y'))
    return [selected[b] for b in order], order

def trim_trailing_zero_months(long_df: pd.DataFrame) -> pd.DataFrame:
    """Recorta meses finales con puro 0 para evitar meses fantasma."""
    sums = long_df.groupby('period', as_index=False)['value'].sum().sort_values('period')
    nz = sums[sums['value'] != 0]
    if not nz.empty:
        last = nz['period'].max()
        return long_df[long_df['period'] <= last]
    return long_df

def normalize_wide_to_long(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """
    Convierte una hoja en ancho a long:
    columnas meta esperadas (si existen): Country, type client, Nuevo, Zona, Cliente, Razon Social
    columnas de meses: Jan-25, Feb-25, ... (con o sin .1, .2, ...)
    """
    meta = ['Country','type client','Nuevo','Zona','Cliente','Razon Social']
    # Selección robusta de meses
    selected_cols, base_order = pick_best_month_columns(df)
    keep = [c for c in meta if c in df.columns] + selected_cols
    sub = df[keep].copy()

    # Renombra a base sin sufijos .1, .2
    rename_map = dict(zip(selected_cols, base_order))
    sub.rename(columns=rename_map, inplace=True)

    # wide -> long
    long = sub.melt(
        id_vars=[c for c in meta if c in sub.columns],
        value_vars=base_order,
        var_name='period_label',
        value_name='value_raw'
    )

    # numérico robusto
    long['value'] = to_numeric_clean(long['value_raw'])
    long.drop(columns=['value_raw'], inplace=True)

    # period YYYY-MM y date primer día del mes
    long['period'] = pd.to_datetime(long['period_label'], format='%b-%y').dt.to_period('M').astype(str)
    long['date'] = pd.to_datetime(long['period'] + '-01')

    # normaliza meta
    ren = {'Country':'pais','type client':'type','Razon Social':'razon_social',
           'Cliente':'cliente','Zona':'zona','Nuevo':'nuevo'}
    long.rename(columns={k:v for k,v in ren.items() if k in long.columns}, inplace=True)

    long['metric'] = metric_name
    long = trim_trailing_zero_months(long)
    return long

@st.cache_data(show_spinner=False, ttl=600)
def load_all() -> pd.DataFrame:
    tables = {}
    for s in SHEETS:
        url = gviz_csv_url(SHEET_ID, s)
        tables[s] = pd.read_csv(url)
    mrr_long = normalize_wide_to_long(tables["MRR"], "MRR")
    tx_long  = normalize_wide_to_long(tables["Transactions"], "Transactions")
    unified = pd.concat([mrr_long, tx_long], ignore_index=True)
    # columnas mínimas homogéneas
    for col in ["pais","type","zona","cliente","razon_social","period","date","value","metric"]:
        if col not in unified.columns:
            unified[col] = np.nan
    return unified

# =========================
# Forecast
# =========================
def fit_prophet_series(series_df: pd.DataFrame, horizon=12) -> pd.DataFrame:
    """
    series_df: columnas ['date','value']
    retorna dataframe con ['date','yhat','yhat_lower','yhat_upper']
    """
    s = series_df.dropna().copy()
    s = s.sort_values("date")
    if s["value"].sum() == 0 or len(s) < 3:
        # Series sin información suficiente; devolvemos vacío
        return pd.DataFrame(columns=["date","yhat","yhat_lower","yhat_upper"])

    m = Prophet(interval_width=0.8, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(s.rename(columns={"date":"ds","value":"y"}))
    fut = m.make_future_dataframe(periods=horizon, freq="MS", include_history=False)
    fc = m.predict(fut)
    out = pd.DataFrame({
        "date": pd.to_datetime(fc["ds"]),
        "yhat": fc["yhat"],
        "yhat_lower": fc["yhat_lower"],
        "yhat_upper": fc["yhat_upper"]
    })
    return out

def plot_hist_fc(hist: pd.DataFrame, fc: pd.DataFrame, title: str, label_mode="extrema"):
    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["value"], mode="lines", name="Histórico"))
    if not fc.empty:
        # banda
        fig.add_trace(go.Scatter(
            x=list(fc["date"])+list(fc["date"][::-1]),
            y=list(fc["yhat_upper"])+list(fc["yhat_lower"][::-1]),
            fill="toself", name="Intervalo", opacity=0.2, line=dict(width=0)
        ))
        # línea forecast
        mode = "lines"
        text = None
        if label_mode in ("extrema","all"):
            y = fc["yhat"].to_numpy()
            if label_mode == "all":
                text = [f"{v:,.0f}" for v in y]
                mode = "lines+markers+text"
            else:
                # marcar picos y valles
                win = 2
                is_max = (pd.Series(y).rolling(win, center=True).max() == pd.Series(y)).fillna(False).to_numpy()
                is_min = (pd.Series(y).rolling(win, center=True).min() == pd.Series(y)).fillna(False).to_numpy()
                text = [f"{y[i]:,.0f}" if (is_max[i] or is_min[i]) else None for i in range(len(y))]
                mode = "lines+markers+text"
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["yhat"], mode=mode, name="Pronóstico", text=text, textposition="top center"))
    fig.update_layout(title=title, height=460, margin=dict(l=10,r=10,t=40,b=10), xaxis_title="", yaxis_title="")
    return fig

def apply_scenario(fc: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if fc.empty:
        return fc
    mult = 1.0
    if scenario == "Pesimista (–10%)":
        mult = 0.90
    elif scenario == "Optimista (+10%)":
        mult = 1.10
    out = fc.copy()
    out["yhat"] = out["yhat"] * mult
    out["yhat_lower"] = out["yhat_lower"] * mult
    out["yhat_upper"] = out["yhat_upper"] * mult
    return out

# =========================
# UI
# =========================
st.title("Instaleap — Forecast V2 (MRR & Transacciones)")
with st.sidebar:
    st.markdown("### Fuente de datos")
    st.write(f"Google Sheet ID: `{SHEET_ID}`")
    if st.button("Refrescar datos"):
        load_all.clear()  # limpia cache
        st.success("Datos recargados.")

data = load_all()

# Filtros simples (opcionales)
with st.expander("Filtros", expanded=False):
    zonas = sorted([z for z in data["zona"].dropna().unique().tolist()])
    tipos = sorted([t for t in data["type"].dropna().unique().tolist()])
    clientes = sorted([c for c in data["cliente"].dropna().unique().tolist()])

    z_sel = st.multiselect("Zona", zonas, default=zonas)
    t_sel = st.multiselect("Tipo", tipos, default=tipos)
    c_sel = st.multiselect("Clientes (opcional)", clientes, default=[])

def apply_filters(d: pd.DataFrame, z_sel, t_sel, c_sel):
    f = d.copy()
    if z_sel:
        f = f[f["zona"].isin(z_sel)]
    if t_sel:
        f = f[f["type"].isin(t_sel)]
    if c_sel:
        f = f[f["cliente"].isin(c_sel)]
    return f

fdata = apply_filters(data, z_sel if 'z_sel' in locals() else None,
                      t_sel if 't_sel' in locals() else None,
                      c_sel if 'c_sel' in locals() else None)

tabs = st.tabs(["MRR (hist + forecast)", "Transacciones (hist + forecast)", "Resumen por cuenta"])

# =========================
# Tab 1: MRR
# =========================
with tabs[0]:
    st.subheader("MRR — Histórico + Pronóstico (12 meses)")
    scenario = st.selectbox("Escenario", ["Base", "Pesimista (–10%)", "Optimista (+10%)"], index=0)
    horizon = st.slider("Horizonte (meses)", 3, 18, DEFAULT_HORIZON, 1)

    m_hist = (fdata[fdata["metric"]=="MRR"]
              .groupby("date", as_index=False)["value"].sum()
              .sort_values("date"))
    m_fc = fit_prophet_series(m_hist, horizon=horizon)
    m_fc = apply_scenario(m_fc, scenario)

    fig = plot_hist_fc(m_hist, m_fc, title=f"MRR (escenario: {scenario})", label_mode="extrema")
    st.plotly_chart(fig, use_container_width=True)

    # Tabla exportable
    merged = m_hist.rename(columns={"value":"historico"}).merge(m_fc, on="date", how="outer").sort_values("date")
    merged["period"] = merged["date"].dt.to_period("M").astype(str)
    st.dataframe(merged, use_container_width=True, height=300)
    st.download_button("Descargar CSV MRR (hist + forecast)", merged.to_csv(index=False).encode("utf-8"),
                       file_name="mrr_hist_forecast.csv", mime="text/csv")

# =========================
# Tab 2: Transacciones
# =========================
with tabs[1]:
    st.subheader("Transacciones — Histórico + Pronóstico (12 meses)")
    scenario_tx = st.selectbox("Escenario TX", ["Base", "Pesimista (–10%)", "Optimista (+10%)"], index=0, key="sc_tx")
    horizon_tx = st.slider("Horizonte TX (meses)", 3, 18, DEFAULT_HORIZON, 12, key="hr_tx")

    t_hist = (fdata[fdata["metric"]=="Transactions"]
              .groupby("date", as_index=False)["value"].sum()
              .sort_values("date"))
    t_fc = fit_prophet_series(t_hist, horizon=horizon_tx)
    t_fc = apply_scenario(t_fc, scenario_tx)

    fig_tx = plot_hist_fc(t_hist, t_fc, title=f"Transacciones (escenario: {scenario_tx})", label_mode="none")
    st.plotly_chart(fig_tx, use_container_width=True)

    merged_tx = t_hist.rename(columns={"value":"historico"}).merge(t_fc, on="date", how="outer").sort_values("date")
    merged_tx["period"] = merged_tx["date"].dt.to_period("M").astype(str)
    st.dataframe(merged_tx, use_container_width=True, height=300)
    st.download_button("Descargar CSV TX (hist + forecast)", merged_tx.to_csv(index=False).encode("utf-8"),
                       file_name="tx_hist_forecast.csv", mime="text/csv")

# =========================
# Tab 3: Resumen por cuenta
# =========================
def per_account_forecast(df_metric: pd.DataFrame, horizon=12, scenario="Base") -> pd.DataFrame:
    rows = []
    # último periodo disponible global (para “último valor”)
    last_per = df_metric["period"].max()
    for cli, sub in df_metric.groupby("cliente"):
        s = sub.groupby("date", as_index=False)["value"].sum().sort_values("date")
        fc = fit_prophet_series(s, horizon=horizon)
        fc = apply_scenario(fc, scenario)
        last_val = float(sub.loc[sub["period"]==last_per, "value"].sum())
        next_month = fc["yhat"].iloc[0] if not fc.empty else np.nan
        sum_12 = fc["yhat"].sum() if not fc.empty else np.nan

        # bandera simple de riesgo: caída del último mes vs promedio 3 meses previos >40%
        risk = None
        if len(s) >= 4:
            last_hist = s["value"].iloc[-1]
            prev3 = s["value"].iloc[-4:-1].mean()
            risk = (prev3 > 0 and (last_hist < 0.6 * prev3))

        rows.append({
            "cliente": cli,
            "ultimo_periodo": last_per,
            "ultimo_valor": round(last_val, 2),
            "forecast_proximo_mes": round(float(next_month), 2) if pd.notna(next_month) else None,
            "forecast_12_meses": round(float(sum_12), 2) if pd.notna(sum_12) else None,
            "riesgo_caida": bool(risk) if risk is not None else False
        })
    out = pd.DataFrame(rows)
    return out.sort_values("forecast_12_meses", ascending=False)

with tabs[2]:
    st.subheader("Resumen por cuenta (MRR y Transacciones)")
    scenario_tbl = st.selectbox("Escenario tabla", ["Base", "Pesimista (–10%)", "Optimista (+10%)"], index=0, key="sc_tbl")
    horizon_tbl = st.slider("Horizonte tabla (meses)", 3, 18, DEFAULT_HORIZON, 12, key="hr_tbl")

    mrr_acc = per_account_forecast(fdata[fdata["metric"]=="MRR"], horizon=horizon_tbl, scenario=scenario_tbl)
    tx_acc  = per_account_forecast(fdata[fdata["metric"]=="Transactions"], horizon=horizon_tbl, scenario=scenario_tbl)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**MRR — Tabla por cuenta (ordenada por forecast 12m)**")
        st.dataframe(mrr_acc, use_container_width=True, height=420)
        st.download_button("Descargar CSV MRR por cuenta", mrr_acc.to_csv(index=False).encode("utf-8"),
                           file_name="mrr_por_cuenta.csv", mime="text/csv")
    with col2:
        st.markdown("**Transacciones — Tabla por cuenta (ordenada por forecast 12m)**")
        st.dataframe(tx_acc, use_container_width=True, height=420)
        st.download_button("Descargar CSV TX por cuenta", tx_acc.to_csv(index=False).encode("utf-8"),
                           file_name="tx_por_cuenta.csv", mime="text/csv")

# Debug opcional
with st.expander("Debug (últimos periodos detectados)"):
    per = sorted(data["period"].dropna().unique().tolist())[-6:]
    st.write(per)
