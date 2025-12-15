import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from prophet import Prophet

# ============================================================
# Config
# ============================================================
st.set_page_config(page_title="Instaleap — Forecast V2", layout="wide")

SHEET_ID = os.environ.get("SHEET_ID", "1ACX9OWNB0vHs8EpxeHxgByuPjDP9VC0E3k9b61V-i1I")
TX_GID = os.environ.get("TX_GID", "1584335131")  # gid (identificador numérico de la hoja) de Transactions
DEFAULT_HORIZON = 12
CACHE_TTL = 600  # TTL (tiempo de vida del cache) en segundos

# ============================================================
# URLs Google Sheets -> CSV
# ============================================================
def gviz_csv_url_by_sheet(sheet_id: str, sheet_name: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

def gviz_csv_url_by_gid(sheet_id: str, gid: str) -> str:
    # Export directo por gid (más robusto cuando el nombre de hoja no coincide)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

# ============================================================
# Helpers limpieza y parsing
# ============================================================
MONTH_SHORT = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Ene|Abr|Ago|Dic|Set)"  # CAMBIO MÍNIMO
MONTH_BASE_RE = re.compile(rf"^{MONTH_SHORT}-\d{{2}}$")
MONTH_COL_RE = re.compile(rf"^{MONTH_SHORT}-\d{{2}}(?:\.\d+)?$")

# CAMBIO MÍNIMO: mapeo de meses ES -> EN para poder parsear "Ene-25", etc.
MONTH_ES_TO_EN = {
    "Ene": "Jan",
    "Abr": "Apr",
    "Ago": "Aug",
    "Dic": "Dec",
    "Set": "Sep",  # a veces usan "Set" para septiembre
}

def ensure_datetime_naive(df: pd.DataFrame, col: str):
    """Convierte columna a datetime sin timezone (zona horaria)."""
    if col not in df.columns:
        return
    df[col] = pd.to_datetime(df[col], errors="coerce")
    try:
        df[col] = df[col].dt.tz_localize(None)
    except Exception:
        pass

def to_numeric_clean(s: pd.Series) -> pd.Series:
    """Convierte strings con $ , . y formatos ES/EN a float seguro."""
    s = s.astype(str).str.strip()
    s = s.replace({"": "0", "nan": "0", "None": "0"})
    s = s.str.replace(r"[\$\s]", "", regex=True)

    # Formato español: 12.345,67 o 12.345
    es_mask = s.str.match(r"^\d{1,3}(?:\.\d{3})+(?:,\d+)?$")
    s.loc[es_mask] = s.loc[es_mask].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)

    # Formato EN con miles: 12,345.67 -> quitar comas
    s = s.str.replace(",", "", regex=False)

    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def col_to_base_label(colname: str) -> str | None:
    """
    Normaliza un encabezado de columna a 'Mon-YY' (ej: 'Sep-25').
    Soporta:
    - Jan-25, Jan-25.1, ...
    - ISO '2025-09' / '2025-09-01'
    - serial de hoja (número grande tipo 45000+) como fecha
    - CAMBIO MÍNIMO: abreviaturas ES como Ene-25, Abr-25, Ago-25, Dic-25, Set-25
    """
    s = str(colname).strip()
    base = s.split(".")[0].strip()

    # CAMBIO MÍNIMO: si viene "Ene-25" lo convertimos a "Jan-25" para parsear
    if re.match(r"^[A-Za-z]{3}-\d{2}$", base):
        mon = base[:3]
        yy = base[4:]
        if mon in MONTH_ES_TO_EN:
            base = f"{MONTH_ES_TO_EN[mon]}-{yy}"

    # Caso 1: Jan-25
    if MONTH_BASE_RE.match(base):
        # si aún quedara en ES (por seguridad), lo convertimos aquí también
        mon = base[:3]
        if mon in MONTH_ES_TO_EN:
            base = f"{MONTH_ES_TO_EN[mon]}-{base[4:]}"
        return base

    # Caso 2: serial numérico (si está en rango razonable)
    if re.match(r"^\d+$", base):
        n = int(base)
        # serial típico de Sheets/Excel para fechas modernas
        if 30000 <= n <= 70000:
            dt = pd.Timestamp("1899-12-30") + pd.to_timedelta(n, unit="D")
            return dt.strftime("%b-%y")
        return None

    # Caso 3: ISO fecha
    try:
        dt = pd.to_datetime(base, errors="raise")
        return dt.strftime("%b-%y")
    except Exception:
        return None

def pick_best_month_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Cuando hay duplicados por mes (ej: Jan-25, Jan-25.1, ...),
    elige por cada mes base la columna con mayor suma numérica.
    Retorna: (selected_cols, base_labels_ordered)
    """
    groups: dict[str, list[str]] = {}

    for c in df.columns:
        base_label = col_to_base_label(c)
        if base_label is None:
            continue
        groups.setdefault(base_label, []).append(c)

    if not groups:
        return [], []

    selected_map: dict[str, str] = {}
    for base_label, cols in groups.items():
        best = max(cols, key=lambda col: float(to_numeric_clean(df[col]).sum()))
        selected_map[base_label] = best

    # ordenar cronológicamente
    base_labels = list(selected_map.keys())
    base_labels_sorted = sorted(base_labels, key=lambda b: pd.to_datetime(b, format="%b-%y"))

    selected_cols_sorted = [selected_map[b] for b in base_labels_sorted]
    return selected_cols_sorted, base_labels_sorted

def trim_trailing_zero_months(long_df: pd.DataFrame) -> pd.DataFrame:
    """Recorta meses finales con puro 0 para evitar meses fantasma."""
    sums = long_df.groupby("period", as_index=False)["value"].sum().sort_values("period")
    nz = sums[sums["value"] != 0]
    if not nz.empty:
        last = nz["period"].max()
        return long_df[long_df["period"] <= last]
    return long_df

def normalize_wide_to_long(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """
    Convierte hoja en ancho a long:
    Meta esperada (si existe): Country, type client, Nuevo, Zona, Cliente, Razon Social
    Meses: Jan-25, Feb-25... (con o sin .1, .2)

    CAMBIO MÍNIMO:
    - Si la hoja (ej: Transactions) no trae 'Cliente', el caller puede renombrar la primera columna a 'Cliente'
      antes de llamar esta función.
    """
    meta = ["Country", "type client", "Nuevo", "Zona", "Cliente", "Razon Social"]

    selected_cols, base_labels = pick_best_month_columns(df)

    keep_meta = [c for c in meta if c in df.columns]
    keep = keep_meta + selected_cols
    sub = df[keep].copy()

    # renombrar selected cols a base label (Jan-25.2 -> Jan-25)
    rename_map = {src: base for src, base in zip(selected_cols, base_labels)}
    sub.rename(columns=rename_map, inplace=True)

    # melt wide -> long (melt = pasar de ancho a largo)
    long = sub.melt(
        id_vars=keep_meta,
        value_vars=base_labels,
        var_name="period_label",
        value_name="value_raw",
    )

    long["value"] = to_numeric_clean(long["value_raw"])
    long.drop(columns=["value_raw"], inplace=True)

    # period/date
    long["period"] = pd.to_datetime(long["period_label"], format="%b-%y", errors="coerce").dt.to_period("M").astype(str)
    long["date"] = pd.to_datetime(long["period"] + "-01", errors="coerce")

    # renombrar metas
    ren = {
        "Country": "pais",
        "type client": "type",
        "Razon Social": "razon_social",
        "Cliente": "cliente",
        "Zona": "zona",
        "Nuevo": "nuevo",
    }
    long.rename(columns={k: v for k, v in ren.items() if k in long.columns}, inplace=True)

    long["metric"] = metric_name
    ensure_datetime_naive(long, "date")

    long = trim_trailing_zero_months(long)

    return long

# ============================================================
# Carga (con cache)
# ============================================================
@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def load_all() -> pd.DataFrame:
    # MRR por nombre de hoja
    mrr_raw = pd.read_csv(gviz_csv_url_by_sheet(SHEET_ID, "MRR"))

    # Transactions por gid (más robusto)
    tx_raw = pd.read_csv(gviz_csv_url_by_gid(SHEET_ID, TX_GID))

    # CAMBIO MÍNIMO (clave): tu hoja Transactions tiene en col A el nombre del cliente.
    # Si no viene una columna llamada "Cliente", renombramos la primera columna a "Cliente".
    if "Cliente" not in tx_raw.columns and len(tx_raw.columns) > 0:
        tx_raw = tx_raw.rename(columns={tx_raw.columns[0]: "Cliente"})

    mrr_long = normalize_wide_to_long(mrr_raw, "MRR")
    tx_long = normalize_wide_to_long(tx_raw, "Transactions")

    unified = pd.concat([mrr_long, tx_long], ignore_index=True)

    # columnas mínimas
    for col in ["pais", "type", "zona", "cliente", "razon_social", "period", "date", "value", "metric"]:
        if col not in unified.columns:
            unified[col] = np.nan

    ensure_datetime_naive(unified, "date")
    return unified

# ============================================================
# Forecast (Prophet)
# ============================================================
def fit_prophet_series(series_df: pd.DataFrame, horizon=12) -> pd.DataFrame:
    """
    series_df: columnas ['date','value']
    retorna: ['date','yhat','yhat_lower','yhat_upper']
    Nota: devuelve DF vacío con dtypes correctos para evitar errores en merge (unión de tablas).
    """
    s = series_df.dropna().copy()
    ensure_datetime_naive(s, "date")
    s = s.sort_values("date")

    empty = pd.DataFrame({
        "date": pd.to_datetime(pd.Series([], dtype="datetime64[ns]")),
        "yhat": pd.Series(dtype="float64"),
        "yhat_lower": pd.Series(dtype="float64"),
        "yhat_upper": pd.Series(dtype="float64"),
    })

    if s.empty or s["value"].sum() == 0 or len(s) < 3:
        return empty

    m = Prophet(
        interval_width=0.8,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    m.fit(s.rename(columns={"date": "ds", "value": "y"}))
    fut = m.make_future_dataframe(periods=horizon, freq="MS", include_history=False)
    fc = m.predict(fut)

    out = pd.DataFrame({
        "date": pd.to_datetime(fc["ds"]).dt.tz_localize(None),
        "yhat": fc["yhat"].astype(float),
        "yhat_lower": fc["yhat_lower"].astype(float),
        "yhat_upper": fc["yhat_upper"].astype(float),
    })
    ensure_datetime_naive(out, "date")
    return out

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

def plot_hist_fc(hist: pd.DataFrame, fc: pd.DataFrame, title: str, label_mode="extrema") -> go.Figure:
    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["value"], mode="lines", name="Histórico"))

    if not fc.empty:
        # intervalo
        fig.add_trace(go.Scatter(
            x=list(fc["date"]) + list(fc["date"][::-1]),
            y=list(fc["yhat_upper"]) + list(fc["yhat_lower"][::-1]),
            fill="toself",
            name="Intervalo",
            opacity=0.2,
            line=dict(width=0),
        ))

        mode = "lines"
        text = None

        if label_mode in ("extrema", "all"):
            y = fc["yhat"].to_numpy()
            if label_mode == "all":
                text = [f"{v:,.0f}" for v in y]
                mode = "lines+markers+text"
            else:
                # picos y valles simples
                win = 2
                y_s = pd.Series(y)
                is_max = (y_s.rolling(win, center=True).max() == y_s).fillna(False).to_numpy()
                is_min = (y_s.rolling(win, center=True).min() == y_s).fillna(False).to_numpy()
                text = [f"{y[i]:,.0f}" if (is_max[i] or is_min[i]) else None for i in range(len(y))]
                mode = "lines+markers+text"

        fig.add_trace(go.Scatter(
            x=fc["date"], y=fc["yhat"], mode=mode, name="Pronóstico",
            text=text, textposition="top center"
        ))

    fig.update_layout(
        title=title,
        height=460,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="",
        yaxis_title="",
    )
    return fig

# ============================================================
# UI
# ============================================================
st.title("Instaleap — Forecast V2 (MRR & Transacciones)")

with st.sidebar:
    st.markdown("### Fuente de datos")
    st.write(f"Sheet ID: `{SHEET_ID}`")
    st.write(f"Transactions gid: `{TX_GID}`")
    if st.button("Refrescar datos (cache)"):
        load_all.clear()
        st.success("Cache limpiado. Vuelve a cargar la página o cambia de tab.")

data = load_all()

# Health check rápido
vc = data["metric"].value_counts().to_dict()
colA, colB, colC = st.columns(3)
colA.metric("Filas MRR", int(vc.get("MRR", 0)))
colB.metric("Filas Transactions", int(vc.get("Transactions", 0)))
colC.metric("Total filas", int(len(data)))

with st.expander("Filtros (opcionales)", expanded=False):
    zonas = sorted([z for z in data["zona"].dropna().unique().tolist()])
    tipos = sorted([t for t in data["type"].dropna().unique().tolist()])
    clientes = sorted([c for c in data["cliente"].dropna().unique().tolist()])

    z_sel = st.multiselect("Zona", zonas, default=zonas)
    t_sel = st.multiselect("Tipo", tipos, default=tipos)
    c_sel = st.multiselect("Clientes (opcional)", clientes, default=[])

def apply_filters_ui(d: pd.DataFrame, z_sel, t_sel, c_sel) -> pd.DataFrame:
    f = d.copy()

    # Zona: conservar también filas sin zona (NaN = valor vacío)
    if z_sel:
        f = f[(f["zona"].isin(z_sel)) | (f["zona"].isna())]

    # Tipo: conservar también filas sin tipo (NaN = valor vacío)
    if t_sel:
        f = f[(f["type"].isin(t_sel)) | (f["type"].isna())]

    # Cliente: este sí debe filtrar estricto (porque Transactions sí tiene cliente)
    if c_sel:
        f = f[f["cliente"].isin(c_sel)]

    return f


fdata = apply_filters_ui(data, z_sel, t_sel, c_sel)

tabs = st.tabs(["MRR (hist + forecast)", "Transactions (hist + forecast)", "Resumen por cuenta"])

# ============================================================
# Tab 1: MRR
# ============================================================
with tabs[0]:
    st.subheader("MRR — Histórico + Pronóstico")

    scenario = st.selectbox("Escenario", ["Base", "Pesimista (–10%)", "Optimista (+10%)"], index=0, key="sc_mrr")
    horizon = st.slider("Horizonte (meses)", 3, 18, DEFAULT_HORIZON, 1, key="hz_mrr")

    m_hist = (fdata[fdata["metric"] == "MRR"]
              .groupby("date", as_index=False)["value"].sum()
              .sort_values("date"))
    ensure_datetime_naive(m_hist, "date")

    if m_hist.empty:
        st.warning("No hay datos de MRR con los filtros actuales.")
    else:
        m_fc = fit_prophet_series(m_hist, horizon=horizon)
        m_fc = apply_scenario(m_fc, scenario)

        fig = plot_hist_fc(m_hist, m_fc, title=f"MRR (escenario: {scenario})", label_mode="extrema")
        st.plotly_chart(fig, use_container_width=True)

        # merge seguro (merge = unión de tablas por llave)
        ensure_datetime_naive(m_fc, "date")
        merged = (m_hist.rename(columns={"value": "historico"})
                  .merge(m_fc, on="date", how="outer")
                  .sort_values("date"))
        merged["period"] = merged["date"].dt.to_period("M").astype(str)

        st.dataframe(merged, use_container_width=True, height=320)
        st.download_button(
            "Descargar CSV MRR (hist + forecast)",
            merged.to_csv(index=False).encode("utf-8"),
            file_name="mrr_hist_forecast.csv",
            mime="text/csv",
        )

# ============================================================
# Tab 2: Transactions
# ============================================================
with tabs[1]:
    st.subheader("Transactions — Histórico + Pronóstico")

    scenario_tx = st.selectbox("Escenario", ["Base", "Pesimista (–10%)", "Optimista (+10%)"], index=0, key="sc_tx")
    horizon_tx = st.slider("Horizonte (meses)", 3, 18, DEFAULT_HORIZON, 1, key="hz_tx")

    t_hist = (fdata[fdata["metric"] == "Transactions"]
              .groupby("date", as_index=False)["value"].sum()
              .sort_values("date"))
    ensure_datetime_naive(t_hist, "date")

    if t_hist.empty:
        st.warning("No hay datos de Transactions con los filtros actuales. Revisa el gid o el nombre de columnas de meses.")
        with st.expander("Debug Transactions"):
            tx_only = fdata[fdata["metric"] == "Transactions"].copy()
            st.write("Filas TX post-filtro:", len(tx_only))
            st.write("Periodos TX (últimos 8):", sorted(tx_only["period"].dropna().unique().tolist())[-8:])
    else:
        t_fc = fit_prophet_series(t_hist, horizon=horizon_tx)
        t_fc = apply_scenario(t_fc, scenario_tx)

        fig_tx = plot_hist_fc(t_hist, t_fc, title=f"Transactions (escenario: {scenario_tx})", label_mode="none")
        st.plotly_chart(fig_tx, use_container_width=True)

        ensure_datetime_naive(t_fc, "date")
        merged_tx = (t_hist.rename(columns={"value": "historico"})
                     .merge(t_fc, on="date", how="outer")
                     .sort_values("date"))
        merged_tx["period"] = merged_tx["date"].dt.to_period("M").astype(str)

        st.dataframe(merged_tx, use_container_width=True, height=320)
        st.download_button(
            "Descargar CSV Transactions (hist + forecast)",
            merged_tx.to_csv(index=False).encode("utf-8"),
            file_name="transactions_hist_forecast.csv",
            mime="text/csv",
        )

# ============================================================
# Tab 3: Resumen por cuenta
# ============================================================
@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def per_account_forecast(d: pd.DataFrame, metric: str, horizon: int, scenario: str) -> pd.DataFrame:
    """
    Calcula forecast por cuenta (por cliente).
    Nota: Prophet por cliente puede ser pesado (costoso computacionalmente) si hay muchos clientes.
    """
    dm = d[d["metric"] == metric].copy()
    if dm.empty:
        return pd.DataFrame(columns=[
            "cliente", "ultimo_periodo", "ultimo_valor", "forecast_proximo_mes", "forecast_12_meses", "riesgo_caida"
        ])

    last_per = dm["period"].max()
    rows = []

    for cli, sub in dm.groupby("cliente"):
        s = sub.groupby("date", as_index=False)["value"].sum().sort_values("date")
        ensure_datetime_naive(s, "date")

        fc = fit_prophet_series(s, horizon=horizon)
        fc = apply_scenario(fc, scenario)

        last_val = float(sub.loc[sub["period"] == last_per, "value"].sum())
        next_month = float(fc["yhat"].iloc[0]) if not fc.empty else np.nan
        sum_h = float(fc["yhat"].sum()) if not fc.empty else np.nan

        # regla simple de riesgo: último mes < 60% del promedio de los 3 previos
        risk = False
        if len(s) >= 4:
            last_hist = float(s["value"].iloc[-1])
            prev3 = float(s["value"].iloc[-4:-1].mean())
            risk = (prev3 > 0 and (last_hist < 0.6 * prev3))

        rows.append({
            "cliente": cli,
            "ultimo_periodo": last_per,
            "ultimo_valor": round(last_val, 2),
            "forecast_proximo_mes": round(next_month, 2) if pd.notna(next_month) else None,
            "forecast_12_meses": round(sum_h, 2) if pd.notna(sum_h) else None,
            "riesgo_caida": bool(risk),
        })

    out = pd.DataFrame(rows)
    # ordenar por forecast
    if "forecast_12_meses" in out.columns:
        out = out.sort_values("forecast_12_meses", ascending=False, na_position="last")
    return out

with tabs[2]:
    st.subheader("Resumen por cuenta — Forecast")

    scenario_tbl = st.selectbox("Escenario", ["Base", "Pesimista (–10%)", "Optimista (+10%)"], index=0, key="sc_tbl")
    horizon_tbl = st.slider("Horizonte (meses)", 3, 18, DEFAULT_HORIZON, 1, key="hz_tbl")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### MRR — por cuenta")
        mrr_acc = per_account_forecast(fdata, "MRR", horizon_tbl, scenario_tbl)
        st.dataframe(mrr_acc, use_container_width=True, height=420)
        st.download_button(
            "Descargar CSV MRR por cuenta",
            mrr_acc.to_csv(index=False).encode("utf-8"),
            file_name="mrr_por_cuenta.csv",
            mime="text/csv",
        )

    with col2:
        st.markdown("### Transactions — por cuenta")
        tx_acc = per_account_forecast(fdata, "Transactions", horizon_tbl, scenario_tbl)
        st.dataframe(tx_acc, use_container_width=True, height=420)
        st.download_button(
            "Descargar CSV Transactions por cuenta",
            tx_acc.to_csv(index=False).encode("utf-8"),
            file_name="transactions_por_cuenta.csv",
            mime="text/csv",
        )

with st.expander("Debug general (últimos periodos detectados)"):
    for met in ["MRR", "Transactions"]:
        dm = data[data["metric"] == met]
        periods = sorted(dm["period"].dropna().unique().tolist())
        st.write(f"{met}: últimos periodos:", periods[-8:])
        st.write(f"{met}: últimos 3 meses (total):",
                 dm.groupby("period")["value"].sum().sort_index().tail(3))
