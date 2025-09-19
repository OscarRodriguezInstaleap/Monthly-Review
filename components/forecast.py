# components/forecast.py (v5)

import pandas as pd
import numpy as np
import plotly.graph_objects as go

try:
    from prophet import Prophet
except Exception:
    Prophet = None


def _naive_forecast(ts: pd.DataFrame, horizon_months: int = 12) -> pd.DataFrame:
    last = ts["value"].iloc[-1] if not ts.empty else 0.0
    future_dates = pd.date_range(
        ts["date"].max() + pd.offsets.MonthBegin(1) if not ts.empty else pd.to_datetime("today"),
        periods=horizon_months,
        freq="MS",
    )
    out = pd.DataFrame({"date": future_dates, "yhat": last, "yhat_lower": last, "yhat_upper": last})
    out["period"] = out["date"].dt.to_period("M").astype(str)
    return out


def fit_prophet(ts: pd.DataFrame, horizon_months: int = 12, yearly_seasonality: bool = True) -> pd.DataFrame:
    # Fallback si no hay datos suficientes o Prophet no está disponible
    if (
        ts.empty
        or ts["value"].sum() == 0
        or ts["value"].nunique() <= 1
        or Prophet is None
    ):
        return _naive_forecast(ts, horizon_months=horizon_months)

    mod = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_range=0.9,
    )
    dfp = ts.rename(columns={"date": "ds", "value": "y"}).copy()
    mod.fit(dfp)
    future = mod.make_future_dataframe(periods=horizon_months, freq="MS")
    fcst = mod.predict(future)
    out = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date"})
    out = out[out["date"] > ts["date"].max()]  # solo futuro
    out["period"] = out["date"].dt.to_period("M").astype(str)
    return out


def make_forecast(df: pd.DataFrame, metric: str, horizon: int, aggregate_first: bool, by=None):
    """
    Devuelve (hist_df, fcst_df)
    - hist_df: histórico (date, value, [group_key opcional])
    - fcst_df: pronóstico (date, yhat, yhat_lower, yhat_upper, [group_key/agrupaciones])
    """
    m = df[df["metric"] == metric].copy()
    if m.empty:
        return pd.DataFrame(), pd.DataFrame()

    if by is None:
        by = []
    group_cols = by + ["date"]
    agg = m.groupby(group_cols, as_index=False)["value"].sum().sort_values("date")

    if aggregate_first or not by:
        results = []
        hist_parts = []
        # Un solo agregado global, o por claves en "by"
        groups = agg.groupby(by) if by else [([], agg)]
        for key, g in groups:
            hist = g[["date", "value"]].sort_values("date")
            fcst = fit_prophet(hist, horizon_months=horizon, yearly_seasonality=True)
            group_key = "|".join(map(str, key)) if by else "TOTAL"
            fcst["group_key"] = group_key
            hist["group_key"] = group_key
            results.append(fcst)
            hist_parts.append(hist)
        return pd.concat(hist_parts, ignore_index=True), pd.concat(results, ignore_index=True)

    # Pronostica por cliente y luego agrega por "by"
    results = []
    for client, g in m.groupby("cliente"):
        series = g.groupby("date", as_index=False)["value"].sum().sort_values("date")
        fcst = fit_prophet(series, horizon_months=horizon, yearly_seasonality=True)
        fcst["cliente"] = client
        results.append(fcst)

    fcst_all = pd.concat(results, ignore_index=True)
    # Adjunta metadatos recientes para poder agrupar
    latest_meta = m.groupby("cliente", as_index=False).agg({"type": "last", "zona": "last", "nuevo": "last", "pais": "last"})
    fcst_all = fcst_all.merge(latest_meta, on="cliente", how="left")
    # Agrega por las claves solicitadas
    fcst_group = fcst_all.groupby(by + ["date", "period"], as_index=False)[["yhat", "yhat_lower", "yhat_upper"]].sum()

    hist_df = agg.rename(columns={"value": "y"}).rename(columns={"y": "value"}).copy()
    return hist_df, fcst_group


def _extrema_mask(y: np.ndarray, window: int = 3):
    """
    Marca picos y valles locales comparando con vecinos en una ventana.
    Devuelve dos boolean arrays: (is_max, is_min)
    """
    n = len(y)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)

    is_max = np.zeros(n, dtype=bool)
    is_min = np.zeros(n, dtype=bool)

    for i in range(n):
        l = max(0, i - window)
        r = min(n, i + window + 1)
        neigh = y[l:r]
        if len(neigh) > 0:
            # Etiquetamos solo si es extremo único en la ventana
            if y[i] == np.max(neigh) and np.sum(neigh == y[i]) == 1:
                is_max[i] = True
            if y[i] == np.min(neigh) and np.sum(neigh == y[i]) == 1:
                is_min[i] = True

    return is_max, is_min


def plot_series_with_forecast(
    hist_df: pd.DataFrame,
    fcst_df: pd.DataFrame,
    title: str,
    label_mode: str = "extrema",
):
    """
    label_mode: "none" | "extrema" | "all"
      - "extrema": añade labels solo en picos/valles del pronóstico (recomendado).
      - "all": etiqueta todos los puntos del pronóstico (puede saturar).
      - "none": sin etiquetas.
    """
    fig = go.Figure()

    # Histórico
    if not hist_df.empty:
        agg_hist = hist_df.groupby("date", as_index=False)["value"].sum()
        fig.add_trace(
            go.Scatter(x=agg_hist["date"], y=agg_hist["value"], name="Histórico", mode="lines")
        )

    # Pronóstico
    if not fcst_df.empty:
        agg_fcst = fcst_df.groupby("date", as_index=False)[["yhat", "yhat_lower", "yhat_upper"]].sum()

        # Banda de intervalo
        fig.add_trace(
            go.Scatter(
                x=list(agg_fcst["date"]) + list(agg_fcst["date"][::-1]),
                y=list(agg_fcst["yhat_upper"]) + list(agg_fcst["yhat_lower"][::-1]),
                fill="toself",
                name="Intervalo",
                opacity=0.2,
                line=dict(width=0),
            )
        )

        # Línea de pronóstico con labels opcionales
        text = None
        textpos = "top center"
        mode = "lines"

        if label_mode in ("all", "extrema"):
            y = agg_fcst["yhat"].to_numpy()
            if label_mode == "all":
                text = [f"{v:,.0f}" for v in y]
            else:
                is_max, is_min = _extrema_mask(y, window=2)
                text = []
                for i, v in enumerate(y):
                    if is_max[i] or is_min[i]:
                        text.append(f"{v:,.0f}")
                    else:
                        text.append(None)
            mode = "lines+markers+text"

        fig.add_trace(
            go.Scatter(
                x=agg_fcst["date"],
                y=agg_fcst["yhat"],
                name="Pronóstico",
                mode=mode,
                text=text,
                textposition=textpos,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title="",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
