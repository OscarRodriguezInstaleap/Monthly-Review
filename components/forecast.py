import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

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

def make_forecast(df: pd.DataFrame, metric: str, horizon: int, aggregate_first: bool, by=None):
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
        latest_meta = m.groupby("cliente", as_index=False).agg({"type":"last","zona":"last","nuevo":"last","pais":"last"})
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
    return fig
