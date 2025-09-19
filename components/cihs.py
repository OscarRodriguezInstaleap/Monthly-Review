# components/cihs.py (v5)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from components.forecast import make_forecast, plot_series_with_forecast

def cihs_section(unified: pd.DataFrame):
    st.subheader("CIHS (Adopción de funcionalidades) — Total")

    c = unified[unified["metric"] == "CIHS"].copy()
    if c.empty:
        st.caption("No se encontraron datos de CIHS.")
        return

    # 1) Serie total (suma, no promedio)
    agg = c.groupby("date", as_index=False)["value"].sum().sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg["date"], y=agg["value"], mode="lines", name="CIHS total"))
    fig.update_layout(
        title="CIHS total (global)",
        xaxis_title="Fecha",
        yaxis_title="CIHS (suma)",
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 2) Pronóstico CIHS consolidado
    st.markdown("##### Pronóstico de CIHS (consolidado)")
    hist_df, fcst_df = make_forecast(unified, metric="CIHS", horizon=6, aggregate_first=True, by=[])
    st.plotly_chart(
        plot_series_with_forecast(hist_df, fcst_df, "CIHS Histórico + Pronóstico", label_mode="extrema"),
        use_container_width=True,
    )

    # 3) Tabla resumen descargable (histórico + yhat + intervalos)
    if not hist_df.empty or not fcst_df.empty:
        agg_hist = hist_df.groupby("date", as_index=False)["value"].sum() if not hist_df.empty else pd.DataFrame()
        agg_fcst = (
            fcst_df.groupby(["date", "period"], as_index=False)[["yhat", "yhat_lower", "yhat_upper"]].sum()
            if not fcst_df.empty else pd.DataFrame()
        )
        merged = pd.merge(agg_hist, agg_fcst, on="date", how="outer").sort_values("date")
        if "date" in merged.columns:
            merged["period"] = merged["date"].dt.to_period("M").astype(str)

        with st.expander("📥 Descargar/inspeccionar datos de pronóstico (CIHS)"):
            st.dataframe(merged, use_container_width=True)
            csv = merged.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar CSV",
                data=csv,
                file_name="cihs_consolidado_forecast.csv",
                mime="text/csv",
                use_container_width=True,
            )
