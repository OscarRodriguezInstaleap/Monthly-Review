import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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
