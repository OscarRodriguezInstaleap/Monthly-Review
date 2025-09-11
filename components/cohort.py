
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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
    total_yoy = stack["yoy"].sum(); total_now = stack["now"].sum()
    fig = go.Figure()
    for _, row in stack.iterrows():
        fig.add_trace(go.Bar(x=["Año anterior", "Mes seleccionado"], y=[row["yoy"], row["now"]], name=str(row["cliente_bucket"])))
    fig.add_trace(go.Scatter(x=["Año anterior"], y=[total_yoy], mode="text", text=[f"${total_yoy:,.0f}"], textposition="top center", showlegend=False))
    fig.add_trace(go.Scatter(x=["Mes seleccionado"], y=[total_now], mode="text", text=[f"${total_now:,.0f}"], textposition="top center", showlegend=False))
    fig.update_layout(barmode="stack", title=f"Cohorte {sel_p}: clientes activos vs mismo mes del año anterior",
                      xaxis_title="", yaxis_title="MRR", height=420, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)
