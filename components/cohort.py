# components/cohort.py (v5)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def yoy_cohort_stacked(df: pd.DataFrame):
    st.subheader("MRR por cohorte (clientes activos en el mes seleccionado)")

    m = df[df["metric"] == "MRR"].copy()
    if m.empty:
        st.caption("No hay datos de MRR para gráfico de cohorte.")
        return

    # 1) Selección de mes (cohorte) y su par YoY
    all_periods = sorted(m["period"].dropna().unique().tolist())
    sel_p = st.selectbox("Selecciona el mes (cohorte)", options=all_periods, index=len(all_periods) - 1, key="cohort_period")
    yoy_p = str((pd.Period(sel_p) - 12))

    # 2) Definir la cohorte: clientes activos en el mes seleccionado
    cohort = sorted(m[(m["period"] == sel_p) & (m["value"] > 0)]["cliente"].unique().tolist())

    # 3) Totales por cliente en mes seleccionado vs mismo mes año anterior
    now_df = (
        m[(m["period"] == sel_p) & (m["cliente"].isin(cohort))]
        .groupby("cliente", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "now"})
    )
    yoy_df = (
        m[(m["period"] == yoy_p) & (m["cliente"].isin(cohort))]
        .groupby("cliente", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "yoy"})
    )
    det = now_df.merge(yoy_df, on="cliente", how="outer").fillna(0.0)
    det["diff"] = det["now"] - det["yoy"]
    det["diff_%"] = det.apply(lambda r: (r["diff"] / r["yoy"] * 100) if r["yoy"] > 0 else None, axis=1)

    # 4) Resumen general
    total_now = det["now"].sum()
    total_yoy = det["yoy"].sum()
    total_diff = total_now - total_yoy

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Total {sel_p}", f"${total_now:,.0f}")
    c2.metric(f"Total {yoy_p}", f"${total_yoy:,.0f}")
    c3.metric("Diferencia", f"${total_diff:,.0f}")

    # 5) Gráfica apilada (TopN + 'Otros') con totales como labels
    topN = st.slider("Top N clientes para apilar (por valor del mes seleccionado)", 5, 25, 12)
    det["rank"] = det["now"].rank(ascending=False, method="first")
    top_clients = det.sort_values("now", ascending=False).head(topN)["cliente"].tolist()
    det["cliente_bucket"] = det["cliente"].where(det["cliente"].isin(top_clients), other="Otros")
    stack = det.groupby("cliente_bucket", as_index=False)[["now", "yoy"]].sum()

    total_yoy_s = stack["yoy"].sum()
    total_now_s = stack["now"].sum()

    fig = go.Figure()
    for _, row in stack.iterrows():
        fig.add_trace(
            go.Bar(
                x=["Año anterior", "Mes seleccionado"],
                y=[row["yoy"], row["now"]],
                name=str(row["cliente_bucket"]),
            )
        )

    # Totales encima de cada barra (label)
    fig.add_trace(
        go.Scatter(
            x=["Año anterior"],
            y=[total_yoy_s],
            mode="text",
            text=[f"${total_yoy_s:,.0f}"],
            textposition="top center",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=["Mes seleccionado"],
            y=[total_now_s],
            mode="text",
            text=[f"${total_now_s:,.0f}"],
            textposition="top center",
            showlegend=False,
        )
    )

    fig.update_layout(
        barmode="stack",
        title=f"Cohorte {sel_p}: clientes activos vs mismo mes del año anterior",
        xaxis_title="",
        yaxis_title="MRR",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 6) Tabla colapsable por cliente con diferencia y %
    with st.expander("📊 Detalle por cliente (diferencia y %)", expanded=False):
        det_sorted = det.sort_values("diff", ascending=False)
        st.dataframe(det_sorted, use_container_width=True)
        csv = det_sorted.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar CSV",
            data=csv,
            file_name=f"cohorte_{sel_p}_detalle.csv",
            mime="text/csv",
            use_container_width=True,
        )
