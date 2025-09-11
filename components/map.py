# --- components/map.py (reemplazo completo) ---
import streamlit as st, pandas as pd, plotly.express as px
from typing import Dict, List, Optional

# Intenta usar clic en el mapa si está disponible
try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None

# 1) Normalización básica de nombres de país (ajusta/añade si hace falta)
def normalize_country_name(name: str) -> str:
    if not isinstance(name, str): return ""
    s = name.strip()
    translations = {
        "estados unidos": "United States",
        "méxico": "Mexico", "mexico": "Mexico",
        "brasil": "Brazil", "brasil ": "Brazil",
        "españa": "Spain",
        "reino unido": "United Kingdom",
        "alemania": "Germany",
        "colombia": "Colombia",
        "perú": "Peru", "peru": "Peru",
        "chile": "Chile",
        "argentina": "Argentina",
        "ecuador": "Ecuador",
        "uruguay": "Uruguay",
        "paraguay": "Paraguay",
        "bolivia": "Bolivia",
        "costa rica": "Costa Rica",
        "panamá": "Panama", "panama": "Panama",
        "guatemala": "Guatemala",
        "honduras": "Honduras",
        "el salvador": "El Salvador",
        "nicaragua": "Nicaragua",
        "república dominicana": "Dominican Republic", "republica dominicana": "Dominican Republic",
        "puerto rico": "Puerto Rico",
        # añade más si tus datos lo requieren
    }
    key = s.lower()
    return translations.get(key, s)

# 2) Clusters / regiones (ajusta según tu negocio)
country_to_cluster: Dict[str, str] = {
    # Sudamérica
    "Argentina":"Sudamérica","Bolivia":"Sudamérica","Brazil":"Sudamérica","Chile":"Sudamérica",
    "Colombia":"Sudamérica","Ecuador":"Sudamérica","Paraguay":"Sudamérica","Peru":"Sudamérica","Uruguay":"Sudamérica",
    # Norteamérica
    "United States":"Norteamérica","Mexico":"Norteamérica","Canada":"Norteamérica",
    # Centro/Caribe
    "Panama":"Centro/Caribe","Costa Rica":"Centro/Caribe","Guatemala":"Centro/Caribe","Honduras":"Centro/Caribe",
    "El Salvador":"Centro/Caribe","Nicaragua":"Centro/Caribe","Dominican Republic":"Centro/Caribe","Puerto Rico":"Centro/Caribe",
    # Europa (ejemplos)
    "Spain":"Europa","United Kingdom":"Europa","Germany":"Europa","France":"Europa","Italy":"Europa",
    # ... añade el resto si necesitas
}

def map_by_country(unified: pd.DataFrame):
    st.subheader("Análisis por país (mapa interactivo)")

    if "pais" not in unified.columns or unified["pais"].dropna().empty:
        st.caption("No hay columna de país disponible en los datos."); return

    # --- Filtros UI para Clúster ---
    st.markdown("##### Opciones del mapa")
    cluster_enabled = st.checkbox("Filtrar por clúster/region", value=True)
    # Normaliza país y asocia clúster
    data = unified.copy()
    data["pais_norm"] = data["pais"].apply(normalize_country_name)
    data["cluster"] = data["pais_norm"].map(country_to_cluster).fillna("Otros")

    clusters = sorted(data["cluster"].dropna().unique().tolist())
    sel_clusters: List[str] = clusters
    if cluster_enabled:
        sel_clusters = st.multiselect("Clusters", options=clusters, default=clusters, help="Filtra qué regiones ver en el mapa.")
        if sel_clusters:
            data = data[data["cluster"].isin(sel_clusters)]

    # --- Tomamos MRR último periodo por país ---
    m = data[data["metric"] == "MRR"].copy()
    if m.empty:
        st.caption("No hay datos de MRR para mapa con los filtros actuales."); return

    last_p = m["period"].max()
    last_df = m[m["period"] == last_p]
    agg = (last_df.groupby(["pais_norm","cluster"], as_index=False)
           .agg(MRR=("value","sum"), clientes=("cliente","nunique")))

    if agg.empty:
        st.caption("No hay datos consolidados para el último periodo."); return

    # Cálculo de variaciones MoM/YoY para hover
    # (no afectan el color; solo enriquecen el tooltip)
    def growths_for_country(df_country: pd.DataFrame):
        by_p = df_country.groupby("period", as_index=False)["value"].sum().sort_values("period")
        # prev (mes anterior)
        periods = by_p["period"].tolist()
        if not periods: return None, None
        last = by_p.iloc[-1]["value"]
        prev = by_p.iloc[-2]["value"] if len(by_p) >= 2 else None
        yoyp = str((pd.Period(by_p.iloc[-1]["period"]) - 12)) if len(by_p) else None
        yoy = by_p.loc[by_p["period"]==yoyp, "value"].sum() if yoyp in by_p["period"].values else None
        mom = ((last - prev)/prev*100) if prev and prev>0 else None
        yoyg = ((last - yoy)/yoy*100) if yoy and yoy>0 else None
        return mom, yoyg

    # Prepara dataframe para choropleth + hover data
    hover_rows = []
    for pais in agg["pais_norm"].unique():
        mom, yoy = growths_for_country(m[m["pais_norm"]==pais])
        row = agg[agg["pais_norm"]==pais].iloc[0].to_dict()
        row["crec_mom_%"] = None if mom is None else round(mom,1)
        row["crec_yoy_%"] = None if yoy is None else round(yoy,1)
        hover_rows.append(row)
    plot_df = pd.DataFrame(hover_rows)

    # --- Choropleth ---
    fig = px.choropleth(
        plot_df,
        locations="pais_norm",
        locationmode="country names",
        color="MRR",
        hover_name="pais_norm",
        hover_data={
            "MRR":":,.0f",
            "clientes":True,
            "crec_mom_%":True,
            "crec_yoy_%":True,
            "cluster":True
        },
        color_continuous_scale="Blues",
        scope="world",
        labels={"MRR":"MRR"}
    )

    # Que se vea BIEN en tema claro/oscuro
    fig.update_geos(
        showcountries=True,
        countrycolor="#555",  # bordes visibles en fondo claro
        showcoastlines=True,
        coastlinecolor="#666",
        showland=True,
        landcolor="#E5ECF6",  # gris azulado claro
        projection_type="equirectangular",  # estable, sin “globo”
        fitbounds="locations"
    )
    fig.update_layout(
        margin=dict(l=8,r=8,t=24,b=8),
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(title="MRR")
    )
    # Hover bien formateado
    fig.update_traces(
        hovertemplate="<b>%{location}</b><br>MRR: $%{z:,.0f}<br>Clientes: %{customdata[0]}<br>MoM: %{customdata[1]}%<br>YoY: %{customdata[2]}%<br>Cluster: %{customdata[3]}<extra></extra>",
        customdata=plot_df[["clientes","crec_mom_%","crec_yoy_%","cluster"]].values
    )

    # CLIC (si lib disponible). Si no, selectbox.
    clicked_country: Optional[str] = None
    if plotly_events is not None:
        selected = plotly_events(fig, click_event=True, select_event=False, hover_event=False,
                                 override_height=520, override_width="100%", key="country_map_events_v2")
        st.caption("Tip: usa zoom/pan. Clic en un país para detalle (si el clic no funciona, usa el selector).")
        if selected:
            idx = selected[0].get("pointNumber")
            if idx is not None and 0 <= idx < len(plot_df):
                clicked_country = plot_df.iloc[idx]["pais_norm"]

    st.plotly_chart(fig, use_container_width=True)

    # Selector de respaldo o para filtrar explícitamente
    all_countries = plot_df["pais_norm"].sort_values().unique().tolist()
    sel_country = st.selectbox("País", options=all_countries,
                               index=(all_countries.index(clicked_country) if clicked_country in all_countries else 0))

    # KPIs por país seleccionado
    country_mask = data["pais_norm"] == sel_country
    m_country = data[(data["metric"]=="MRR") & country_mask]
    if m_country.empty:
        st.caption("Sin datos para el país seleccionado."); return

    ser = m_country.groupby("period", as_index=False)["value"].sum().sort_values("period")
    last_p = ser["period"].max()
    prev_p = ser["period"].iloc[-2] if len(ser)>=2 else None
    yoy_p = str((pd.Period(last_p) - 12)) if last_p else None

    total_last = float(ser.loc[ser["period"]==last_p, "value"].sum()) if last_p else 0.0
    total_prev = float(ser.loc[ser["period"]==prev_p, "value"].sum()) if prev_p else 0.0
    total_yoy_prev = float(ser.loc[ser["period"]==yoy_p, "value"].sum()) if yoy_p else 0.0
    mom = ((total_last - total_prev)/total_prev*100) if total_prev>0 else None
    yoy = ((total_last - total_yoy_prev)/total_yoy_prev*100) if total_yoy_prev>0 else None
    arr = total_last * 12.0

    st.markdown(f"### {sel_country}")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric(f"MRR ({last_p})", f"${total_last:,.0f}")
    c2.metric("ARR", f"${arr:,.0f}")
    c3.metric("MoM", f"{mom:,.1f}%" if mom is not None else "—")
    c4.metric("YoY", f"{yoy:,.1f}%" if yoy is not None else "—")

    # Top cuentas del último periodo (solo MRR)
    last_clients = (m_country[m_country["period"]==last_p]
                    .groupby("cliente", as_index=False)["value"].sum()
                    .sort_values("value", ascending=False)
                    .rename(columns={"value":"MRR"}))
    st.write(f"**Cuentas activas ({len(last_clients)})**")
    st.dataframe(last_clients, use_container_width=True)
