
import streamlit as st, pandas as pd, plotly.express as px
from streamlit_plotly_events import plotly_events

def normalize_country_name(name: str) -> str:
    if not isinstance(name, str): return ""
    s = name.strip()
    translations = {
        "estados unidos": "United States",
        "méxico": "Mexico", "mexico": "Mexico",
        "brasil": "Brazil",
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
        "puerto rico": "Puerto Rico"
    }
    key = s.lower()
    return translations.get(key, s)

def map_by_country(unified: pd.DataFrame, make_forecast, plot_series_with_forecast):
    st.subheader("Análisis por país (mapa interactivo)")
    if "pais" not in unified.columns or unified["pais"].dropna().empty:
        st.caption("No hay columna de país disponible en los datos."); return
    m = unified[unified["metric"]=="MRR"].copy()
    if m.empty:
        st.caption("No hay datos de MRR para mapa."); return

    m["pais_norm"] = m["pais"].apply(normalize_country_name)
    last_p = m["period"].max()
    agg = m[m["period"]==last_p].groupby("pais_norm", as_index=False)["value"].sum().rename(columns={"value":"MRR"})
    if agg.empty:
        st.caption("No hay datos consolidados para el último periodo."); return

    fig = px.choropleth(
        agg,
        locations="pais_norm",
        color="MRR",
        locationmode="country names",
        hover_name="pais_norm",
        color_continuous_scale=px.colors.sequential.Blues
    )
    fig.update_traces(customdata=agg[["pais_norm"]].values)
    fig.update_geos(
        showcoastlines=True,
        showcountries=True,
        countrycolor="rgba(255,255,255,0.35)",
        showland=True,
        landcolor="rgba(255,255,255,0.03)",
        projection_type="natural earth",
        fitbounds="locations"
    )
    fig.update_layout(
        height=520,
        margin=dict(l=8, r=8, t=24, b=8),
        dragmode="pan",
        coloraxis_colorbar=dict(title="MRR")
    )

    selected = plotly_events(
        fig,
        click_event=True,
        select_event=False,
        hover_event=False,
        override_height=520,
        override_width="100%",
        key="country_map_events"
    )

    st.caption("Tip: usa zoom y pan. Haz clic en un país para ver el detalle.")

    sel_country = None
    if selected:
        cd = selected[0].get("customdata")
        if isinstance(cd, (list, tuple)) and len(cd):
            sel_country = cd[0]
        else:
            idx = selected[0].get("pointNumber")
            if idx is not None and 0 <= idx < len(agg):
                sel_country = agg.iloc[idx]["pais_norm"]

    all_countries = agg["pais_norm"].sort_values().unique().tolist()
    sel_country = st.selectbox("País", all_countries, index=(all_countries.index(sel_country) if sel_country in all_countries else 0), key="country_select")

    country_mask = unified["pais"].apply(normalize_country_name) == sel_country
    country_df = unified[(unified["metric"]=="MRR") & country_mask].copy()
    if country_df.empty:
        st.caption("Sin datos para el país seleccionado."); return

    agg_period = country_df.groupby("period", as_index=False)["value"].sum().sort_values("period")
    last_p = agg_period["period"].max()
    periods = agg_period["period"].tolist()
    prev_p = periods[-2] if len(periods) >= 2 else None
    try: yoy_p = str((pd.Period(last_p) - 12))
    except Exception: yoy_p = None

    total_last = float(agg_period.loc[agg_period["period"]==last_p, "value"].sum()) if last_p else 0.0
    total_prev = float(agg_period.loc[agg_period["period"]==prev_p, "value"].sum()) if prev_p else 0.0
    total_yoy_prev = float(agg_period.loc[agg_period["period"]==yoy_p, "value"].sum()) if yoy_p else 0.0

    mom = ((total_last - total_prev) / total_prev * 100.0) if total_prev>0 else None
    yoy = ((total_last - total_yoy_prev) / total_yoy_prev * 100.0) if total_yoy_prev>0 else None
    arr = total_last * 12.0

    st.write(f"### {sel_country}")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric(f"MRR ({last_p})", f"${total_last:,.0f}")
    c2.metric("ARR", f"${arr:,.0f}")
    c3.metric("MoM", f"{mom:,.1f}%" if mom is not None else "—")
    c4.metric("YoY", f"{yoy:,.1f}%" if yoy is not None else "—")

    last_clients = country_df[country_df["period"]==last_p].groupby("cliente", as_index=False)["value"].sum().sort_values("value", ascending=False)
    st.write(f"**Cuentas activas ({len(last_clients)})**")
    st.dataframe(last_clients.rename(columns={"value":"MRR"}), use_container_width=True)

    hist_df, fcst_df = make_forecast(unified[country_mask], metric="MRR", horizon=6, aggregate_first=True, by=[])
    st.plotly_chart(plot_series_with_forecast(hist_df, fcst_df, f"{sel_country} - MRR Histórico + Pronóstico"), use_container_width=True)
