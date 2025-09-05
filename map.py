import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events

def normalize_country_name(name: str) -> str:
    if not isinstance(name, str): return ""
    s = name.strip()
    translations = {
        "estados unidos": "United States",
        "méxico": "Mexico",
        "mexico": "Mexico",
        "brasil": "Brazil",
        "españa": "Spain",
        "reino unido": "United Kingdom",
        "alemania": "Germany",
        "colombia": "Colombia",
        "perú": "Peru",
        "peru": "Peru",
        "chile": "Chile",
        "argentina": "Argentina",
        "ecuador": "Ecuador",
        "uruguay": "Uruguay",
        "paraguay": "Paraguay",
        "bolivia": "Bolivia",
        "costa rica": "Costa Rica",
        "panamá": "Panama",
        "panama": "Panama",
        "guatemala": "Guatemala",
        "honduras": "Honduras",
        "el salvador": "El Salvador",
        "nicaragua": "Nicaragua",
        "república dominicana": "Dominican Republic",
        "republica dominicana": "Dominican Republic",
        "puerto rico": "Puerto Rico"
    }
    key = s.lower()
    return translations.get(key, s)

def map_by_country(unified: pd.DataFrame, make_forecast, plot_series_with_forecast):
    st.subheader("Análisis por país (mapa interactivo)")
    if "pais" not in unified.columns or unified["pais"].dropna().empty:
        st.caption("No hay columna de país disponible en los datos.")
        return
    m = unified[unified["metric"]=="MRR"].copy()
    if m.empty:
        st.caption("No hay datos de MRR para mapa."); return
    m["pais_norm"] = m["pais"].apply(normalize_country_name)
    last_p = m["period"].max()
    agg = m[m["period"]==last_p].groupby("pais_norm", as_index=False)["value"].sum().rename(columns={"value":"MRR"})
    if agg.empty:
        st.caption("No hay datos consolidados para el último periodo."); return

    fig = px.choropleth(agg, locations="pais_norm", color="MRR",
                        locationmode="country names", hover_name="pais_norm",
                        color_continuous_scale="Blues")
    fig.update_layout(height=460, margin=dict(l=10,r=10,t=10,b=0))
    selected = plotly_events(fig, click_event=True, hover_event=False, override_height=460, override_width="100%")
    st.caption("Tip: usa zoom y pan del mapa. Haz clic en un país para ver el detalle.")

    all_countries = sorted(agg["pais_norm"].unique().tolist())
    default_country = selected[0]["location"] if selected else None
    sel_country = st.selectbox("País", options=all_countries, index=all_countries.index(default_country) if default_country in all_countries else 0)

    country_df = unified[(unified["metric"]=="MRR") & (unified["pais"].apply(normalize_country_name)==sel_country)].copy()
    if country_df.empty:
        st.caption("Sin datos para el país seleccionado."); return

    agg_period = country_df.groupby("period", as_index=False)["value"].sum().sort_values("period")
    last_p = agg_period["period"].max()
    periods = sorted(agg_period["period"].unique().tolist())
    idx = periods.index(last_p) if last_p in periods else -1
    prev_p = periods[idx-1] if idx-1>=0 else None
    try: yoy_p = str((pd.Period(last_p) - 12))
    except Exception: yoy_p = None

    total_last = float(agg_period[agg_period["period"]==last_p]["value"].sum()) if last_p else 0.0
    total_prev = float(agg_period[agg_period["period"]==prev_p]["value"].sum()) if prev_p else 0.0
    total_yoy_prev = float(agg_period[agg_period["period"]==yoy_p]["value"].sum()) if yoy_p else 0.0

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

    hist_df, fcst_df = make_forecast(unified[unified["pais"].apply(normalize_country_name)==sel_country], metric="MRR", horizon=6, aggregate_first=True, by=[])
    st.plotly_chart(plot_series_with_forecast(hist_df, fcst_df, f"{sel_country} - MRR Histórico + Pronóstico"), use_container_width=True)
