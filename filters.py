import streamlit as st
import pandas as pd

def filters_ui(df: pd.DataFrame):
    st.subheader("Filtros")
    with st.container():
        cols = st.columns(4)

        zones_all = sorted([z for z in df["zona"].dropna().unique() if z])
        sel_all_z = cols[0].checkbox("Seleccionar todo (Zona)", value=True, key="all_zona")
        sel_zones = cols[0].multiselect("Zona", zones_all, default=zones_all if sel_all_z else [])

        df_z = df[df["zona"].isin(sel_zones)] if sel_zones else df

        types_all = sorted([t for t in df_z["type"].dropna().unique() if t])
        sel_all_t = cols[1].checkbox("Seleccionar todo (Tipo)", value=True, key="all_type")
        sel_types = cols[1].multiselect("Tipo", types_all, default=types_all if sel_all_t else [])
        df_zt = df_z[df_z["type"].isin(sel_types)] if sel_types else df_z

        nuevos_all = sorted([n for n in df_zt["nuevo"].dropna().unique() if n])
        sel_all_n = cols[2].checkbox("Seleccionar todo (Nuevo)", value=True, key="all_nuevo")
        sel_nuevo = cols[2].multiselect("Nuevo", nuevos_all, default=nuevos_all if sel_all_n else [])
        df_ztn = df_zt[df_zt["nuevo"].isin(sel_nuevo)] if sel_nuevo else df_zt

        clients_all = sorted([c for c in df_ztn["cliente"].dropna().unique() if c])
        sel_all_c = cols[3].checkbox("Seleccionar todo (Clientes)", value=False, key="all_clients")
        sel_clients = cols[3].multiselect("Clientes", clients_all, default=clients_all if sel_all_c else [])

        out = df_ztn[df_ztn["cliente"].isin(sel_clients)] if sel_clients else df_ztn

        st.markdown("---")
        use_top = st.checkbox("Filtrar por Top-N MRR (último mes)", value=False)
        if use_top:
            n_top = st.slider("Top N", 3, 50, 10, step=1)
            mrr_df = out[out["metric"]=="MRR"]
            if not mrr_df.empty:
                last_period = mrr_df["period"].max()
                last = mrr_df[mrr_df["period"]==last_period].groupby("cliente", as_index=False)["value"].sum()
                top_clients = last.sort_values("value", ascending=False).head(n_top)["cliente"].tolist()
                out = out[out["cliente"].isin(top_clients)]

        return out, sel_clients
