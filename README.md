# Revenue & Health Forecaster (Streamlit)

Carga un CSV o un Excel con hojas `MRR`, `CIHS` y `Transactions` para:
- Explorar KPI de MRR (MoM, YoY, Top clientes)
- Filtrar por `zona`, `type`, `nuevo`, Top-N MRR
- Pronosticar hasta 12 meses con Prophet (agregado o por cliente)
- Ver alertas de churn (heurísticas)
- Ver correlaciones entre métricas (si hay al menos dos)

## Estructura esperada

### CSV (demo)
- Columnas meta (ej.: `type client`, `Nuevo`, `Zona`, `Cliente`, `Razon Social`)
- Columnas mensuales: `Jan-19`, `Feb-19`, ... (valores numéricos)

### Excel
- Hojas: `MRR`, `CIHS` y `Transactions` (nombres no sensibles a mayúsculas/minúsculas)
- Cada hoja con la misma estructura de columnas meta + columnas mensuales.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Cloud

1. Crea un repositorio en GitHub y sube `app.py`, `requirements.txt` y `README.md`.
2. Ve a https://share.streamlit.io, conecta tu cuenta de GitHub y selecciona el repo.
3. Asigna `app.py` como archivo principal.
4. Sube tus archivos de datos desde la UI de la app (no los incluyas públicos si son sensibles).
