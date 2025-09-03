# Revenue & Health Forecaster (Streamlit)

## Novedades
- Gráficas también para **Transacciones** (con opción de pronóstico).
- Filtro por **grupos de clientes** (multiselect).
- KPIs extendidos: **ARR**, **NRR YoY (cohorte)**, **Top crecimiento/decrecimiento** (YoY por cliente).
- **Gráfico YoY por cohorte**: compara, para el mes seleccionado, el MRR de los clientes activos vs los mismos clientes un año atrás.

## Cómo usar
1. Sube un **Excel** con las hojas `MRR`, `CIHS`, `Transactions` (o un CSV equivalente para MRR).
2. Aplica filtros en la barra lateral (Zona, Tipo, Nuevo, Clientes, Top-N).
3. Revisa KPIs, gráficos de **MRR** y **Transacciones**, el **YoY por cohorte**, **alertas de churn** y **correlaciones**.
4. Descarga tabulados y alertas en CSV cuando lo necesites.

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```
