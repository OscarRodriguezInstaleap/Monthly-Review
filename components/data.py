# components/data.py (v5)

import pandas as pd, numpy as np, re
from datetime import datetime
from typing import Dict, List, Tuple

# Mapeo de meses ES -> EN (incluye abreviaturas comunes)
SPANISH_MONTH_MAP = {
    "enero": "january", "febrero": "february", "marzo": "march", "abril": "april",
    "mayo": "may", "junio": "june", "julio": "july", "agosto": "august",
    "setiembre": "september", "septiembre": "september",
    "octubre": "october", "noviembre": "november", "diciembre": "december",
    # abreviaturas
    "ene": "jan", "abr": "apr", "ago": "aug", "sept": "sep", "dic": "dec"
}

def _es_to_en_months(s: str) -> str:
    """Reemplaza nombres de meses en español por inglés para facilitar el parseo."""
    t = str(s).strip().lower()
    for es, en in SPANISH_MONTH_MAP.items():
        t = re.sub(rf"\b{re.escape(es)}\b", en, t, flags=re.IGNORECASE)
    return t

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas y mapea aliases a los esperados por la app."""
    df = df.copy()
    # Pasa a minúsculas salvo que sea un Timestamp (se usa luego)
    lower_map = {c: (str(c).strip().lower() if not isinstance(c, (pd.Timestamp, datetime)) else c) for c in df.columns}
    df.rename(columns=lower_map, inplace=True)

    # Mapear aliases -> nombres canónicos
    mapping = {
        "type client": "type", "tipo cliente": "type", "tipo de cliente": "type",
        "nuevo": "nuevo",
        "zona": "zona",
        "cliente": "cliente", "client": "cliente",
        "razon social": "razon_social", "razón social": "razon_social", "razon_social": "razon_social",
        "pais": "pais", "país": "pais", "country": "pais"
    }
    for k, v in mapping.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)
    return df

def detect_meta_and_time(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Separa columnas meta (dimensiones) vs columnas de tiempo (meses/fechas)."""
    def is_time_header(col):
        if isinstance(col, (pd.Timestamp, datetime)):
            return True
        s = str(col).strip()
        s_norm = _es_to_en_months(s)

        # Patrones tipo "Aug-2025", "August 2025", "2025-08", "08/2025", etc.
        month_regex = re.compile(
            r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-_/ ]?\d{2,4}$",
            re.IGNORECASE
        )
        if month_regex.match(s_norm):
            return True

        for fmt in ("%b-%y", "%b-%Y", "%B-%y", "%B-%Y", "%Y-%m", "%Y/%m", "%m-%Y", "%m/%Y"):
            try:
                pd.to_datetime(s_norm, format=fmt)
                return True
            except Exception:
                pass

        # Último intento flexible (evita tomar columnas meta conocidas)
        try:
            pd.to_datetime(s_norm, errors="raise")
            if s.lower() not in {"type", "zona", "nuevo", "cliente", "razon social", "razón social", "razon_social", "pais", "país", "country"}:
                return True
        except Exception:
            return False

        return False

    meta_cols, ts_cols = [], []
    for c in df.columns:
        (ts_cols if is_time_header(c) else meta_cols).append(c)
    return meta_cols, ts_cols

def to_long(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Convierte una hoja wide (columnas por mes) a long (date/value) con 'metric'."""
    df = standardize_columns(df)
    meta_cols, ts_cols = detect_meta_and_time(df)

    if not ts_cols:
        return pd.DataFrame(columns=meta_cols + ["date", "period", "value", "metric"])

    long_df = df.melt(
        id_vars=meta_cols,
        value_vars=ts_cols,
        var_name="month",
        value_name="value_raw"
    )

    # Limpieza de valores
    long_df["value"] = (
        long_df["value_raw"].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace({"": "None", "nan": "None", "None": "None", "NULL": "None", "null": "None"})
    ).replace("None", pd.NA)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce").fillna(0.0)

    # Parseo robusto de fechas
    def _parse_date(x):
        if isinstance(x, (pd.Timestamp, datetime)):
            return pd.to_datetime(x)
        s = _es_to_en_months(str(x))
        for fmt in ("%b-%y", "%b-%Y", "%B-%y", "%B-%Y", "%Y-%m", "%Y/%m", "%m-%Y", "%m/%Y"):
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                pass
        try:
            return pd.to_datetime(s)
        except Exception:
            return pd.NaT

    long_df["date"] = long_df["month"].map(_parse_date)
    long_df["period"] = long_df["date"].dt.to_period("M")
    long_df["metric"] = metric_name
    return long_df

def build_unified_long(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une las hojas MRR / CIHS / Transactions (y otras) en un dataframe unificado long."""
    parts = []
    for key, df in tables.items():
        key_norm = key.strip().lower()
        if key_norm in ("mrr", "revenue"):
            parts.append(to_long(df, "MRR"))
        elif key_norm in ("cihs", "health"):
            parts.append(to_long(df, "CIHS"))
        elif key_norm.startswith("trans") or key_norm in ("tx", "orders", "transactions"):
            parts.append(to_long(df, "Transactions"))
        else:
            # Por si hay otras hojas adicionales
            parts.append(to_long(df, key))

    if not parts:
        return pd.DataFrame(columns=["type", "zona", "nuevo", "cliente", "razon_social", "pais", "date", "period", "value", "metric"])

    out = pd.concat(parts, ignore_index=True)
    out = standardize_columns(out)

    # Asegura columnas meta
    for col in ["type", "zona", "nuevo", "cliente", "razon_social", "pais"]:
        if col not in out.columns:
            out[col] = None
        out[col] = out[col].astype(str).str.strip().replace({"None": None, "nan": None, "": None})

    # Completa metadatos "último no nulo" por cliente
    meta_cols = ["type", "zona", "nuevo", "razon_social", "pais"]
    def last_non_null(s):
        s2 = s.dropna()
        return s2.iloc[-1] if len(s2) else None

    tmp = out.sort_values("date")
    latest_meta = tmp.groupby("cliente", as_index=False)[meta_cols].agg(last_non_null)
    out = out.merge(latest_meta, on="cliente", suffixes=("", "_bf"), how="left")
    for col in meta_cols:
        out[col] = out[col].fillna(out[f"{col}_bf"])
        out.drop(columns=[f"{col}_bf"], inplace=True)

    # Normaliza 'period' a str para orden/joins consistentes
    out["period"] = pd.PeriodIndex(out["period"]).astype(str)
    return out
