import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Tuple

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower_map = {c: (str(c).strip().lower() if not isinstance(c, (pd.Timestamp, datetime)) else c) for c in df.columns}
    df.rename(columns=lower_map, inplace=True)
    mapping = {
        "type client": "type",
        "tipo cliente": "type",
        "tipo de cliente": "type",
        "nuevo": "nuevo",
        "zona": "zona",
        "cliente": "cliente",
        "client": "cliente",
        "razon social": "razon_social",
        "razón social": "razon_social",
        "razon_social": "razon_social",
        "pais": "pais",
        "país": "pais",
        "country": "pais"
    }
    for k,v in mapping.items():
        if k in df.columns:
            df.rename(columns={k:v}, inplace=True)
    return df

def detect_meta_and_time(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    def is_time_header(col):
        if isinstance(col, (pd.Timestamp, datetime)):
            return True
        s = str(col).strip()
        month_regex = re.compile(
            r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
            r"ene|feb|mar|abr|may|jun|jul|ago|sept|sep|oct|nov|dic)"
            r"[a-z]*[-_/ ]?\d{2,4}$", re.IGNORECASE
        )
        if month_regex.match(s):
            return True
        for fmt in ("%b-%y", "%b-%Y", "%B-%y", "%B-%Y"):
            try:
                pd.to_datetime(s, format=fmt)
                return True
            except Exception:
                pass
        try:
            dt = pd.to_datetime(s, errors="raise")
            if s.lower() not in {"type", "zona", "nuevo", "cliente", "razon social", "razón social", "razon_social"}:
                return True
        except Exception:
            return False
        return False
    meta_cols, ts_cols = [], []
    for c in df.columns:
        if is_time_header(c):
            ts_cols.append(c)
        else:
            meta_cols.append(c)
    return meta_cols, ts_cols

def to_long(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    df = standardize_columns(df)
    meta_cols, ts_cols = detect_meta_and_time(df)
    if not ts_cols:
        return pd.DataFrame(columns=meta_cols + ["date","value","metric"])
    long_df = df.melt(id_vars=meta_cols, value_vars=ts_cols, var_name="month", value_name="value_raw")
    long_df["value"] = (
        long_df["value_raw"].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
            .replace({"": None, "nan": None, "None": None, "NULL": None, "null": None})
    )
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce").fillna(0.0)
    def _parse_date(x):
        if isinstance(x, (pd.Timestamp, datetime)):
            return pd.to_datetime(x)
        x = str(x)
        for fmt in ("%b-%y", "%b-%Y", "%B-%y", "%B-%Y"):
            try:
                return pd.to_datetime(x, format=fmt)
            except Exception:
                pass
        try:
            return pd.to_datetime(x)
        except Exception:
            return pd.NaT
    long_df["date"] = long_df["month"].map(_parse_date)
    long_df["period"] = long_df["date"].dt.to_period("M")
    long_df["metric"] = metric_name
    return long_df

def build_unified_long(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for key, df in tables.items():
        key_norm = key.strip().lower()
        if key_norm in ("mrr","revenue"):
            parts.append(to_long(df, "MRR"))
        elif key_norm in ("cihs","health"):
            parts.append(to_long(df, "CIHS"))
        elif key_norm.startswith("trans") or key_norm in ("tx","orders","transactions"):
            parts.append(to_long(df, "Transactions"))
        else:
            parts.append(to_long(df, key))
    if parts:
        out = pd.concat(parts, ignore_index=True)
        out = standardize_columns(out)
        for col in ["type","zona","nuevo","cliente","razon_social","pais"]:
            if col not in out.columns:
                out[col] = None
        for col in ["type","zona","nuevo","cliente","razon_social","pais"]:
            out[col] = out[col].astype(str).str.strip().replace({"None": None, "nan": None, "": None})

        meta_cols = ["type","zona","nuevo","razon_social","pais"]
        def last_non_null(s):
            s2 = s.dropna()
            return s2.iloc[-1] if len(s2) else None
        tmp = out.sort_values("date")
        latest_meta = tmp.groupby("cliente", as_index=False)[meta_cols].agg(last_non_null)
        out = out.merge(latest_meta, on="cliente", suffixes=("", "_bf"), how="left")
        for col in meta_cols:
            out[col] = out[col].fillna(out[f"{col}_bf"])
            out.drop(columns=[f"{col}_bf"], inplace=True)

        out["period"] = pd.PeriodIndex(out["period"]).astype(str)
        return out
    return pd.DataFrame(columns=["type","zona","nuevo","cliente","razon_social","pais","date","period","value","metric"])
