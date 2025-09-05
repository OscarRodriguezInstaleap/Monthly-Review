import pandas as pd
import numpy as np

def churn_alerts(df: pd.DataFrame) -> pd.DataFrame:
    m = df[df["metric"]=="MRR"].copy()
    tx = df[df["metric"]=="Transactions"].copy()
    c = df[df["metric"]=="CIHS"].copy()

    alerts = []
    for client, g in m.groupby("cliente"):
        g = g.sort_values("date")
        if g.empty: continue
        last = g.iloc[-1]["value"]
        prev = g.iloc[-2]["value"] if len(g)>=2 else np.nan

        if len(g) >= 6:
            last3 = g.iloc[-3:]["value"].sum()
            prev3 = g.iloc[-6:-3]["value"].sum()
        else:
            last3 = g.iloc[-min(3,len(g)):]["value"].sum()
            prev3 = g.iloc[:max(0,len(g)-min(3,len(g)))].tail(min(3,len(g)))["value"].sum()

        risk = None; rule = None; signals = []
        if (not np.isnan(prev)) and prev > 0 and last == 0:
            risk, rule = "HARD CHURN", "Último mes = 0 y penúltimo > 0"
        else:
            drop = (prev3 - last3) / prev3 * 100 if prev3 > 0 else 0.0
            if drop > 30: risk, rule = "ALTO", f"MRR: caída {drop:.1f}% en 3m"
            elif drop > 15: risk, rule = "MEDIO", f"MRR: caída {drop:.1f}% en 3m"

        if not tx[tx["cliente"]==client].empty:
            txg = tx[tx["cliente"]==client].sort_values("date")
            if len(txg)>=6:
                tx_last3 = txg.iloc[-3:]["value"].sum()
                tx_prev3 = txg.iloc[-6:-3]["value"].sum()
                tx_drop = (tx_prev3 - tx_last3)/tx_prev3*100 if tx_prev3>0 else 0.0
                if tx_drop>25: signals.append(f"TX caída {tx_drop:.0f}%")

        if not c[c["cliente"]==client].empty:
            cg = c[c["cliente"]==client].sort_values("date")
            if len(cg)>=3:
                c_last3 = cg.iloc[-3:]["value"].mean()
                c_prev3 = cg.iloc[-6:-3]["value"].mean() if len(cg)>=6 else cg.iloc[:max(0,len(cg)-3)].tail(3)["value"].mean()
                c_drop = c_prev3 - c_last3
                if c_drop>5: signals.append(f"CIHS baja {c_drop:.1f} pts")

        if risk or signals:
            alerts.append({
                "cliente": client,
                "zona": g["zona"].iloc[-1] if "zona" in g.columns else None,
                "type": g["type"].iloc[-1] if "type" in g.columns else None,
                "pais": g["pais"].iloc[-1] if "pais" in g.columns else None,
                "ultimo_mrr": last,
                "mrr_penultimo": prev if not np.isnan(prev) else None,
                "riesgo": risk if risk else "SEÑALES",
                "evidencias": "; ".join(signals) if signals else rule,
            })
    return pd.DataFrame(alerts)
