# -*- coding: utf-8 -*-
"""
Verifica dell'ipotesi: la minore produzione metaforica spontanea nel gruppo Voices
e' spiegata piu' dalla sintomatologia generale (disorganizzazione/ritiro/appiattimento,
sottoscala PANSS General) che da un effetto specifico del tipo di sintomo positivo
(item P1 = Delirio vs item P3 = Comportamento allucinatorio).

Correla il NUMERO di metafore prodotte da ciascun paziente (0 per chi non ne ha
prodotta nessuna: la domanda riguarda la produzione spontanea, quindi i non-produttori
contano) con: item P1 (delirio), item P3 (allucinazioni), e le sottoscale classiche
POS/NEG/GEN come riferimento. Spearman, per l'intero campione e separatamente per
gruppo Delusion/Voices.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PATHS = {
    "metaphors_file": "data/Schizo_Metaphors.xlsx",
    "database_voices": "data/Database_Voices.xlsx",
    "database_delusions": "data/Database_Delusions_14042026.xlsx",
}
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
ALPHA = 0.05


def load_panss_items(path, id_col, classic_source):
    """Estrae item P1 (delirio), P3 (allucinazioni) e le sottoscale classiche
    POS/NEG/GEN (stessa logica gia' usata in NLI_Analysis.py::load_panss_file)."""
    df = pd.read_excel(path)
    df = df.rename(columns={id_col: "File"})
    df["File"] = df["File"].astype(str).str.strip()
    out = df[["File"]].copy()

    if classic_source == "voices":
        out["PANSS_P1_delirio"] = pd.to_numeric(df.get("p1-deliri"), errors="coerce")
        out["PANSS_P3_allucinazioni"] = pd.to_numeric(df.get("p3-comportamento allucinatorio"), errors="coerce")
        out["PANSS_POS"] = pd.to_numeric(df.get("PANSS POS"), errors="coerce")
        out["PANSS_NEG"] = pd.to_numeric(df.get("PANSS NEG"), errors="coerce")
        tot = pd.to_numeric(df.get("PANSS TOT"), errors="coerce")
        out["PANSS_GEN"] = tot - out["PANSS_POS"] - out["PANSS_NEG"]
    else:
        out["PANSS_P1_delirio"] = pd.to_numeric(df.get("T0_PANSS_p1"), errors="coerce")
        out["PANSS_P3_allucinazioni"] = pd.to_numeric(df.get("T0_PANSS_p3"), errors="coerce")
        p_cols = [f"T0_PANSS_p{i}" for i in range(1, 8)]
        n_cols = [f"T0_PANSS_n{i}" for i in range(1, 8)]
        g_cols = [f"T0_PANSS_g{i}" for i in range(1, 17)]
        for c in p_cols + n_cols + g_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        out["PANSS_POS"] = df[p_cols].sum(axis=1, skipna=False) if all(c in df.columns for c in p_cols) else np.nan
        out["PANSS_NEG"] = df[n_cols].sum(axis=1, skipna=False) if all(c in df.columns for c in n_cols) else np.nan
        out["PANSS_GEN"] = df[g_cols].sum(axis=1, skipna=False) if all(c in df.columns for c in g_cols) else np.nan
    return out


panss_voices = load_panss_items(PATHS["database_voices"], "ID", "voices")
panss_voices["Group"] = "Voices"
panss_delusions = load_panss_items(PATHS["database_delusions"], "ID", "delusions")
panss_delusions["Group"] = "Delusion"
panss_items = pd.concat([panss_delusions, panss_voices], ignore_index=True)
print("Popolazione totale registrata (PANSS):", panss_items.shape[0])

# NUMERO DI METAFORE PER PAZIENTE (0 per chi non ne ha prodotta nessuna)
df_meta = pd.read_excel(PATHS["metaphors_file"]).dropna(subset=["Metaphor"]).reset_index(drop=True)
df_meta["File"] = df_meta["ID"].astype(str).str.strip()
n_metaphors = df_meta.groupby("File").size()
panss_items["N_Metaphors"] = panss_items["File"].map(n_metaphors).fillna(0).astype(int)

panss_items.to_csv(f"{OUTPUT_DIR}/N_metaphors_vs_PANSS_items.csv", index=False)
print(f"Pazienti senza nessuna metafora: {(panss_items['N_Metaphors'] == 0).sum()}/{len(panss_items)}")

vars_to_test = ["PANSS_P1_delirio", "PANSS_P3_allucinazioni", "PANSS_POS", "PANSS_NEG", "PANSS_GEN"]


def corr_table(sub, label):
    rows = []
    for v in vars_to_test:
        pair = sub[["N_Metaphors", v]].dropna()
        if len(pair) < 3:
            rows.append({"Group": label, "Variable": v, "r": np.nan, "p_value": np.nan, "N": len(pair), "Significant": False})
            continue
        r, p = spearmanr(pair["N_Metaphors"], pair[v])
        rows.append({"Group": label, "Variable": v, "r": round(r, 3), "p_value": p, "N": len(pair), "Significant": p < ALPHA})
    return pd.DataFrame(rows)


results = pd.concat([
    corr_table(panss_items, "ALL"),
    corr_table(panss_items[panss_items["Group"] == "Delusion"], "Delusion"),
    corr_table(panss_items[panss_items["Group"] == "Voices"], "Voices"),
], ignore_index=True)

results.to_csv(f"{OUTPUT_DIR}/correlation_Nmetaphors_vs_PANSS_specific_items.csv", index=False)
print()
print(results.to_string(index=False))
