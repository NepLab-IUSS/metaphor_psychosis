# -*- coding: utf-8 -*-
"""
# COSE CHE MANCANO:
1. "threhsold in partenza" su item IPASE: dove lo vogliamo mettere?
2. correlazione con APACS/PMM z-scored
3. maxi legenda con item titoli e testo (non so dove metterla, però la vedi nella prima cella)

# Analisi Metafore — NLI + Sottoscale PANSS (Van der Gaag)

File di input per le metafore (`Schizo_Metaphors.xlsx`), contenente sia le metafore
Delusion (ID hash) sia Voices (ID "pt").

1. Setup modello NLI e scala IPASE
2. Classificazione NLI sul file unico di metafore -> df_all
3. Merge con sottoscale PANSS (gia' calcolate) + vehicle/TARGET/SOURCE
4. Aggregazione a livello paziente
5. Macro-categorie IPASE
6. Barplot (medie, nessuno split High/Low)
7. Correlazioni Spearman: item PANSS, macro-categorie PANSS
8. Source Domains x IPASE
9. Tabelle descrittive: N pazienti, N metafore, PANSS medio per sottoscala

## 1. Setup: import, config, modello NLI, scala IPASE
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from scipy.stats import spearmanr

PATHS = {
    "metaphors_file": "data/Schizo_Metaphors.xlsx",
    "database_voices": "data/Database_Voices.xlsx",
    "database_delusions": "data/Database_Delusions_14042026.xlsx",
}

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA = 0.05
EXTRA_COLS = ["vehicle1", "vehicle2", "vehicle3", "TARGET", "SOURCE1", "SOURCE2"]

panss_cols_t0 = ["T0_POS_PANSS", "T0_NEG_PANSS", "T0_DIS_PANSS", "T0_EXC_PANSS", "T0_EMO_PANSS", "T0_TOT_PANSS"]
panss_cols_t1 = ["T1_TOT_PANSS", "T1_POS_PANSS", "T1_NEG_PANSS", "T1_DIS_PANSS", "T1_EXC_PANSS", "T1_EMO_PANSS"]
panss_cols_all = panss_cols_t0 + panss_cols_t1
panss_scale_names = ["POS_PANSS", "NEG_PANSS", "DIS_PANSS", "EXC_PANSS", "EMO_PANSS", "TOT_PANSS"]  # Van der Gaag (5 fattori + Totale)

# PANSS classico (3 fattori: Positive/Negative/General + Totale)
panss_cols_classic_t0 = ["T0_PANSS_POS", "T0_PANSS_NEG", "T0_PANSS_GEN", "T0_PANSS_TOT"]
panss_scale_names_classic = ["PANSS_POS", "PANSS_NEG", "PANSS_GEN", "PANSS_TOT"]

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# SCALA IPASE — 57 item (titolo: testo)
items = [
    {"id": 1, "titolo": "Pensieri generati da altri", "testo": "Ho come la sensazione che i miei pensieri siano generati da qualcun altro."},
    {"id": 2, "titolo": "Vita attuale scollegata dal futuro", "testo": "Ho come la sensazione che la mia vita attuale non sia collegata alla mia vita futura."},
    {"id": 3, "titolo": "Guardarsi dall'esterno", "testo": "A volte, ho come la sensazione di potermi guardare dall'esterno."},
    {"id": 4, "titolo": "Perdita di identità", "testo": "Mi sento come se non avessi più un'identità."},
    {"id": 5, "titolo": "Guardarsi da fuori dal corpo", "testo": "Ho avuto la sensazione di guardarmi da fuori del corpo."},
    {"id": 6, "titolo": "Dubbio tra provare e immaginare", "testo": "È difficile per me dire se sto provando davvero qualcosa o lo sto soltanto immaginando."},
    {"id": 7, "titolo": "Vera identità scomparsa", "testo": "Ho come la sensazione che la mia vera identità sia scomparsa."},
    {"id": 8, "titolo": "Età percepita alterata", "testo": "Ho avuto la sensazione di essere più vecchio/a o più giovane di come sono veramente."},
    {"id": 9, "titolo": "Dubbio sulla propria esistenza", "testo": "Mi chiedo se esisto veramente oppure no."},
    {"id": 10, "titolo": "Perdita di contatto con se stessi", "testo": "È come se avessi perso il contatto con me stesso/a."},
    {"id": 11, "titolo": "Corpo percepito come cambiato", "testo": "Ho come la sensazione che il mio corpo sia cambiato."},
    {"id": 12, "titolo": "Parole lette da qualcun altro", "testo": "Quando leggo, ho come la sensazione che le parole siano lette da qualcun altro."},
    {"id": 13, "titolo": "Arti percepiti come non propri", "testo": "A volte ho come la sensazione che le braccia, le gambe o altre parti del mio corpo non siano veramente mie."},
    {"id": 14, "titolo": "Dubbio tra dire e pensare", "testo": "Mi capita di avere il dubbio di aver detto veramente qualcosa o di averla solo pensata."},
    {"id": 15, "titolo": "Distanza da se stessi", "testo": "Mi sento distante da me stesso."},
    {"id": 16, "titolo": "Perdita di controllo del corpo", "testo": "A volte ho come la sensazione di non riuscire a tenere sotto controllo le diverse parti del mio corpo."},
    {"id": 17, "titolo": "Mancanza di idee proprie", "testo": "Spesso ho come la sensazione di dover dare ragione agli altri perché non ho idee mie."},
    {"id": 18, "titolo": "Perdita totale di sé", "testo": "È come se avessi perso totalmente me stesso/a."},
    {"id": 19, "titolo": "Controllo allo specchio", "testo": "Spesso mi guardo allo specchio per vedere se sono cambiato/a."},
    {"id": 20, "titolo": "Pensieri non propri", "testo": "È come se i miei pensieri non mi appartenessero."},
    {"id": 21, "titolo": "Ricerca di identità tramite oggetti", "testo": "Cerco di capire chi sono guardando cose come foto, appunti, diari."},
    {"id": 22, "titolo": "Alterazione della percezione del tempo", "testo": "Sembra che il tempo sia più veloce o più lento del solito."},
    {"id": 23, "titolo": "Vivere in un altro mondo", "testo": "Sto vivendo in un altro mondo."},
    {"id": 24, "titolo": "Sensazioni elettriche nel corpo", "testo": "Ho sensazioni di tipo elettrico nel mio corpo."},
    {"id": 25, "titolo": "Non far parte del mondo", "testo": "Mi sento come se non facessi più parte di questo mondo."},
    {"id": 26, "titolo": "Paura di perdersi", "testo": "Ho paura di perdere me stesso/a."},
    {"id": 27, "titolo": "Sensazioni termiche senza causa", "testo": "Ho avuto sensazioni di caldo o freddo nel mio corpo senza che ci fosse un reale cambiamento della temperatura intorno a me."},
    {"id": 28, "titolo": "Pensieri come scritti", "testo": "Quando penso, è come se i miei pensieri venissero messi per scritto."},
    {"id": 29, "titolo": "Estraneità a se stessi", "testo": "Mi sento di essere un estraneo a me stesso/a."},
    {"id": 30, "titolo": "Dolore ai rumori", "testo": "Ho provato dolore nel sentire alcuni rumori."},
    {"id": 31, "titolo": "Tempo che accelera, rallenta o si ferma", "testo": "Ho come la sensazione che il tempo si precipiti in avanti, rallenti o si fermi."},
    {"id": 32, "titolo": "Perdita di connessione col mondo", "testo": "Mi sento come se non avessi più una connessione con il mondo."},
    {"id": 33, "titolo": "Gambe che si piegano da ferme", "testo": "Ho avuto la sensazione che le mie gambe si piegassero o che il mio corpo dondolasse quando in realtà stavo fermo."},
    {"id": 34, "titolo": "Sentirsi come un fantasma", "testo": "A volte mi sento come un fantasma."},
    {"id": 35, "titolo": "Scarsa presenza nel mondo", "testo": "Sento di non essere molto presente in questo mondo."},
    {"id": 36, "titolo": "Incapacità di controllare il corpo", "testo": "Ho avuto momenti in cui non ero capace di controllare il mio corpo."},
    {"id": 37, "titolo": "Pensieri percepiti come rumorosi", "testo": "Quando penso, i miei pensieri mi sembrano così rumorosi che mi chiedo se gli altri possano sentirli."},
    {"id": 38, "titolo": "Non essere più la stessa persona", "testo": "Sento di non essere più la stessa persona che sono sempre stato/a."},
    {"id": 39, "titolo": "Impossibilità di muoversi", "testo": "Ho avuto momenti in cui cercavo di muovermi ma non ci riuscivo."},
    {"id": 40, "titolo": "Vuoto interiore", "testo": "Sento di avere un vuoto interiore."},
    {"id": 41, "titolo": "Amnesia di azioni compiute", "testo": "A volte non riesco a ricordare di aver fatto cose che so di aver fatto."},
    {"id": 42, "titolo": "Pensare a se stessi come un'altra persona", "testo": "Quando penso a me stesso/a, ho come la sensazione di pensare ad un'altra persona."},
    {"id": 43, "titolo": "Debolezza improvvisa di un arto", "testo": "Ho avuto di colpo un senso di debolezza ad un braccio, o ad una gamba o ad un'altra parte del mio corpo."},
    {"id": 44, "titolo": "Disconnessione da sé e dai propri pensieri", "testo": "Ho come la sensazione che non ci sia connessione tra me stesso/a e ciò che sto pensando."},
    {"id": 45, "titolo": "Evitamento per mancanza di opinioni", "testo": "Evito le discussioni perché non riesco ad avere un'opinione personale sulle cose."},
    {"id": 46, "titolo": "Svanire dall'esistenza", "testo": "È come se stia svanendo dall'esistenza."},
    {"id": 47, "titolo": "Movimento percepito per imitazione", "testo": "Quando vedo muoversi qualcuno, ho come la sensazione di muovermi anch'io, anche se sono fermo."},
    {"id": 48, "titolo": "Assorbimento nell'introspezione", "testo": "Sono talmente preso dal guardarmi dentro che ho difficoltà a seguire ciò che accade intorno a me."},
    {"id": 49, "titolo": "Pensieri ripetuti o echeggianti", "testo": "È come se i miei pensieri siano ripetuti o che riecheggino fuori dalla testa."},
    {"id": 50, "titolo": "Sensazione di non essere l'autore delle proprie azioni", "testo": "Quando faccio qualcosa, ho come la sensazione di non essere proprio io a farla."},
    {"id": 51, "titolo": "Osservatore passivo del mondo", "testo": "Ho come la sensazione di essere un osservatore passivo del mondo."},
    {"id": 52, "titolo": "Disallineamento tra espressione e vissuto interno", "testo": "Spesso le mie espressioni del viso, i miei discorsi, il mio modo di fare non sono in linea con quello che penso o con quello che provo veramente."},
    {"id": 53, "titolo": "Sensazione di stare cambiando", "testo": "Ho la strana sensazione di stare in qualche modo cambiando."},
    {"id": 54, "titolo": "Scomparsa della barriera tra sé e il mondo", "testo": "È come se la barriera tra me ed il mondo sia scomparsa."},
    {"id": 55, "titolo": "Appiattimento emotivo", "testo": "Non provo più sensazioni forti come ero solito/a provare."},
    {"id": 56, "titolo": "Vedere i propri pensieri", "testo": "Quando penso, posso vedere i miei pensieri, uno davanti all'altro."},
    {"id": 57, "titolo": "Alterazione del significato del mondo", "testo": "Il significato e l'importanza del mio mondo sembrano cambiati."},
]
items = {it["titolo"]: it["testo"] for it in items}
item_keys = list(items.keys())
id_to_title = {i + 1: titolo for i, titolo in enumerate(item_keys)}
hyp_list = list(items.values())
print(f"{len(item_keys)} item IPASE caricati")

"""## 2. Classificazione NLI"""

def classify_batch(premises, sub_batch_size=32):
    hyps_per_row, prem_per_row = [], []
    for p in premises:
        hyps_per_row.extend(hyp_list)
        prem_per_row.extend([p] * len(hyp_list))

    all_probs = []
    for start in range(0, len(prem_per_row), sub_batch_size):
        end = start + sub_batch_size
        inputs = tokenizer(
            prem_per_row[start:end], hyps_per_row[start:end],
            return_tensors="pt", padding=True, truncation=True,
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
        all_probs.append(torch.softmax(logits, dim=-1).cpu())
        del inputs, logits
        torch.cuda.empty_cache()

    probs = torch.cat(all_probs, dim=0)
    n_items = len(item_keys)
    results = []
    for i in range(len(premises)):
        start, end = i * n_items, (i + 1) * n_items
        results.append({k: round(p[0].item() * 100, 2) for k, p in zip(item_keys, probs[start:end])})
    return results


def run_nli_classification(df, text_col, batch_size=4):
    #Classifica ogni riga di df[text_col] sui 57 item IPASE.
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    orig_cols = list(df.columns)
    output_rows, buf_text, buf_meta = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"NLI su {text_col}"):
        buf_text.append(str(row[text_col]))
        buf_meta.append(row)
        if len(buf_text) == batch_size:
            for meta, res in zip(buf_meta, classify_batch(buf_text)):
                output_rows.append({**meta.to_dict(), **res})
            buf_text, buf_meta = [], []
            torch.cuda.empty_cache()

    if buf_text:
        for meta, res in zip(buf_meta, classify_batch(buf_text)):
            output_rows.append({**meta.to_dict(), **res})
        torch.cuda.empty_cache()

    out = pd.DataFrame(output_rows)[orig_cols + item_keys]
    out[item_keys] = out[item_keys] / 100
    return out


def get_item_cols(df):
    return [c for c in item_keys if c in df.columns]

"""## 3. Classificazione `df_all`

`Schizo_Metaphors.xlsx` contiene tutte le metafore (colonne: `ID, Metaphor, vehicle1, vehicle2,
vehicle3, TARGET, SOURCE1, SOURCE2, notes`). Il `Group` (Delusion vs Voices) viene dedotto dal
formato dell'ID: se finisce in `pt` è Voices, altrimenti è Delusion.
"""

df_meta = pd.read_excel(PATHS["metaphors_file"])
df_meta = df_meta.dropna(subset=["Metaphor"]).reset_index(drop=True)
df_meta["ID"] = df_meta["ID"].astype(str).str.strip()

df_all = run_nli_classification(df_meta, text_col="Metaphor")
df_all["Metaphor_Text"] = df_all["Metaphor"]
df_all["File"] = df_all["ID"]
df_all["Group"] = np.where(df_all["ID"].str.endswith("pt"), "Voices", "Delusion")
df_all["TimePoint"] = "T0"

for c in EXTRA_COLS:
    if c not in df_all.columns:
        df_all[c] = np.nan

item_cols = get_item_cols(df_all)
print("df_all:", df_all.shape)
df_all.head()

# SOGLIA DI RILEVANZA: la distribuzione degli score NLI per singolo item e' bimodale
# (la maggior parte delle metafore e' irrilevante, score vicino a 0; un piccolo
# sottoinsieme e' fortemente pertinente, score vicino a 1) invece che continua. Per
# questo, ovunque si aggreghi uno score di ITEM (non di macro-categoria, che essendo
# gia' una media su piu' item e' meno bimodale), si usa la PROPORZIONE di metafore con
# score > RELEVANCE_THRESHOLD invece della media grezza continua.
RELEVANCE_THRESHOLD = 0.5
item_cols_rel = [f"{c}__rel" for c in item_cols]
df_all[item_cols_rel] = (df_all[item_cols] > RELEVANCE_THRESHOLD).astype(int)

"""## 4. Merge con sottoscale PANSS (già calcolate) + vehicle/TARGET/SOURCE

"""

def load_panss_file(path, id_col="File", classic_source="delusions"):
    """Carica anche il PANSS classico (Positive/Negative/General): per `classic_source`
    'voices' usa le colonne aggregate gia' presenti nel file (PANSS POS/NEG/TOT, il
    General e' derivato come Tot-Pos-Neg); per 'delusions' somma gli item singoli
    (p1-p7, n1-n7, g1-g16). Se un item e' mancante (es. placeholder non numerico),
    la sottoscala risultante e' NaN invece di un totale parziale fuorviante."""
    df = pd.read_excel(path)
    if id_col != "File":
        df = df.rename(columns={id_col: "File"})
    df["File"] = df["File"].astype(str).str.strip()

    out = df[["File"]].copy()
    for c in panss_cols_all:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan

    if classic_source == "voices":
        out["T0_PANSS_POS"] = pd.to_numeric(df.get("PANSS POS"), errors="coerce")
        out["T0_PANSS_NEG"] = pd.to_numeric(df.get("PANSS NEG"), errors="coerce")
        tot_classic = pd.to_numeric(df.get("PANSS TOT"), errors="coerce")
        out["T0_PANSS_GEN"] = tot_classic - out["T0_PANSS_POS"] - out["T0_PANSS_NEG"]
    else:
        p_cols = [f"T0_PANSS_p{i}" for i in range(1, 8)]
        n_cols = [f"T0_PANSS_n{i}" for i in range(1, 8)]
        g_cols = [f"T0_PANSS_g{i}" for i in range(1, 17)]
        for c in p_cols + n_cols + g_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        out["T0_PANSS_POS"] = df[p_cols].sum(axis=1, skipna=False) if all(c in df.columns for c in p_cols) else np.nan
        out["T0_PANSS_NEG"] = df[n_cols].sum(axis=1, skipna=False) if all(c in df.columns for c in n_cols) else np.nan
        out["T0_PANSS_GEN"] = df[g_cols].sum(axis=1, skipna=False) if all(c in df.columns for c in g_cols) else np.nan
    out["T0_PANSS_TOT"] = out["T0_PANSS_POS"] + out["T0_PANSS_NEG"] + out["T0_PANSS_GEN"]
    return out


panss_voices = load_panss_file(PATHS["database_voices"], id_col="ID", classic_source="voices")
panss_delusions = load_panss_file(PATHS["database_delusions"], id_col="ID", classic_source="delusions")

panss_all = pd.concat([panss_delusions, panss_voices], ignore_index=True)
panss_all.to_csv(f"{OUTPUT_DIR}/PANSS_subscales_ALL.csv", index=False)
print("PANSS caricate:", panss_all.shape)

df_all = df_all.merge(panss_all, on="File", how="left")
df_all.to_csv(f"{OUTPUT_DIR}/Metaphor_ALL_NLI_IPASE.csv", index=False)
df_all.to_excel(f"{OUTPUT_DIR}/Metaphor_ALL_NLI_IPASE.xlsx", index=False)

print("df_all con PANSS e vehicle/TARGET/SOURCE:", df_all.shape)
df_all


print("Colonne Database_Voices:")
print(pd.read_excel(PATHS["database_voices"]).columns.tolist())

print("\nColonne Database_Delusions:")
print(pd.read_excel(PATHS["database_delusions"]).columns.tolist())

"""## 5. Aggregazione a livello paziente (media item per File)"""

# AGGREGAZIONE A LIVELLO PAZIENTE: proporzione di metafore del paziente con score
# item > RELEVANCE_THRESHOLD (non media grezza, vedi nota sulla bimodalita' sopra)
df_patient_avg = (
    df_all.groupby(["File", "Group"])[item_cols_rel]
    .mean()
    .rename(columns=dict(zip(item_cols_rel, item_cols)))
    .reset_index()
    .merge(panss_all[["File"] + panss_cols_t0 + panss_cols_classic_t0], on="File", how="left")
)
df_patient_avg.to_csv(f"{OUTPUT_DIR}/patient_avg_ratings_panss_T0.csv", index=False)
print("Rating medi per paziente:", df_patient_avg.shape)
df_patient_avg

# VERSIONE SENZA SOGLIA (media grezza continua, come prima di introdurre
# RELEVANCE_THRESHOLD): tenuta in parallelo su richiesta, per confrontare i due approcci
df_patient_avg_raw = (
    df_all.groupby(["File", "Group"])[item_cols]
    .mean()
    .reset_index()
    .merge(panss_all[["File"] + panss_cols_t0 + panss_cols_classic_t0], on="File", how="left")
)
df_patient_avg_raw.to_csv(f"{OUTPUT_DIR}/patient_avg_ratings_panss_T0_raw.csv", index=False)
print("Rating medi per paziente (senza soglia):", df_patient_avg_raw.shape)

"""## 6. Macro-categorie IPASE, a livello paziente"""

from matplotlib.patches import Patch

subscales = {
    "Cognition": [1, 12, 20, 28, 37, 49, 56],
    "Self-Awareness and Presence": [2, 4, 7, 10, 15, 18, 21, 23, 26, 29, 32, 35, 38, 40, 42, 44, 46, 48, 50, 53, 55, 57],
    "Consciousness": [6, 14, 22, 31, 41, 52],
    "Somatization": [3, 5, 8, 11, 13, 16, 19, 24, 27, 30, 33, 36, 39, 43, 47, 51, 54],
    "Demarcation/Transitivism": [9, 17, 25, 34, 45],
}
subscale_titles = {name: [id_to_title[i] for i in ids] for name, ids in subscales.items()}
subscale_names = list(subscales.keys())
macro_colors = {"Cognition": "#1b9e77", "Self-Awareness and Presence": "#d95f02", "Consciousness": "#7570b3", "Somatization": "#e7298a", "Demarcation/Transitivism": "#66a61e"}
item_to_macro = {t: name for name, titles in subscale_titles.items() for t in titles}
item_cols = [t for name in subscale_names for t in subscale_titles[name] if t in item_cols]
macro_legend_handles = [Patch(facecolor=macro_colors[name], label=name) for name in subscale_names]

for name, titles in subscale_titles.items():
    titles_present = [t for t in titles if t in df_patient_avg.columns]
    df_patient_avg[name] = df_patient_avg[titles_present].mean(axis=1)

df_patient_avg.to_csv(f"{OUTPUT_DIR}/patient_avg_ratings_panss_T0.csv", index=False)
df_patient_avg[["File", "Group"] + subscale_names].head()

# stesso calcolo sulla versione senza soglia
for name, titles in subscale_titles.items():
    titles_present = [t for t in titles if t in df_patient_avg_raw.columns]
    df_patient_avg_raw[name] = df_patient_avg_raw[titles_present].mean(axis=1)
df_patient_avg_raw.to_csv(f"{OUTPUT_DIR}/patient_avg_ratings_panss_T0_raw.csv", index=False)

"""## 6.1 File dedicato: macro-categorie IPASE per paziente (+ punteggio IPASE medio complessivo)

Oltre alle macro-categorie già aggiunte a `df_patient_avg`, qui creiamo un **file standalone**
con solo `File`, `Group`, le 5 macro-categorie, e un **punteggio IPASE medio complessivo**
(media di tutti i 57 item), così da avere un unico output leggero da condividere/ispezionare
senza portarsi dietro tutte le altre colonne di `df_patient_avg`.
"""

# PUNTEGGIO IPASE MEDIO COMPLESSIVO (media di tutti i 57 item), per paziente
df_patient_avg["IPASE_Mean_Score"] = df_patient_avg[item_cols].mean(axis=1)
df_patient_avg_raw["IPASE_Mean_Score"] = df_patient_avg_raw[item_cols].mean(axis=1)

# FILE STANDALONE: macro-categorie + punteggio medio complessivo, per paziente
macro_export_cols = ["File", "Group"] + subscale_names + ["IPASE_Mean_Score"]
df_macro_categories = df_patient_avg[macro_export_cols].copy()

df_macro_categories.to_csv(f"{OUTPUT_DIR}/IPASE_macrocategories_by_patient.csv", index=False)
df_macro_categories.to_excel(f"{OUTPUT_DIR}/IPASE_macrocategories_by_patient.xlsx", index=False)

print("File macro-categorie salvato:", df_macro_categories.shape)
df_macro_categories.head()

# stessa cosa per la versione senza soglia
df_macro_categories_raw = df_patient_avg_raw[macro_export_cols].copy()
df_macro_categories_raw.to_csv(f"{OUTPUT_DIR}/IPASE_macrocategories_by_patient_raw.csv", index=False)
df_macro_categories_raw.to_excel(f"{OUTPUT_DIR}/IPASE_macrocategories_by_patient_raw.xlsx", index=False)

"""## 6.2 Barplot (medie complessive)"""

def add_macro_brackets(ax, labels, macro_means, x_start):
    """Disegna, a destra delle barre, una parentesi quadra per ogni gruppo contiguo di
    item della stessa macro-categoria, con accanto il mean score della macro-categoria
    (cosi' non serve un secondo grafico solo per le macro-categorie)."""
    groups = []
    for i, lbl in enumerate(labels):
        macro = item_to_macro.get(lbl)
        if groups and groups[-1][0] == macro:
            groups[-1] = (macro, groups[-1][1], i)
        else:
            groups.append((macro, i, i))

    xmax = ax.get_xlim()[1]
    tick = xmax * 0.012
    for macro, i_start, i_end in groups:
        if macro is None or macro not in macro_means:
            continue
        y_top, y_bottom = i_start - 0.42, i_end + 0.42
        color = macro_colors.get(macro, "black")
        ax.plot([x_start, x_start], [y_top, y_bottom], color=color, lw=1.3, clip_on=False)
        ax.plot([x_start, x_start + tick], [y_top, y_top], color=color, lw=1.3, clip_on=False)
        ax.plot([x_start, x_start + tick], [y_bottom, y_bottom], color=color, lw=1.3, clip_on=False)
        ax.text(x_start + tick * 1.8, (y_top + y_bottom) / 2, f"{macro_means[macro]:.2f}",
                va="center", ha="left", fontsize=11, fontweight="bold", color=color)


def make_mean_barplot(means, labels, title, filepath, xlabel="Mean score (0-1)", bar_colors=None, macro_means=None):
    fig, ax = plt.subplots(figsize=(18, max(6, len(labels) * 0.4)))
    y = np.arange(len(labels))
    bars = ax.barh(y, means, color=bar_colors if bar_colors is not None else "#4c72b0")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.2f}",
                va="center", ha="left", fontsize=13)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13)
    for lbl in ax.get_yticklabels():
        lbl.set_color(macro_colors.get(item_to_macro.get(lbl.get_text()), "black"))
    ax.invert_yaxis()
    max_val = max(means) if max(means) > 0 else 1
    ax.set_xlim(0, max_val * (1.42 if macro_means is not None else 1.15))
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_title(title, fontsize=18, wrap=True)
    if macro_means is not None:
        add_macro_brackets(ax, labels, macro_means, max_val * 1.22)
    if bar_colors is not None:
        ax.legend(handles=macro_legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, title="Macro-category")
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.show()
    plt.close(fig)


# macro-categorie a livello di singola metafora: servono PRIMA del barplot item, per
# affiancare alle barre la parentesi con il mean score di macro-categoria
for name, titles in subscale_titles.items():
    titles_present = [t for t in titles if t in df_all.columns]
    df_all[name] = df_all[titles_present].mean(axis=1)

means_items = pd.Series({c: df_all[f"{c}__rel"].mean() for c in item_cols})

# la parentesi di macro-categoria nel barplot mostra la media delle proporzioni dei suoi
# item (coerente con la definizione di macro usata a livello paziente in df_patient_avg),
# non la media del composito continuo per-metafora (quella resta per la heatmap SOURCE)
means_macro_dict = {
    name: means_items[[t for t in titles if t in means_items.index]].mean()
    for name, titles in subscale_titles.items()
}

item_bar_colors = [macro_colors.get(item_to_macro.get(lbl), "#4c72b0") for lbl in item_cols]
make_mean_barplot(
    means_items.values, means_items.index.tolist(),
    title=f"Proporzione di metafore rilevanti per item IPASE (score > {RELEVANCE_THRESHOLD}); mean score di macro-categoria tra parentesi",
    filepath=f"{OUTPUT_DIR}/barplot_items_ALL.png",
    xlabel=f"Proporzione metafore (score > {RELEVANCE_THRESHOLD})",
    bar_colors=item_bar_colors,
    macro_means=means_macro_dict,
)

# VERSIONE SENZA SOGLIA: media grezza continua su tutte le metafore (nessuna binarizzazione)
means_items_raw = df_all[item_cols].mean().reindex(item_cols)
means_macro_dict_raw = {
    name: means_items_raw[[t for t in titles if t in means_items_raw.index]].mean()
    for name, titles in subscale_titles.items()
}
make_mean_barplot(
    means_items_raw.values, means_items_raw.index.tolist(),
    title="Mean score per item IPASE, senza soglia (media grezza su tutte le metafore); mean score di macro-categoria tra parentesi",
    filepath=f"{OUTPUT_DIR}/barplot_items_ALL_raw.png",
    xlabel="Mean score (0-1)",
    bar_colors=item_bar_colors,
    macro_means=means_macro_dict_raw,
)

# DISTRIBUZIONE PER CIASCUN ITEM IPASE (boxplot, ordinato per mediana)

def plot_item_distributions(df, value_cols, title, filepath):
    # ordino gli item per mediana decrescente
    medians = df[value_cols].median().sort_values(ascending=False)
    ordered_cols = medians.index.tolist()

    data = [df[col].dropna().values for col in ordered_cols]

    fig, ax = plt.subplots(figsize=(10, max(6, len(ordered_cols) * 0.3)))
    bp = ax.boxplot(
        data, vert=False, patch_artist=True, showfliers=True,
        medianprops={"color": "black", "linewidth": 1.5},
        boxprops={"facecolor": "#4c72b0", "alpha": 0.7},
        flierprops={"markersize": 3, "markerfacecolor": "gray", "markeredgecolor": "none"},
    )

    ax.set_yticks(np.arange(1, len(ordered_cols) + 1))
    ax.set_yticklabels(ordered_cols, fontsize=9)
    for lbl in ax.get_yticklabels():
        lbl.set_color(macro_colors.get(item_to_macro.get(lbl.get_text()), "black"))
    ax.invert_yaxis()
    ax.set_xlabel("Score (0-1)", fontsize=12)
    ax.set_title(title, fontsize=15, wrap=True)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=macro_legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, title="Macro-category")

    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.show()
    plt.close(fig)


plot_item_distributions(
    df_all, item_cols,
    title="Item distribution",
    filepath=f"{OUTPUT_DIR}/boxplot_items_distribution.png",
)

# GRIGLIA DI DENSITY PLOT, UNO PER CIASCUN ITEM IPASE
import textwrap
from scipy.stats import gaussian_kde

def plot_item_density_grid(df, value_cols, title, filepath, n_cols=6, color="#4c72b0", title_wrap_width=22):
    n_items = len(value_cols)
    n_rows = int(np.ceil(n_items / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.4))
    axes = axes.flatten()

    x_grid = np.linspace(0, 1, 200)

    for i, col in enumerate(value_cols):
        ax = axes[i]
        vals = df[col].dropna().values

        if len(vals) >= 2 and np.std(vals) > 0:
            kde = gaussian_kde(vals)
            ax.fill_between(x_grid, kde(x_grid), color=color, alpha=0.6)
            ax.plot(x_grid, kde(x_grid), color=color, linewidth=1)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=8)

        wrapped_title = "\n".join(textwrap.wrap(col, width=title_wrap_width))
        ax.set_title(wrapped_title, fontsize=12, color=macro_colors.get(item_to_macro.get(col), "black"))
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=6)

    # nascondo eventuali assi vuoti in eccesso
    for j in range(n_items, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=16, y=1.001)
    fig.legend(handles=macro_legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(macro_legend_handles), fontsize=9)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


plot_item_density_grid(
    df_all, item_cols,
    title="Density per item",
    filepath=f"{OUTPUT_DIR}/density_grid_items.png",
)

"""## 7. Correlazioni su tutto il sample di pazienti (Delusion + Voices) Spearman: item - PANSS, macro-categorie - PANSS, score IPASE tot - PANSS"""

def compute_correlation(df, panss_col, value_cols):
    sub = df.dropna(subset=[panss_col])
    rows = []
    for col in value_cols:
        pair = sub[[panss_col, col]].dropna()
        if len(pair) < 3:
            rows.append({"Variable": col, "r": np.nan, "p_value": np.nan, "N": len(pair), "Significant": False})
            continue
        r, p = spearmanr(pair[panss_col], pair[col])
        rows.append({"Variable": col, "r": round(r, 3), "p_value": round(p, 4), "N": len(pair), "Significant": p < ALPHA})
    return pd.DataFrame(rows)

def plot_correlation_grid(df_corr_dict, row_labels, title, filepath, vmin=-1, vmax=1):
    scale_names = list(df_corr_dict.keys())
    n_rows, n_cols = len(row_labels), len(scale_names)

    r_matrix = np.full((n_rows, n_cols), np.nan)
    p_matrix = np.full((n_rows, n_cols), np.nan)
    sig_matrix = np.zeros((n_rows, n_cols), dtype=bool)

    for j, scale_name in enumerate(scale_names):
        d = df_corr_dict[scale_name].set_index("Variable").reindex(row_labels)
        r_matrix[:, j] = d["r"].values
        p_matrix[:, j] = d["p_value"].values
        sig_matrix[:, j] = d["Significant"].fillna(False).values

    display_vals = np.where(sig_matrix, r_matrix, np.nan)

    fig, ax = plt.subplots(figsize=(3 * n_cols + 3, max(8, n_rows * 0.45)))
    cmap = plt.cm.RdBu_r
    cmap.set_bad(color="#e8e8e8")
    im = ax.imshow(display_vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            r, p = r_matrix[i, j], p_matrix[i, j]
            if np.isnan(r):
                text, color = "n/a", "black"
            else:
                text = f"r={r:.2f}\np={p:.3f}"
                is_dark = sig_matrix[i, j] and (abs(r) / (vmax - vmin) > 0.25)
                color = "white" if is_dark else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=10,
                    color=color, fontweight="bold" if sig_matrix[i, j] else "normal")

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=11)
    for lbl in ax.get_yticklabels():
        lbl.set_color(macro_colors.get(item_to_macro.get(lbl.get_text()), "black"))
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(scale_names, fontsize=13)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_title(title, fontsize=17, wrap=True, pad=40)

    # separatore visivo tra PANSS Van der Gaag e PANSS classico, cosi' le due scale
    # non si confondono a colpo d'occhio
    is_classic = [s in panss_scale_names_classic for s in scale_names]
    if any(is_classic) and not all(is_classic):
        boundary = is_classic.index(True)
        ax.set_ylim(n_rows - 0.5, -1.8)
        ax.plot([boundary - 0.5, boundary - 0.5], [-1.8, n_rows - 0.5],
                color="black", linewidth=2, clip_on=False)
        if boundary > 0:
            ax.text((boundary - 1) / 2, -1.2, "PANSS Van der Gaag",
                     ha="center", va="center", fontsize=12, fontweight="bold")
        if boundary < n_cols:
            ax.text(boundary + (n_cols - boundary - 1) / 2, -1.2, "PANSS classico",
                     ha="center", va="center", fontsize=12, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("r (p < 0.05)", fontsize=10)

    if any(lbl in item_to_macro for lbl in row_labels):
        ax.legend(handles=macro_legend_handles, loc="upper left", bbox_to_anchor=(1.15, 1), fontsize=9, title="Macro-category")

    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.show()
    plt.close(fig)


panss_scales_to_test = [(f"T0_{s}", s) for s in panss_scale_names] + list(
    zip(panss_cols_classic_t0, panss_scale_names_classic)
)

def run_all_sample_correlations(df, suffix, title_suffix):
    """Correlazioni item/macro/mean-score vs PANSS sull'intero campione. `suffix` vuoto
    per la versione con soglia (default), "_raw" per quella senza soglia (media grezza)."""
    item_corr_tables = {}
    for panss_col, scale_name in panss_scales_to_test:
        df_corr = compute_correlation(df, panss_col, item_cols)
        item_corr_tables[scale_name] = df_corr
        df_corr.to_csv(f"{OUTPUT_DIR}/correlation_items_{scale_name}{suffix}.csv", index=False)

    plot_correlation_grid(
        item_corr_tables, item_cols,
        title=f"Correlation IPASE items - PANSS (Spearman){title_suffix}",
        filepath=f"{OUTPUT_DIR}/correlation_grid_items_ALL{suffix}.png",
    )

    macro_corr_tables = {}
    for panss_col, scale_name in panss_scales_to_test:
        df_corr = compute_correlation(df, panss_col, subscale_names)
        macro_corr_tables[scale_name] = df_corr
        df_corr.to_csv(f"{OUTPUT_DIR}/correlation_macro_{scale_name}{suffix}.csv", index=False)

    plot_correlation_grid(
        macro_corr_tables, subscale_names,
        title=f"Correlation IPASE Macro - PANSS (Spearman){title_suffix}",
        filepath=f"{OUTPUT_DIR}/correlation_grid_macro_ALL{suffix}.png",
    )

    if "IPASE_Mean_Score" not in df.columns:
        df["IPASE_Mean_Score"] = df[item_cols].mean(axis=1)

    mean_score_corr_tables = {}
    for panss_col, scale_name in panss_scales_to_test:
        df_corr = compute_correlation(df, panss_col, ["IPASE_Mean_Score"])
        mean_score_corr_tables[scale_name] = df_corr
        df_corr.to_csv(f"{OUTPUT_DIR}/correlation_meanscore_{scale_name}{suffix}.csv", index=False)

    plot_correlation_grid(
        mean_score_corr_tables, ["IPASE_Mean_Score"],
        title=f"Correlation IPASE Mean Score - PANSS (Spearman){title_suffix}",
        filepath=f"{OUTPUT_DIR}/correlation_grid_meanscore_ALL{suffix}.png",
    )

    return item_corr_tables, macro_corr_tables, mean_score_corr_tables


item_corr_tables, macro_corr_tables, mean_score_corr_tables = run_all_sample_correlations(
    df_patient_avg, "", " — con soglia di rilevanza"
)
item_corr_tables_raw, macro_corr_tables_raw, mean_score_corr_tables_raw = run_all_sample_correlations(
    df_patient_avg_raw, "_raw", " — senza soglia (media grezza)"
)

"""## 7.1 Correlazioni separate per gruppo (Delusion / Voices / ALL)"""

# Correlazioni (item, macro, punteggio medio) separatamente per Delusion e Voices
def run_correlation_suite(df_subset, group_label, output_dir):
    """Calcola e salva le correlazioni Spearman (item, macro-categorie, punteggio
    IPASE medio complessivo) vs le sottoscale PANSS, per un sottoinsieme di pazienti."""

    value_col_groups = {
        "items": item_cols,
        "macro": subscale_names,
        "mean_score": ["IPASE_Mean_Score"],
    }

    results = {}

    for value_name, value_cols in value_col_groups.items():
        corr_tables = {}
        for panss_col, scale_name in panss_scales_to_test:
            df_corr = compute_correlation(df_subset, panss_col, value_cols)
            corr_tables[scale_name] = df_corr
            df_corr.to_csv(f"{output_dir}/correlation_{value_name}_{scale_name}_{group_label}.csv", index=False)

        results[value_name] = corr_tables

        plot_correlation_grid(
            corr_tables, value_cols,
            title=f"Correlation IPASE {value_name} - PANSS (Spearman) — {group_label}",
            filepath=f"{output_dir}/correlation_grid_{value_name}_{group_label}.png",
        )

    return results

correlation_results_by_group = {}

for df_source, label_suffix in [(df_patient_avg, ""), (df_patient_avg_raw, "_raw")]:
    for base_group_label in ["Delusion", "Voices"]:
        df_subset = df_source[df_source["Group"] == base_group_label]
        group_label = base_group_label + label_suffix

        print(f"\nCorrelazioni per gruppo: {group_label} (N pazienti = {df_subset['File'].nunique()})")
        correlation_results_by_group[group_label] = run_correlation_suite(df_subset, group_label, OUTPUT_DIR)

"""### Riepilogo: item e macro-categorie significativi per gruppo"""

summary_rows = []

for group_label, results in correlation_results_by_group.items():
    for value_name, corr_tables in results.items():
        for scale_name, df_corr in corr_tables.items():
            n_sig = df_corr["Significant"].sum()
            summary_rows.append({
                "Group": group_label,
                "Variable_type": value_name,
                "PANSS_scale": scale_name,
                "N_Significant": n_sig,
                "N_Total": len(df_corr),
            })

df_correlation_summary = pd.DataFrame(summary_rows)
df_correlation_summary.to_csv(f"{OUTPUT_DIR}/correlation_summary_by_group.csv", index=False)
print(df_correlation_summary.to_string(index=False))
df_correlation_summary

"""## 8 HEATMAP SOURCE x IPASE
senza threshold e con threshold SOLO PER LA VISUALIZZAZIONE
"""

# MACRO-CATEGORIE ANCHE A LIVELLO DI SINGOLA METAFORA
for name, titles in subscale_titles.items():
    titles_present = [t for t in titles if t in df_all.columns]
    df_all[name] = df_all[titles_present].mean(axis=1)

df_all["SOURCE1"] = df_all["SOURCE1"].astype(str).str.strip()
df_all.loc[df_all["SOURCE1"].isin(["nan", "", "None"]), "SOURCE1"] = np.nan

# HEATMAP: righe = item/macro, colonne = SOURCE1 — mediana robusta (bootstrap CI + shrinkage)
N_BOOT = 1000          # ricampionamenti bootstrap per l'IC della mediana
CI_LEVEL = 95           # livello dell'intervallo di confidenza (%)
SHRINK_K = 5            # costante di shrinkage (James-Stein-like): w = n / (n + SHRINK_K)


def bootstrap_stat_ci(values, n_boot=N_BOOT, ci=CI_LEVEL, seed=42, stat="median"):
    """Bootstrap con reinserimento: ricampiona `values` n_boot volte, calcola la
    statistica (mediana per punteggi continui, media/proporzione per item binarizzati
    dalla soglia di rilevanza) ad ogni ricampionamento e ne ricava l'intervallo di
    confidenza."""
    stat_func = np.mean if stat == "mean" else np.median
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot_vals = np.array([
        stat_func(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return stat_func(values), np.percentile(boot_vals, lo), np.percentile(boot_vals, hi)


def compute_source_stats(df, value_cols, source_col, min_n=3,
                          n_boot=N_BOOT, ci=CI_LEVEL, shrink_k=SHRINK_K, seed=42, stat="median"):
    """Per ogni (source, value_col): n, statistica grezza, IC bootstrap e statistica con
    shrinkage verso il valore globale (empirical Bayes / James-Stein-like), pesata per n:
        shrunk = w * locale + (1 - w) * globale,   w = n / (n + shrink_k)
    Le categorie con pochi dati (n piccolo -> w piccolo) vengono quindi "tirate" verso
    il valore generale, riducendo il rumore delle stime poco supportate.
    `stat="mean"` per dati item gia' binarizzati dalla soglia di rilevanza (la media di
    0/1 e' la proporzione di metafore rilevanti); `stat="median"` per punteggi continui
    (es. macro-categorie, che essendo gia' medie su piu' item sono meno bimodali)."""
    sub = df.dropna(subset=[source_col])
    counts = sub[source_col].value_counts()
    valid_sources = counts[counts >= min_n].index.tolist()
    sub = sub[sub[source_col].isin(valid_sources)]

    global_stat = df[value_cols].mean() if stat == "mean" else df[value_cols].median()

    records = []
    for source in valid_sources:
        grp = sub[sub[source_col] == source]
        n = len(grp)
        w = n / (n + shrink_k)
        for col in value_cols:
            val, lo, hi = bootstrap_stat_ci(grp[col].values, n_boot=n_boot, ci=ci, seed=seed, stat=stat)
            ci_width = hi - lo if not (np.isnan(hi) or np.isnan(lo)) else np.nan
            shrunk = w * val + (1 - w) * global_stat[col] if not np.isnan(val) else np.nan
            records.append({
                "source": source, "value_col": col, "n": n,
                "stat_value": val, "ci_low": lo, "ci_high": hi, "ci_width": ci_width,
                "shrunk_value": shrunk,
            })
    return pd.DataFrame(records), valid_sources, counts


def plot_heatmap_by_source_median(df, value_cols, source_col, title, filepath,
                                   min_n=3, cmap_name="RdBu_r", vmin=-1, vmax=1, threshold=None,
                                   use_shrunk=True, max_ci_width=None, n_boot=N_BOOT,
                                   ci=CI_LEVEL, shrink_k=SHRINK_K, stat="median"):
    """Come plot_heatmap_by_source_threshold, ma con mediana (o media/proporzione, vedi
    `stat`) al posto della statistica di cella, annotata con il semi-IC bootstrap. Mostra
    tutte le celle (nessun filtro sul valore) a meno che `threshold` non venga impostato
    esplicitamente; le celle con IC troppo ampio (> max_ci_width, se specificato) vengono
    escluse (lasciate grigie/senza numero)."""
    stats, valid_sources, counts = compute_source_stats(
        df, value_cols, source_col, min_n=min_n, n_boot=n_boot, ci=ci, shrink_k=shrink_k, stat=stat,
    )

    value_field = "shrunk_value" if use_shrunk else "stat_value"
    matrix = stats.pivot(index="value_col", columns="source", values=value_field).reindex(
        index=value_cols, columns=valid_sources)
    ci_width_matrix = stats.pivot(index="value_col", columns="source", values="ci_width").reindex(
        index=value_cols, columns=valid_sources)

    n_rows, n_cols = matrix.shape
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.7), max(6, n_rows * 0.4)))

    if max_ci_width is None:
        uncertain = pd.DataFrame(False, index=ci_width_matrix.index, columns=ci_width_matrix.columns)
    else:
        uncertain = ci_width_matrix.isna() | (ci_width_matrix > max_ci_width)
    above_threshold = matrix.values > threshold if threshold is not None else ~np.isnan(matrix.values)
    display_vals = np.where(above_threshold & ~uncertain.values, matrix.values, np.nan)

    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#e8e8e8")  # celle sotto soglia o con IC troppo ampio: grigio chiaro
    im = ax.imshow(display_vals, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix.values[i, j]
            if np.isnan(val) or (threshold is not None and val <= threshold):
                continue
            ciw = ci_width_matrix.values[i, j]
            if max_ci_width is not None and (np.isnan(ciw) or ciw > max_ci_width):
                continue  # esclusa: intervallo di confidenza troppo ampio
            text_color = "white" if abs(val) > (vmax - vmin) * 0.25 else "black"
            label = f"{val:.2f}\n±{ciw/2:.2f}" if not np.isnan(ciw) else f"{val:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=text_color)

    xticklabels = [f"{s} (n={counts[s]})" for s in valid_sources]
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(xticklabels, fontsize=10, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(matrix.index.tolist(), fontsize=10)
    for lbl in ax.get_yticklabels():
        lbl.set_color(macro_colors.get(item_to_macro.get(lbl.get_text()), "black"))

    ax.set_title(title, fontsize=16, wrap=True)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    base_label = "proportion" if stat == "mean" else "median"
    stat_label = f"Shrunk {base_label}" if use_shrunk else base_label.capitalize()
    score_note = f"(solo > {threshold})" if threshold is not None else "(tutti i punteggi)"
    cbar.set_label(f"{stat_label} NLI score ± bootstrap CI/2 {score_note}", fontsize=11)

    if any(lbl in item_to_macro for lbl in matrix.index.tolist()):
        ax.legend(handles=macro_legend_handles, loc="upper left", bbox_to_anchor=(1.15, 1), fontsize=9, title="Macro-category")

    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.show()
    plt.close(fig)

    return stats

# heatmap ITEM: uso i punteggi binarizzati dalla soglia di rilevanza (proporzione di
# metafore rilevanti per source, non mediana del punteggio continuo grezzo)
df_all_item_rel = df_all.copy()
for c in item_cols:
    df_all_item_rel[c] = df_all[f"{c}__rel"]

plot_heatmap_by_source_median(
    df_all_item_rel, item_cols, source_col="SOURCE1",
    title=f"Item IPASE × SOURCE (proporzione rilevanti, score > {RELEVANCE_THRESHOLD})",
    filepath=f"{OUTPUT_DIR}/heatmap_items_by_SOURCE1_median.png",
    stat="mean",
)

# heatmap ITEM senza soglia: mediana del punteggio continuo grezzo (come prima di
# introdurre RELEVANCE_THRESHOLD), tenuta in parallelo per confronto
plot_heatmap_by_source_median(
    df_all, item_cols, source_col="SOURCE1",
    title="Item IPASE × SOURCE (median, senza soglia, tutti i punteggi)",
    filepath=f"{OUTPUT_DIR}/heatmap_items_by_SOURCE1_median_raw.png",
)

# heatmap MACRO: resta sul punteggio continuo (mediana), le macro-categorie sono gia'
# medie su piu' item quindi molto meno bimodali
plot_heatmap_by_source_median(
    df_all, subscale_names, source_col="SOURCE1",
    title="IPASE MACRO × SOURCE (median, tutti i punteggi)",
    filepath=f"{OUTPUT_DIR}/heatmap_macro_by_SOURCE1_median.png",
)

"""## 8.1 Item IPASE ad alta prevalenza: heatmap filtrata + correlazioni con PANSS per SOURCE domain

Si selezionano gli item IPASE con proporzione globale di metafore rilevanti (score >
RELEVANCE_THRESHOLD, su tutte le metafore indipendentemente dal source) sopra il 75°
percentile della distribuzione — una soglia oggettiva per isolare gli item piu' "presenti"
nel corpus di metafore. Su questo sottoinsieme di item:
1. si rigenera la heatmap item x SOURCE tagliando (celle grigie) gli item a bassa prevalenza;
2. si calcolano le correlazioni Spearman item-PANSS separatamente per ciascun SOURCE domain
   con almeno `MIN_PATIENTS_PER_SOURCE` pazienti che hanno metafore in quel dominio.
"""

item_prevalence_global = pd.Series({c: df_all[f"{c}__rel"].mean() for c in item_cols})
ITEM_PREVALENCE_Q3 = item_prevalence_global.quantile(0.75)
item_cols_high_median = item_prevalence_global[item_prevalence_global > ITEM_PREVALENCE_Q3].index.tolist()
print(f"Item IPASE ad alta prevalenza (proporzione globale > Q3 = {ITEM_PREVALENCE_Q3:.3f}): {len(item_cols_high_median)}/{len(item_cols)}")
print(item_cols_high_median)

plot_heatmap_by_source_median(
    df_all_item_rel, item_cols_high_median, source_col="SOURCE1",
    title=f"Item IPASE × SOURCE (proporzione rilevanti) — solo item ad alta prevalenza (> Q3={ITEM_PREVALENCE_Q3:.2f})",
    filepath=f"{OUTPUT_DIR}/heatmap_items_by_SOURCE1_median_highmedian_only.png",
    stat="mean",
)

# VERSIONE SENZA SOGLIA: selezione item ad alta MEDIANA grezza (come prima di introdurre
# RELEVANCE_THRESHOLD), tenuta in parallelo per confronto
item_medians_global_raw = df_all[item_cols].median()
ITEM_MEDIAN_Q3_RAW = item_medians_global_raw.quantile(0.75)
item_cols_high_median_raw = item_medians_global_raw[item_medians_global_raw > ITEM_MEDIAN_Q3_RAW].index.tolist()
print(f"Item IPASE a mediana grezza alta, senza soglia (mediana globale > Q3 = {ITEM_MEDIAN_Q3_RAW:.3f}): {len(item_cols_high_median_raw)}/{len(item_cols)}")

plot_heatmap_by_source_median(
    df_all, item_cols_high_median_raw, source_col="SOURCE1",
    title=f"Item IPASE × SOURCE (median, senza soglia) — solo item a mediana alta (> Q3={ITEM_MEDIAN_Q3_RAW:.2f})",
    filepath=f"{OUTPUT_DIR}/heatmap_items_by_SOURCE1_median_highmedian_only_raw.png",
)

MIN_PATIENTS_PER_SOURCE = 10
SOURCE_CORR_DIR = f"{OUTPUT_DIR}/correlation_by_source"
SOURCE_CORR_DIR_RAW = f"{OUTPUT_DIR}/correlation_by_source_raw"
os.makedirs(SOURCE_CORR_DIR, exist_ok=True)
os.makedirs(SOURCE_CORR_DIR_RAW, exist_ok=True)


def build_patient_by_source(df_all, value_cols, source_col="SOURCE1", panss_df=panss_all,
                             panss_cols=panss_cols_t0 + panss_cols_classic_t0, use_relevance=True):
    """Aggregazione a livello paziente x source domain, solo sulle metafore di quel
    paziente appartenenti a quel source, poi merge con PANSS. Se `use_relevance` e' True
    (default): proporzione di metafore con score > RELEVANCE_THRESHOLD (vedi nota sulla
    bimodalita' degli item); se False: media grezza continua, senza soglia."""
    sub = df_all.dropna(subset=[source_col])
    if use_relevance:
        rel_cols = [f"{c}__rel" for c in value_cols]
        agg = sub.groupby(["File", "Group", source_col])[rel_cols].mean().rename(
            columns=dict(zip(rel_cols, value_cols))
        )
    else:
        agg = sub.groupby(["File", "Group", source_col])[value_cols].mean()
    return (
        agg.reset_index()
        .merge(panss_df[["File"] + panss_cols], on="File", how="left")
    )


def run_source_correlations(df_all, item_cols_set, use_relevance, corr_dir, label):
    """Correlazioni item-PANSS per ciascun SOURCE domain con >= MIN_PATIENTS_PER_SOURCE
    pazienti; `label` distingue i titoli/print tra versione con e senza soglia."""
    df_patient_by_source = build_patient_by_source(df_all, item_cols_set, use_relevance=use_relevance)

    source_patient_counts = df_all.dropna(subset=["SOURCE1"]).groupby("SOURCE1")["File"].nunique()
    valid_sources = source_patient_counts[source_patient_counts >= MIN_PATIENTS_PER_SOURCE].index.tolist()
    print(f"SOURCE domain con >= {MIN_PATIENTS_PER_SOURCE} pazienti ({label}): {len(valid_sources)}/{source_patient_counts.shape[0]}")

    correlation_results = {}
    summary_rows = []

    for source in valid_sources:
        df_source = df_patient_by_source[df_patient_by_source["SOURCE1"] == source]
        n_patients_source = df_source["File"].nunique()
        safe_source = re.sub(r"[^A-Za-z0-9]+", "_", source).strip("_")

        corr_tables = {}
        for panss_col, scale_name in panss_scales_to_test:
            df_corr = compute_correlation(df_source, panss_col, item_cols_set)
            corr_tables[scale_name] = df_corr
            df_corr.to_csv(f"{corr_dir}/correlation_items_highmedian_{scale_name}_{safe_source}.csv", index=False)
            summary_rows.append({
                "SOURCE1": source, "N_Patients": n_patients_source, "PANSS_scale": scale_name,
                "N_Significant": df_corr["Significant"].sum(), "N_Total": len(df_corr),
            })
        correlation_results[source] = corr_tables

        plot_correlation_grid(
            corr_tables, item_cols_set,
            title=f"Correlation IPASE item ({label}) - PANSS (Spearman) — SOURCE: {source} (n={n_patients_source})",
            filepath=f"{corr_dir}/correlation_grid_items_highmedian_{safe_source}.png",
        )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(f"{corr_dir}/correlation_summary_by_source.csv", index=False)
    print(df_summary.to_string(index=False))
    return correlation_results, df_summary


correlation_results_by_source, df_correlation_summary_by_source = run_source_correlations(
    df_all, item_cols_high_median, use_relevance=True, corr_dir=SOURCE_CORR_DIR, label="proporzione rilevanti"
)
correlation_results_by_source_raw, df_correlation_summary_by_source_raw = run_source_correlations(
    df_all, item_cols_high_median_raw, use_relevance=False, corr_dir=SOURCE_CORR_DIR_RAW, label="senza soglia"
)

"""## 8.2 Heatmap MACRO a prevalenza + heatmap ITEM con clustering gerarchico

Due estensioni della heatmap SOURCE:
1. **Macro a prevalenza**: la heatmap macro x SOURCE usa la mediana continua del composito
   per-metafora; qui si applica la stessa logica di soglia usata per gli item (binarizzo
   il composito macro a RELEVANCE_THRESHOLD e mostro la proporzione), per confronto diretto
   con la versione item.
2. **Clustering gerarchico**: invece di ordinare righe/colonne per macro-categoria/frequenza,
   le riordino per somiglianza (linkage gerarchico su distanza euclidea), cosi' item e source
   con pattern simili finiscono vicini. Uso anche una soglia piu' stringente (90° percentile
   invece di Q3=75°) per isolare solo gli item piu' estremi.
"""

# MACRO A PREVALENZA: binarizzo il composito macro per-metafora a RELEVANCE_THRESHOLD
macro_cols_rel = [f"{name}__rel" for name in subscale_names]
df_all[macro_cols_rel] = (df_all[subscale_names] > RELEVANCE_THRESHOLD).astype(int)
df_all_macro_rel = df_all.copy()
for name in subscale_names:
    df_all_macro_rel[name] = df_all[f"{name}__rel"]

plot_heatmap_by_source_median(
    df_all_macro_rel, subscale_names, source_col="SOURCE1",
    title=f"IPASE MACRO × SOURCE (proporzione rilevanti, score > {RELEVANCE_THRESHOLD})",
    filepath=f"{OUTPUT_DIR}/heatmap_macro_by_SOURCE1_median_prevalence.png",
    stat="mean",
)

# CLUSTERING GERARCHICO: item ad alta prevalenza/mediana (top decile), righe e colonne
# riordinate per somiglianza invece che per macro-categoria/frequenza
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

CLUSTER_QUANTILE = 0.90


def cluster_order(matrix_df):
    """Ordina righe e colonne per somiglianza (clustering gerarchico, distanza euclidea,
    linkage average); i NaN vengono imputati con la media globale solo per il calcolo
    delle distanze, i valori mostrati in heatmap restano quelli originali."""
    filled = matrix_df.fillna(matrix_df.mean(axis=1).mean())
    row_order = leaves_list(linkage(pdist(filled.values, metric="euclidean"), method="average"))
    col_order = leaves_list(linkage(pdist(filled.values.T, metric="euclidean"), method="average"))
    return matrix_df.index[row_order].tolist(), matrix_df.columns[col_order].tolist()


def plot_clustered_heatmap(df, value_cols, title, filepath, stat="median", vmin=-1, vmax=1):
    stats, valid_sources, counts = compute_source_stats(df, value_cols, "SOURCE1", stat=stat)
    matrix = stats.pivot(index="value_col", columns="source", values="shrunk_value").reindex(
        index=value_cols, columns=valid_sources)

    row_order, col_order = cluster_order(matrix)
    matrix = matrix.reindex(index=row_order, columns=col_order)

    n_rows, n_cols = matrix.shape
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.7), max(6, n_rows * 0.45)))
    cmap = plt.cm.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#e8e8e8")
    im = ax.imshow(matrix.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix.values[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if abs(val) > (vmax - vmin) * 0.25 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    xticklabels = [f"{s} (n={counts[s]})" for s in matrix.columns]
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(xticklabels, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(matrix.index.tolist(), fontsize=10)
    for lbl in ax.get_yticklabels():
        lbl.set_color(macro_colors.get(item_to_macro.get(lbl.get_text()), "black"))
    ax.set_title(title, fontsize=15, wrap=True)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(f"Shrunk {stat}", fontsize=10)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.show()
    plt.close(fig)
    return matrix


item_prevalence_q90 = item_prevalence_global.quantile(CLUSTER_QUANTILE)
item_cols_top_decile = item_prevalence_global[item_prevalence_global > item_prevalence_q90].index.tolist()
print(f"Item a prevalenza top {int((1 - CLUSTER_QUANTILE) * 100)}% (> {item_prevalence_q90:.3f}): {len(item_cols_top_decile)}")
print(item_cols_top_decile)

plot_clustered_heatmap(
    df_all_item_rel, item_cols_top_decile,
    title=f"Item IPASE × SOURCE, clustered (proporzione rilevanti, top {int((1 - CLUSTER_QUANTILE) * 100)}%)",
    filepath=f"{OUTPUT_DIR}/heatmap_items_by_SOURCE1_clustered_top10.png",
    stat="mean",
)

item_medians_q90_raw = item_medians_global_raw.quantile(CLUSTER_QUANTILE)
item_cols_top_decile_raw = item_medians_global_raw[item_medians_global_raw > item_medians_q90_raw].index.tolist()
print(f"Item a mediana top {int((1 - CLUSTER_QUANTILE) * 100)}%, senza soglia (> {item_medians_q90_raw:.3f}): {len(item_cols_top_decile_raw)}")
print(item_cols_top_decile_raw)

plot_clustered_heatmap(
    df_all, item_cols_top_decile_raw,
    title=f"Item IPASE × SOURCE, clustered (median, senza soglia, top {int((1 - CLUSTER_QUANTILE) * 100)}%)",
    filepath=f"{OUTPUT_DIR}/heatmap_items_by_SOURCE1_clustered_top10_raw.png",
    stat="median",
)

"""##9. Tabelle descrittive: N pazienti, N metafore, PANSS medio per sottoscala"""

panss_delusions_labeled = panss_delusions.copy()
panss_delusions_labeled["Group"] = "Delusion"

panss_voices_labeled = panss_voices.copy()
panss_voices_labeled["Group"] = "Voices"

total_population = pd.concat([panss_delusions_labeled, panss_voices_labeled], ignore_index=True)


def build_descriptive_table(df_metaphors, df_panss_source, total_population, panss_cols, group_col="Group"):
    rows = []
    for group in [None] + sorted(df_metaphors[group_col].dropna().unique().tolist()):
        m_sub = df_metaphors if group is None else df_metaphors[df_metaphors[group_col] == group]
        panss_sub = df_panss_source if group is None else df_panss_source[df_panss_source[group_col] == group]
        t_sub = total_population if group is None else total_population[total_population[group_col] == group]

        all_patient_ids = t_sub["File"].unique()
        metaphors_per_patient = m_sub.groupby("File").size().reindex(all_patient_ids, fill_value=0)

        row = {
            "Group": "TOTALE" if group is None else group,
            "N_Patients_Total": t_sub["File"].nunique(),
            "N_Patients_With_Metaphors": m_sub["File"].nunique(),
            "N_Metaphors": len(m_sub),
            "Metaphors_per_Patient_mean": round(metaphors_per_patient.mean(), 2),
            "Metaphors_per_Patient_std": round(metaphors_per_patient.std(), 2),
        }
        for c in panss_cols:
            row[f"{c}_mean"] = round(panss_sub[c].mean(), 2) if c in panss_sub.columns else np.nan
            row[f"{c}_std"] = round(panss_sub[c].std(), 2) if c in panss_sub.columns else np.nan

        rows.append(row)
    return pd.DataFrame(rows)


def combine_mean_std(df, decimals=2):
    df = df.copy()
    for mean_col in [c for c in df.columns if c.endswith("_mean")]:
        base, std_col = mean_col[:-5], mean_col[:-5] + "_std"
        if std_col not in df.columns:
            continue
        df[base] = df.apply(
            lambda r: np.nan if pd.isna(r[mean_col])
            else f"{r[mean_col]:.{decimals}f} (SD={r[std_col]:.{decimals}f})" if pd.notna(r[std_col])
            else f"{r[mean_col]:.{decimals}f} (SD=NA)",
            axis=1
        )
        df = df.drop(columns=[mean_col, std_col])
    return df


#  calcolo su chi ha metafore
df_descriptive = build_descriptive_table(df_all, df_patient_avg, total_population, panss_cols_t0)
df_descriptive.to_csv(f"{OUTPUT_DIR}/descriptive_summary_ALL.csv", index=False)
combine_mean_std(df_descriptive).to_csv(f"{OUTPUT_DIR}/descriptive_summary_ALL_formatted.csv", index=False)

#  calcolo su tutta la popolazione registrata
df_descriptive_total = build_descriptive_table(df_all, total_population, total_population, panss_cols_t0)
df_descriptive_total.to_csv(f"{OUTPUT_DIR}/descriptive_summary_ALL_total_population.csv", index=False)
combine_mean_std(df_descriptive_total).to_csv(f"{OUTPUT_DIR}/descriptive_summary_ALL_total_population_formatted.csv", index=False)

df_descriptive_total

"""## 9.1 Tabella descrittiva CORPUS: lunghezza testo (parole/caratteri), totale e per gruppo"""

df_all["N_Words"] = df_all["Metaphor_Text"].astype(str).str.split().apply(len)
df_all["N_Chars"] = df_all["Metaphor_Text"].astype(str).str.len()


def build_corpus_table(df, group_col="Group"):
    rows = []
    for group in [None] + sorted(df[group_col].dropna().unique().tolist()):
        sub = df if group is None else df[df[group_col] == group]
        per_patient = sub.groupby("File").size()
        rows.append({
            "Group": "TOTALE" if group is None else group,
            "N_Metaphors": len(sub),
            "N_Patients": sub["File"].nunique(),
            "Metaphors_per_Patient_mean": round(per_patient.mean(), 2),
            "Metaphors_per_Patient_std": round(per_patient.std(), 2),
            "Metaphors_per_Patient_min": int(per_patient.min()),
            "Metaphors_per_Patient_max": int(per_patient.max()),
            "Total_Words": int(sub["N_Words"].sum()),
            "Words_per_Metaphor_mean": round(sub["N_Words"].mean(), 2),
            "Words_per_Metaphor_std": round(sub["N_Words"].std(), 2),
            "Chars_per_Metaphor_mean": round(sub["N_Chars"].mean(), 2),
            "Chars_per_Metaphor_std": round(sub["N_Chars"].std(), 2),
        })
    return pd.DataFrame(rows)


df_corpus_descriptive = build_corpus_table(df_all)
df_corpus_descriptive.to_csv(f"{OUTPUT_DIR}/descriptive_corpus_table.csv", index=False)
print("Tabella descrittiva corpus:", df_corpus_descriptive.shape)
df_corpus_descriptive

"""## 9.2 Tabella descrittiva SOGGETTI: anagrafica, PANSS (entrambe le scale), pragmatica ricettiva

Delusion e Voices sono state valutate con strumenti di pragmatica diversi
(APACS-BRIEF vs PMM): i punteggi vanno letti entro gruppo, non confrontati
direttamente tra gruppi.
"""

dd_raw_full = pd.read_excel(PATHS["database_delusions"]).rename(columns={"ID": "File"})
dd_raw_full["File"] = dd_raw_full["File"].astype(str).str.strip()
dv_raw_full = pd.read_excel(PATHS["database_voices"]).rename(columns={"ID": "File"})
dv_raw_full["File"] = dv_raw_full["File"].astype(str).str.strip()

subject_delusion = dd_raw_full[[
    "File", "Età", "Sex", "Scolarità (anni)", "T0_SOFAS", "Diagnosis Grouping",
    "T0_APACS_BRIEF_A_INTERVISTA_PROP", "T0_APACS_BRIEF_A_BRANO_PROP",
    "T0_APACS_BRIEF_A_LING_FIG_1_PROP", "T0_APACS_BRIEF_A_UMORISMO_PROP",
    "T0_APACS_BRIEF_A_Fig2_PROP",
    "T0_APACS_BRIEF_A_COMP_PROP", "T0_APACS_BRIEF_A_PROD_PROP", "T0_APACS_BRIEF_A_TOTAL",
]].merge(panss_delusions, on="File", how="left")

subject_voices = dv_raw_full[[
    "File", "SCOLARITA'", "ONSET", "CORRETTEZZA", "INTERPRETAZIONE", "TOTALE",
]].merge(panss_voices, on="File", how="left")


def mean_std_row(series, decimals=2):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "N/D"
    return f"{s.mean():.{decimals}f} ± {s.std():.{decimals}f} (n={len(s)})"


df_panss_vdg_subject = pd.DataFrame([
    {
        "Sottoscala": c.replace("T0_", "").replace("_PANSS", ""),
        "Delusion": mean_std_row(subject_delusion[c]),
        "Voices": mean_std_row(subject_voices[c]),
    }
    for c in panss_cols_t0
])
df_panss_vdg_subject.to_csv(f"{OUTPUT_DIR}/descriptive_subject_panss_vandergaag.csv", index=False)

df_panss_classic_subject = pd.DataFrame([
    {
        "Sottoscala": c.replace("T0_PANSS_", ""),
        "Delusion": mean_std_row(subject_delusion[c]),
        "Voices": mean_std_row(subject_voices[c]),
    }
    for c in panss_cols_classic_t0
])
df_panss_classic_subject.to_csv(f"{OUTPUT_DIR}/descriptive_subject_panss_classic.csv", index=False)

sex_counts = subject_delusion["Sex"].value_counts()
sex_summary = ", ".join(f"{k}:{v}" for k, v in sex_counts.items())
df_demographics_subject = pd.DataFrame([
    {"Misura": "Età", "Delusion": mean_std_row(subject_delusion["Età"]), "Voices": "N/D (non disponibile)"},
    {"Misura": "Sesso (M/F/altro)", "Delusion": sex_summary, "Voices": "N/D (non disponibile)"},
    {"Misura": "Scolarità (anni)", "Delusion": mean_std_row(subject_delusion["Scolarità (anni)"]), "Voices": mean_std_row(subject_voices["SCOLARITA'"])},
    {"Misura": "Età esordio (onset)", "Delusion": "N/D", "Voices": mean_std_row(subject_voices["ONSET"])},
    {"Misura": "SOFAS", "Delusion": mean_std_row(subject_delusion["T0_SOFAS"]), "Voices": "N/D (non disponibile)"},
])
df_demographics_subject.to_csv(f"{OUTPUT_DIR}/descriptive_subject_demographics.csv", index=False)

diagnosis_counts = subject_delusion["Diagnosis Grouping"].value_counts(dropna=False)
diagnosis_total = diagnosis_counts.sum()
df_diagnosis_subject = pd.DataFrame([
    {
        "Diagnosis Grouping": "Non specificato/anonimizzato" if diag == "AAA" else diag,
        "N (%)": f"{n} ({n / diagnosis_total * 100:.1f}%)",
    }
    for diag, n in diagnosis_counts.items()
])
df_diagnosis_subject.to_csv(f"{OUTPUT_DIR}/descriptive_subject_diagnosis_delusion.csv", index=False)

df_pragmatics_subject = pd.DataFrame([
    {"Gruppo": "Delusion", "Strumento": "APACS-BRIEF", "Misura": "Intervista (INTERVISTA_PROP)", "Valore": mean_std_row(subject_delusion["T0_APACS_BRIEF_A_INTERVISTA_PROP"])},
    {"Gruppo": "Delusion", "Strumento": "APACS-BRIEF", "Misura": "Discorso narrativo (BRANO_PROP)", "Valore": mean_std_row(subject_delusion["T0_APACS_BRIEF_A_BRANO_PROP"])},
    {"Gruppo": "Delusion", "Strumento": "APACS-BRIEF", "Misura": "Linguaggio figurato (LING_FIG_1_PROP)", "Valore": mean_std_row(subject_delusion["T0_APACS_BRIEF_A_LING_FIG_1_PROP"])},
    {"Gruppo": "Delusion", "Strumento": "APACS-BRIEF", "Misura": "Umorismo (UMORISMO_PROP)", "Valore": mean_std_row(subject_delusion["T0_APACS_BRIEF_A_UMORISMO_PROP"])},
    {"Gruppo": "Delusion", "Strumento": "APACS-BRIEF", "Misura": "Linguaggio figurato 2 (Fig2_PROP)", "Valore": mean_std_row(subject_delusion["T0_APACS_BRIEF_A_Fig2_PROP"])},
    {"Gruppo": "Delusion", "Strumento": "APACS-BRIEF", "Misura": "Comprensione (COMP_PROP)", "Valore": mean_std_row(subject_delusion["T0_APACS_BRIEF_A_COMP_PROP"])},
    {"Gruppo": "Delusion", "Strumento": "APACS-BRIEF", "Misura": "Produzione (PROD_PROP)", "Valore": mean_std_row(subject_delusion["T0_APACS_BRIEF_A_PROD_PROP"])},
    {"Gruppo": "Delusion", "Strumento": "APACS-BRIEF", "Misura": "Totale", "Valore": mean_std_row(subject_delusion["T0_APACS_BRIEF_A_TOTAL"])},
    {"Gruppo": "Voices", "Strumento": "PMM", "Misura": "Correttezza", "Valore": mean_std_row(subject_voices["CORRETTEZZA"])},
    {"Gruppo": "Voices", "Strumento": "PMM", "Misura": "Interpretazione", "Valore": mean_std_row(subject_voices["INTERPRETAZIONE"])},
    {"Gruppo": "Voices", "Strumento": "PMM", "Misura": "Totale", "Valore": mean_std_row(subject_voices["TOTALE"])},
])
df_pragmatics_subject.to_csv(f"{OUTPUT_DIR}/descriptive_subject_pragmatics.csv", index=False)

print("Tabella soggetti — Anagrafica:")
print(df_demographics_subject.to_string(index=False))
print("\nTabella soggetti — PANSS Van der Gaag:")
print(df_panss_vdg_subject.to_string(index=False))
print("\nTabella soggetti — PANSS classico:")
print(df_panss_classic_subject.to_string(index=False))
print("\nTabella soggetti — Pragmatica ricettiva:")
print(df_pragmatics_subject.to_string(index=False))
print("\nTabella soggetti — Diagnosi (solo Delusion):")
print(df_diagnosis_subject.to_string(index=False))