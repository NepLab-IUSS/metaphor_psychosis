# -*- coding: utf-8 -*-
"""
Scoring delle metafore (Schizo_Metaphors.xlsx) sui 57 item della scala EASE, usando come
ipotesi NLI le "domande possibili" del clinico (foglio EASE originale) convertite da
domanda a frase dichiarativa in prima persona — invece della parafrasi delle definizioni
cliniche usata in EASE_Analysis.py. Per i 4 item senza una domanda d'esempio nel foglio
originale (5.3, 5.4, 5.5, 5.6) si riusa la stessa frase di EASE_Analysis.py (segnalato nei
commenti sotto).

Stessa logica di classificazione NLI di NLI_Analysis.py / EASE_Analysis.py. Oltre allo
scoring, produce le stesse descrittive visive di EASE_Analysis.py (barplot item con
parentesi di dominio, boxplot di distribuzione); niente merge PANSS/aggregazione
paziente/correlazioni in questo script.
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
}

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXTRA_COLS = ["vehicle1", "vehicle2", "vehicle3", "TARGET", "SOURCE1", "SOURCE2"]

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# SCALA EASE — 57 item, ipotesi derivate dalle "domande possibili" del clinico (foglio
# EASE originale), convertite da domanda in prima/seconda persona a frase dichiarativa
# in prima persona. Stessi titolo/id/dominio di EASE_Analysis.py per confrontabilita'.
items = [
    # 1. Cognitivita' e flusso della coscienza
    {"id": 1, "titolo": "Interferenza del pensiero", "testo": "Mi capita di non riuscire a concentrarmi su qualcosa, perché pensieri banali e irrilevanti sopraggiungono a distrarmi."},
    {"id": 2, "titolo": "Perdita dell'ipseità del pensiero", "testo": "Mi è capitato di esperire dei pensieri che mi appaiono anonimi o di una stranezza indescrivibile, tanto da chiedermi se siano davvero miei."},
    {"id": 3, "titolo": "Pressione del pensiero", "testo": "Mi è capitato di sentire i miei pensieri accavallarsi rapidamente nella testa, come se cambiassero in continuazione, senza che io ne abbia il controllo pieno."},
    {"id": 4, "titolo": "Blocco del pensiero", "testo": "Mi capita di perdere il filo dei pensieri."},
    {"id": 5, "titolo": "Eco silente del pensiero", "testo": "Mi capita di sentire i miei pensieri ripetuti in automatico quando li penso."},
    {"id": 6, "titolo": "Ruminazioni-ossessioni", "testo": "Mi capita che mi compaiano in testa dei pensieri o delle immagini, che provo ad allontanare perché fastidiose."},
    {"id": 7, "titolo": "Oggettivazione percettiva del linguaggio interiore", "testo": "Mi capita di vedere i miei pensieri scritti nella mente, come su un foglio di carta o su un cartellone."},
    {"id": 8, "titolo": "Spazializzazione dell'esperienza", "testo": "Mi capita di percepire i miei pensieri in una determinata localizzazione della testa."},
    {"id": 9, "titolo": "Ambivalenza", "testo": "Mi capita di avere difficoltà nel prendere le decisioni, anche per le questioni più banali della vita di tutti i giorni."},
    {"id": 10, "titolo": "Incapacità di discriminare tra i modi dell'intenzionalità", "testo": "Mi capita di pensare, quando faccio qualcosa, se stia accadendo davvero o se invece sia frutto della mia fantasia."},
    {"id": 11, "titolo": "Disturbo dell'iniziativa del pensiero", "testo": "Mi capita di avere fatica nel pensare o nello strutturare il pensiero, come se non fosse qualcosa di automatico."},
    {"id": 12, "titolo": "Disturbi dell'attenzione", "testo": "Mi capita che un particolare dell'ambiente che mi circonda catturi così tanto la mia attenzione da farmi perdere di vista una conversazione o quello che stavo facendo."},
    {"id": 13, "titolo": "Disturbo della memoria a breve termine", "testo": "Mi capita di dimenticare cose che ho appena fatto."},
    {"id": 14, "titolo": "Disturbo dell'esperienza del tempo", "testo": "Mi capita di avere la sensazione che il tempo scorra più lentamente o più velocemente del solito."},
    {"id": 15, "titolo": "Discontinuità del proprio agire", "testo": "Mi capita di non ricordare come è avvenuta una cosa, per esempio di non ricordare come sono arrivato in un posto."},
    {"id": 16, "titolo": "Discordanza tra intenzione ed espressione", "testo": "Mi capita di non riuscire a esprimere quello che vorrei, sia a livello di parole sia di comportamento, come se non avessi un pieno controllo."},
    {"id": 17, "titolo": "Disturbo del linguaggio espressivo", "testo": "Mi capita di avere difficoltà nel dire le cose, come se non le dicessi bene o come vorrei."},
    # 2. Consapevolezza del Se' e Presenza
    {"id": 18, "titolo": "Diminuito senso del Sé di base", "testo": "Mi capita di sentire di aver perso il contatto con il mio vero io, come se avvertissi un vuoto interiore."},
    {"id": 19, "titolo": "Distorta prospettiva in prima persona", "testo": "Mi è capitato di percepire una distanza tra me stesso e quello che faccio o penso."},
    {"id": 20, "titolo": "Altri stati di depersonalizzazione", "testo": "Mi è successo di non sentirmi me stesso, di sentirmi cambiato rispetto a un tempo."},
    {"id": 21, "titolo": "Diminuita presenza", "testo": "Ho la sensazione di essere un po' distaccato emotivamente rispetto a quello che mi succede."},
    {"id": 22, "titolo": "Derealizzazione", "testo": "Mi è capitato di percepire l'ambiente attorno a me diverso, cambiato, persino irreale."},
    {"id": 23, "titolo": "Iperriflessività", "testo": "Mi capita di dover pensare a cosa pensare, come se non avvenisse in maniera automatica."},
    {"id": 24, "titolo": "Scissione dell'Io", "testo": "Mi è successo di sentire il mio io come diviso, come se ci fosse un'altra parte di me."},
    {"id": 25, "titolo": "Depersonalizzazione dissociativa", "testo": "Mi è capitato di avere la sensazione di guardarmi dall'esterno."},
    {"id": 26, "titolo": "Confusione dell'identità", "testo": "Mi è capitato di confondermi con un'altra persona, come se fossi lei."},
    {"id": 27, "titolo": "Cambiamento in relazione all'età", "testo": "Mi capita di sentirmi più giovane o più vecchio rispetto alla mia reale età."},
    {"id": 28, "titolo": "Cambiamento in relazione al genere", "testo": "Mi capita di pensare di poter essere del sesso opposto al mio, o di sperimentare confusione riguardo al mio sesso."},
    {"id": 29, "titolo": "Perdita del senso comune", "testo": "Penso di essere una persona che riflette molto sulle cose, come se mi mancasse un senso comune."},
    {"id": 30, "titolo": "Ansia", "testo": "Penso di essere una persona ansiosa."},
    {"id": 31, "titolo": "Ansia ontologica", "testo": "Mi sento una persona apprensiva, insicura, che fa un po' fatica nelle cose di tutti i giorni."},
    {"id": 32, "titolo": "Diminuita trasparenza della coscienza", "testo": "Mi capita di sentirmi poco sveglio o vigile, come se avessi a volte una sorta di annebbiamento nella testa."},
    {"id": 33, "titolo": "Diminuzione dell'iniziativa", "testo": "Mi sembra di percepire fatica nel fare le cose, nell'iniziare le azioni."},
    {"id": 34, "titolo": "Ipoedonia", "testo": "Non riesco a provare piacere per le cose belle che mi succedono."},
    {"id": 35, "titolo": "Diminuita vitalità", "testo": "Mi sento una persona con poca energia, sia fisica che mentale."},
    # 3. Esperienze corporee
    {"id": 36, "titolo": "Cambiamento morfologico", "testo": "Mi è capitato di percepire sensazioni strane riguardo al corpo, per esempio che certe parti del corpo fossero di dimensioni diverse dal solito."},
    {"id": 37, "titolo": "Fenomeni legati allo specchio", "testo": "Mi è capitato di guardarmi allo specchio per controllare che il mio viso o il mio corpo fossero uguali a prima."},
    {"id": 38, "titolo": "Depersonalizzazione somatica", "testo": "Mi capita di percepire una parte del mio corpo come estranea o isolata dal resto."},
    {"id": 39, "titolo": "Discordanza psicofisica", "testo": "Ho avuto la sensazione di una certa distanza tra la mia mente e il mio corpo, quasi come se non fossero coordinati."},
    {"id": 40, "titolo": "Disintegrazione corporea", "testo": "Ho avuto la sensazione che una parte del mio corpo si stesse dissolvendo o perdendo consistenza."},
    {"id": 41, "titolo": "Spazializzazione delle esperienze corporee", "testo": "Mi è capitato di avvertire fisicamente i miei organi e le loro attività."},
    {"id": 42, "titolo": "Esperienze cenestesiche", "testo": "Mi è capitato di avere delle sensazioni strane riguardo al mio corpo."},
    {"id": 43, "titolo": "Disturbi motori", "testo": "Mi è capitato di produrre involontariamente movimenti o parole che interferiscono con l'azione o il discorso che stavo facendo."},
    {"id": 44, "titolo": "Esperienza mimetica/risonanza del movimento", "testo": "Mi è capitato di sentire come se i movimenti di oggetti o altre persone fossero miei, o comunque connessi ai miei movimenti."},
    # 4. Demarcazione/Transitivismo
    {"id": 45, "titolo": "Confusione con l'altro", "testo": "Mi è capitato di sentirmi come invaso o mischiato agli altri, al punto da non capire se fossi io o loro l'origine dei pensieri e dei sentimenti che provavo."},
    {"id": 46, "titolo": "Confusione con la propria immagine riflessa", "testo": "Ho avuto la sensazione di non essere io mentre mi specchiavo o guardavo la mia immagine in fotografia."},
    {"id": 47, "titolo": "Contatto corporeo vissuto come minaccia", "testo": "Mi sento a disagio, o provo ansia, per il contatto fisico con le altre persone."},
    {"id": 48, "titolo": "Stato d'animo di passività", "testo": "Mi sento come in una condizione di pericolo o vulnerabilità, come in balìa di qualcosa o oppresso da qualcosa."},
    {"id": 49, "titolo": "Altri fenomeni di transitivismo", "testo": "Mi capita di sentirmi come trasparente, come se le altre persone potessero vedermi dentro, leggere dentro di me."},
    # 5. Riorientamento Esistenziale
    {"id": 50, "titolo": "Fenomeni primari di autoriferimento", "testo": "Mi è capitato che ciò che accadeva attorno a me mi sembrasse originare, dipendere o essere in qualche modo riferito a me."},
    {"id": 51, "titolo": "Sentimento di centralità", "testo": "Mi è successo, anche solo per un breve momento, di sentirmi come se fossi al centro del mondo, dell'universo."},
    # 52-54: nessuna "domanda possibile" nel foglio EASE originale, riuso la stessa frase di EASE_Analysis.py
    {"id": 52, "titolo": "Il proprio campo esperienziale come unica realtà", "testo": "Ho la sensazione che solo ciò che percepisco io sia davvero reale, come se il resto non esistesse."},
    {"id": 53, "titolo": "Sentimento di straordinario potere creativo/insight", "testo": "Ho la sensazione di avere un potere creativo straordinario o un accesso privilegiato a dimensioni nascoste della realtà."},
    {"id": 54, "titolo": "Il mondo come apparente o illusorio", "testo": "Ho la sensazione che il mondo che vivo non sia veramente reale, ma solo apparente o illusorio."},
    {"id": 55, "titolo": "Idee magiche", "testo": "Penso che certi eventi possano essere collegati tra loro da una causa non fisica o magica."},
    {"id": 56, "titolo": "Cambiamento esistenziale o intellettuale", "testo": "Mi sono appassionato in modo particolare a temi filosofici, politici o religiosi, e mi trovo spesso a riflettere su tematiche esistenziali come il senso della vita, il bene e il male."},
    {"id": 57, "titolo": "Grandiosità solipsistica", "testo": "Mi sento diverso rispetto alle altre persone, ritengo di essere in qualche modo speciale, e penso di avere qualche capacità particolare."},
]
items = {it["titolo"]: it["testo"] for it in items}
item_keys = list(items.keys())
id_to_title = {i + 1: titolo for i, titolo in enumerate(item_keys)}
hyp_list = list(items.values())
print(f"{len(item_keys)} item EASE (domande cliniche) caricati")

# DOMINI EASE (macro-categorie), identici a EASE_Analysis.py
domains = {
    "Cognitività e flusso della coscienza": list(range(1, 18)),
    "Consapevolezza del Sé e Presenza": list(range(18, 36)),
    "Esperienze corporee": list(range(36, 45)),
    "Demarcazione/Transitivismo": list(range(45, 50)),
    "Riorientamento Esistenziale": list(range(50, 58)),
}
domain_titles = {name: [id_to_title[i] for i in ids] for name, ids in domains.items()}
domain_names = list(domains.keys())
item_to_domain = {t: name for name, titles in domain_titles.items() for t in titles}


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
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    orig_cols = list(df.columns)
    output_rows, buf_text, buf_meta = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"NLI EASE (domande cliniche) su {text_col}"):
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


# CLASSIFICAZIONE df_all
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

item_cols = [t for name in domain_names for t in domain_titles[name] if t in item_keys]

# MACRO-DOMINI EASE a livello di singola metafora
for name, titles in domain_titles.items():
    titles_present = [t for t in titles if t in df_all.columns]
    df_all[name] = df_all[titles_present].mean(axis=1)

df_all["EASE_Mean_Score"] = df_all[item_cols].mean(axis=1)

df_all.to_csv(f"{OUTPUT_DIR}/Metaphor_ALL_NLI_EASE_clinicalQ.csv", index=False)
df_all.to_excel(f"{OUTPUT_DIR}/Metaphor_ALL_NLI_EASE_clinicalQ.xlsx", index=False)

print("df_all EASE (domande cliniche):", df_all.shape)
print(df_all[["File", "Group"] + domain_names + ["EASE_Mean_Score"]].head())

"""## Grafici descrittivi (stessa logica di EASE_Analysis.py)"""

domain_colors = {
    "Cognitività e flusso della coscienza": "#e41a1c",
    "Consapevolezza del Sé e Presenza": "#377eb8",
    "Esperienze corporee": "#4daf4a",
    "Demarcazione/Transitivismo": "#984ea3",
    "Riorientamento Esistenziale": "#ff7f00",
}

# SOGLIA DI RILEVANZA (vedi NLI_Analysis.py/EASE_Analysis.py per la nota sulla
# bimodalita' degli item): proporzione di metafore con score > 0.5, invece della
# media grezza continua
RELEVANCE_THRESHOLD = 0.5
item_cols_rel = [f"{c}__rel" for c in item_cols]
df_all[item_cols_rel] = (df_all[item_cols] > RELEVANCE_THRESHOLD).astype(int)

means_items = pd.Series({c: df_all[f"{c}__rel"].mean() for c in item_cols})
means_domains_dict = {
    name: means_items[[t for t in titles if t in means_items.index]].mean()
    for name, titles in domain_titles.items()
}


def add_domain_brackets(ax, labels, domain_means, x_start):
    groups = []
    for i, lbl in enumerate(labels):
        domain = item_to_domain.get(lbl)
        if groups and groups[-1][0] == domain:
            groups[-1] = (domain, groups[-1][1], i)
        else:
            groups.append((domain, i, i))
    xmax = ax.get_xlim()[1]
    tick = xmax * 0.012
    for domain, i_start, i_end in groups:
        if domain is None or domain not in domain_means:
            continue
        y_top, y_bottom = i_start - 0.42, i_end + 0.42
        color = domain_colors.get(domain, "black")
        ax.plot([x_start, x_start], [y_top, y_bottom], color=color, lw=1.3, clip_on=False)
        ax.plot([x_start, x_start + tick], [y_top, y_top], color=color, lw=1.3, clip_on=False)
        ax.plot([x_start, x_start + tick], [y_bottom, y_bottom], color=color, lw=1.3, clip_on=False)
        ax.text(x_start + tick * 1.8, (y_top + y_bottom) / 2, f"{domain_means[domain]:.2f}",
                va="center", ha="left", fontsize=11, fontweight="bold", color=color)


def make_mean_barplot_ease(means, labels, title, filepath, xlabel="Mean score (0-1)", bar_colors=None, domain_means=None):
    fig, ax = plt.subplots(figsize=(18, max(6, len(labels) * 0.4)))
    y = np.arange(len(labels))
    bars = ax.barh(y, means, color=bar_colors if bar_colors is not None else "#4c72b0")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", ha="left", fontsize=13)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13)
    for lbl in ax.get_yticklabels():
        lbl.set_color(domain_colors.get(item_to_domain.get(lbl.get_text()), "black"))
    ax.invert_yaxis()
    max_val = max(means) if max(means) > 0 else 1
    ax.set_xlim(0, max_val * (1.42 if domain_means is not None else 1.15))
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_title(title, fontsize=18, wrap=True)
    if domain_means is not None:
        add_domain_brackets(ax, labels, domain_means, max_val * 1.22)
    if bar_colors is not None:
        from matplotlib.patches import Patch
        legend_handles = [Patch(facecolor=domain_colors[name], label=name) for name in domain_names]
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, title="Dominio EASE")
    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.show()
    plt.close(fig)


item_bar_colors = [domain_colors.get(item_to_domain.get(lbl), "#4c72b0") for lbl in item_cols]
make_mean_barplot_ease(
    means_items.values, means_items.index.tolist(),
    title=f"Proporzione di metafore rilevanti per item EASE, domande cliniche (score > {RELEVANCE_THRESHOLD}); mean score di dominio tra parentesi",
    filepath=f"{OUTPUT_DIR}/EASE_clinicalQ_barplot_items.png",
    xlabel=f"Proporzione metafore (score > {RELEVANCE_THRESHOLD})",
    bar_colors=item_bar_colors,
    domain_means=means_domains_dict,
)

# BOXPLOT distribuzione item (a livello di singola metafora), ordinato per mediana
medians_ease = df_all[item_cols].median().sort_values(ascending=False)
ordered_cols_ease = medians_ease.index.tolist()
data_ease = [df_all[col].dropna().values for col in ordered_cols_ease]

fig, ax = plt.subplots(figsize=(10, max(6, len(ordered_cols_ease) * 0.3)))
bp = ax.boxplot(
    data_ease, vert=False, patch_artist=True, showfliers=True,
    medianprops={"color": "black", "linewidth": 1.5},
    boxprops={"facecolor": "#4c72b0", "alpha": 0.7},
    flierprops={"markersize": 3, "markerfacecolor": "gray", "markeredgecolor": "none"},
)
ax.set_yticks(np.arange(1, len(ordered_cols_ease) + 1))
ax.set_yticklabels(ordered_cols_ease, fontsize=9)
for lbl in ax.get_yticklabels():
    lbl.set_color(domain_colors.get(item_to_domain.get(lbl.get_text()), "black"))
ax.invert_yaxis()
ax.set_xlabel("Score (0-1)", fontsize=12)
ax.set_title("Item EASE, domande cliniche - distribuzione (a livello di metafora)", fontsize=15, wrap=True)
ax.grid(axis="x", alpha=0.3)
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=domain_colors[name], label=name) for name in domain_names]
ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, title="Dominio EASE")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/EASE_clinicalQ_boxplot_items_distribution.png", dpi=200)
plt.show()
plt.close(fig)

"""## Correlazione EASE (domande cliniche) x IPASE (a livello paziente)

Stessa logica di EASE_Analysis.py: aggregazione a livello paziente dei domini (media
continua, mai binarizzata — vedi nota sulla soglia sopra, si applica solo agli item),
poi merge con output/IPASE_macrocategories_by_patient.csv (richiede che NLI_Analysis.py
sia gia' stato eseguito).
"""

df_patient_ease_clinicalQ = df_all.groupby(["File", "Group"])[domain_names + ["EASE_Mean_Score"]].mean().reset_index()
df_patient_ease_clinicalQ.to_csv(f"{OUTPUT_DIR}/EASE_clinicalQ_patient_avg.csv", index=False)

ipase_patient_path = f"{OUTPUT_DIR}/IPASE_macrocategories_by_patient.csv"
if not os.path.exists(ipase_patient_path):
    raise FileNotFoundError(
        f"{ipase_patient_path} non trovato: esegui prima NLI_Analysis.py (sezione 6.1) "
        "per generare i punteggi IPASE a livello paziente."
    )

df_patient_ipase = pd.read_csv(ipase_patient_path)
ipase_domain_names = ["Cognition", "Self-Awareness and Presence", "Consciousness", "Somatization", "Demarcation/Transitivism"]

df_ease_ipase_clinicalQ = df_patient_ease_clinicalQ[["File", "Group"] + domain_names + ["EASE_Mean_Score"]].merge(
    df_patient_ipase[["File"] + ipase_domain_names + ["IPASE_Mean_Score"]], on="File", how="inner",
)
df_ease_ipase_clinicalQ.to_csv(f"{OUTPUT_DIR}/EASE_clinicalQ_IPASE_patient_merged.csv", index=False)
print(f"\nPazienti con entrambi i punteggi EASE (domande cliniche) e IPASE: {df_ease_ipase_clinicalQ['File'].nunique()}")

ALPHA = 0.05
ease_row_labels = domain_names + ["EASE_Mean_Score"]
ipase_col_labels = ipase_domain_names + ["IPASE_Mean_Score"]


def compute_correlation_ease(df, ipase_col, value_cols):
    sub = df.dropna(subset=[ipase_col])
    rows = []
    for col in value_cols:
        pair = sub[[ipase_col, col]].dropna()
        if len(pair) < 3:
            rows.append({"Variable": col, "r": np.nan, "p_value": np.nan, "N": len(pair), "Significant": False})
            continue
        r, p = spearmanr(pair[ipase_col], pair[col])
        # p_value non arrotondato: con correlazioni forti e N~100 il p reale puo'
        # essere < 1e-30, un round(p,4) lo schiaccerebbe a "p=0.000" (fuorviante)
        rows.append({"Variable": col, "r": round(r, 3), "p_value": p, "N": len(pair), "Significant": p < ALPHA})
    return pd.DataFrame(rows)


def apply_bonferroni_ease(corr_tables, alpha=ALPHA):
    """Correzione di Bonferroni: famiglia = tutte le celle della griglia (domini EASE x
    domini IPASE mostrati insieme). Sovrascrive 'Significant' con la soglia corretta."""
    n_tests = sum(len(df) for df in corr_tables.values())
    alpha_corrected = alpha / n_tests if n_tests > 0 else alpha
    for df in corr_tables.values():
        df["Significant"] = df["p_value"] < alpha_corrected
    return alpha_corrected


ease_ipase_corr_tables_clinicalQ = {}
for ipase_col in ipase_col_labels:
    ease_ipase_corr_tables_clinicalQ[ipase_col] = compute_correlation_ease(df_ease_ipase_clinicalQ, ipase_col, ease_row_labels)
alpha_bonf = apply_bonferroni_ease(ease_ipase_corr_tables_clinicalQ)
print(f"Bonferroni EASE (domande cliniche) x IPASE: alpha corretto = {alpha_bonf:.2e} "
      f"(n_test={len(ease_row_labels) * len(ipase_col_labels)})")

for ipase_col, df_corr in ease_ipase_corr_tables_clinicalQ.items():
    safe_ipase_col = re.sub(r"[^A-Za-z0-9]+", "_", ipase_col).strip("_")
    df_corr.to_csv(f"{OUTPUT_DIR}/correlation_EASE_clinicalQ_domains_vs_{safe_ipase_col}.csv", index=False)


def plot_ease_ipase_grid(df_corr_dict, row_labels, title, filepath, vmin=-1, vmax=1):
    col_labels = list(df_corr_dict.keys())
    n_rows, n_cols = len(row_labels), len(col_labels)

    r_matrix = np.full((n_rows, n_cols), np.nan)
    p_matrix = np.full((n_rows, n_cols), np.nan)
    sig_matrix = np.zeros((n_rows, n_cols), dtype=bool)

    for j, col_label in enumerate(col_labels):
        d = df_corr_dict[col_label].set_index("Variable").reindex(row_labels)
        r_matrix[:, j] = d["r"].values
        p_matrix[:, j] = d["p_value"].values
        sig_matrix[:, j] = d["Significant"].fillna(False).values

    display_vals = np.where(sig_matrix, r_matrix, np.nan)

    fig, ax = plt.subplots(figsize=(3 * n_cols + 3, max(6, n_rows * 0.6)))
    cmap = plt.cm.RdBu_r
    cmap.set_bad(color="#e8e8e8")
    im = ax.imshow(display_vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            r, p = r_matrix[i, j], p_matrix[i, j]
            if np.isnan(r):
                text, color = "n/a", "black"
            else:
                p_label = f"{p:.1e}" if p < 0.001 else f"{p:.3f}"
                text = f"r={r:.2f}\np={p_label}"
                is_dark = sig_matrix[i, j] and (abs(r) / (vmax - vmin) > 0.25)
                color = "white" if is_dark else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=10,
                    color=color, fontweight="bold" if sig_matrix[i, j] else "normal")

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=11)
    for lbl in ax.get_yticklabels():
        lbl.set_color(domain_colors.get(lbl.get_text(), "black"))
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_title(title, fontsize=16, wrap=True, pad=30)
    ax.set_ylabel("Domini EASE (domande cliniche)", fontsize=12)
    ax.text(0.5, -0.12, "Domini IPASE", ha="center", va="center", fontsize=12, transform=ax.transAxes)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("r (p < 0.05)", fontsize=10)

    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.show()
    plt.close(fig)


plot_ease_ipase_grid(
    ease_ipase_corr_tables_clinicalQ, ease_row_labels,
    title="Correlazione domini EASE (domande cliniche) x domini IPASE (Spearman, a livello paziente)",
    filepath=f"{OUTPUT_DIR}/correlation_grid_EASE_clinicalQ_vs_IPASE.png",
)

print("\nOK - correlazione EASE (domande cliniche) x IPASE completata")

print("\nOK - grafici EASE (domande cliniche) generati")
