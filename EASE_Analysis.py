# -*- coding: utf-8 -*-
"""
Scoring delle metafore (Schizo_Metaphors.xlsx) sui 57 item della scala EASE
(Examination of Anomalous Self-Experience), 5 domini:
1. Cognitivita' e flusso della coscienza (17 item)
2. Consapevolezza del Se' e Presenza (18 item)
3. Esperienze corporee (9 item)
4. Demarcazione/Transitivismo (5 item)
5. Riorientamento Esistenziale (8 item)

Stessa logica di classificazione NLI usata per l'IPASE (NLI_Analysis.py):
ogni metafora (premise) viene testata contro ogni item EASE (hypothesis, frase in
prima persona) con un modello di Natural Language Inference; lo score e' la
probabilita' di entailment.

Oltre allo scoring, lo script produce:
- descrittive degli score EASE (per item e per dominio, a livello paziente e di
  singola metafora) con barplot/boxplot;
- correlazione (Spearman) tra i domini EASE e i domini IPASE, a livello paziente
  (richiede che NLI_Analysis.py sia gia' stato eseguito almeno fino alla sezione
  6.1, cosi' da avere output/IPASE_macrocategories_by_patient.csv).
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

# SCALA EASE — 57 item (titolo: testo in prima persona, costruito dalla definizione
# clinica e dalle domande d'intervista del foglio EASE originale)
items = [
    # 1. Cognitivita' e flusso della coscienza
    {"id": 1, "titolo": "Interferenza del pensiero", "testo": "Mi capita che pensieri banali e irrilevanti irrompano improvvisamente interrompendo il filo dei miei pensieri."},
    {"id": 2, "titolo": "Perdita dell'ipseità del pensiero", "testo": "Ho la sensazione che alcuni miei pensieri siano anonimi o stranamente indescrivibili, come se non fossero generati da me."},
    {"id": 3, "titolo": "Pressione del pensiero", "testo": "Ho la sensazione che troppi pensieri diversi si accavallino rapidamente nella mia mente, senza che io riesca a controllarli."},
    {"id": 4, "titolo": "Blocco del pensiero", "testo": "Mi capita che i miei pensieri si blocchino o svaniscano improvvisamente, come se venissero tagliati via."},
    {"id": 5, "titolo": "Eco silente del pensiero", "testo": "Ho la sensazione che i miei pensieri vengano ripetuti automaticamente non appena li penso."},
    {"id": 6, "titolo": "Ruminazioni-ossessioni", "testo": "Mi capita che certi pensieri o immagini fastidiosi continuino a ripresentarsi nella mia mente, senza che io riesca ad allontanarli."},
    {"id": 7, "titolo": "Oggettivazione percettiva del linguaggio interiore", "testo": "Ho la sensazione di sentire i miei pensieri ad alta voce, o di vederli scritti nella mente, e temo che anche gli altri possano percepirli."},
    {"id": 8, "titolo": "Spazializzazione dell'esperienza", "testo": "Ho la sensazione di percepire fisicamente i miei pensieri in una precisa zona della testa."},
    {"id": 9, "titolo": "Ambivalenza", "testo": "Ho una persistente difficoltà a decidere anche di fronte a scelte banali della vita quotidiana."},
    {"id": 10, "titolo": "Incapacità di discriminare tra i modi dell'intenzionalità", "testo": "Mi capita di non essere sicuro se qualcosa sia davvero accaduto o sia stato solo frutto della mia immaginazione o di un ricordo."},
    {"id": 11, "titolo": "Disturbo dell'iniziativa del pensiero", "testo": "Faccio fatica a pensare o a organizzare i miei pensieri, come se non fosse più un processo automatico."},
    {"id": 12, "titolo": "Disturbi dell'attenzione", "testo": "Mi capita che un dettaglio dell'ambiente catturi così tanto la mia attenzione da farmi perdere il filo di una conversazione o di ciò che stavo facendo."},
    {"id": 13, "titolo": "Disturbo della memoria a breve termine", "testo": "Mi capita di dimenticare rapidamente cose che ho appena fatto o detto."},
    {"id": 14, "titolo": "Disturbo dell'esperienza del tempo", "testo": "Ho la sensazione che il tempo scorra in modo anomalo, più lento, più veloce o addirittura fermo."},
    {"id": 15, "titolo": "Discontinuità del proprio agire", "testo": "Mi capita di non ricordare come sono arrivato a fare o a trovarmi in una certa situazione."},
    {"id": 16, "titolo": "Discordanza tra intenzione ed espressione", "testo": "Ho la sensazione di non riuscire a esprimere davvero ciò che vorrei, né a parole né con i miei comportamenti."},
    {"id": 17, "titolo": "Disturbo del linguaggio espressivo", "testo": "Faccio fatica a trovare le parole giuste per dire quello che penso o provo."},
    # 2. Consapevolezza del Se' e Presenza
    {"id": 18, "titolo": "Diminuito senso del Sé di base", "testo": "Ho un senso pervasivo di vuoto interiore, come se non avessi una vera identità o fossi profondamente diverso dagli altri."},
    {"id": 19, "titolo": "Distorta prospettiva in prima persona", "testo": "Ho la sensazione di essere distante da me stesso, come se osservassi costantemente i miei pensieri e le mie azioni dall'esterno invece di viverli direttamente."},
    {"id": 20, "titolo": "Altri stati di depersonalizzazione", "testo": "Mi sento estraneo a me stesso, come se non fossi più la stessa persona di prima."},
    {"id": 21, "titolo": "Diminuita presenza", "testo": "Anche quando mi capita qualcosa di bello, resto emotivamente freddo e distaccato."},
    {"id": 22, "titolo": "Derealizzazione", "testo": "L'ambiente intorno a me mi sembra a volte trasformato, irreale o estraneo."},
    {"id": 23, "titolo": "Iperriflessività", "testo": "Devo continuamente riflettere su cosa pensare o come comportarmi, come se nulla avvenisse più in modo spontaneo."},
    {"id": 24, "titolo": "Scissione dell'Io", "testo": "Ho la sensazione che la mia personalità sia divisa in più parti, come se non fossi un tutto unico."},
    {"id": 25, "titolo": "Depersonalizzazione dissociativa", "testo": "Ho la sensazione di guardarmi dall'esterno, come se fossi un doppio che osserva se stesso."},
    {"id": 26, "titolo": "Confusione dell'identità", "testo": "Mi capita di confondermi con un'altra persona, come se fossi lei."},
    {"id": 27, "titolo": "Cambiamento in relazione all'età", "testo": "Mi sento più giovane o più vecchio rispetto alla mia reale età."},
    {"id": 28, "titolo": "Cambiamento in relazione al genere", "testo": "Provo confusione riguardo al mio genere, o penso di poter essere del sesso opposto al mio."},
    {"id": 29, "titolo": "Perdita del senso comune", "testo": "Mi manca una comprensione automatica e naturale delle cose, e mi ritrovo a riflettere e interrogarmi continuamente anche su questioni banali."},
    {"id": 30, "titolo": "Ansia", "testo": "Sono una persona ansiosa."},
    {"id": 31, "titolo": "Ansia ontologica", "testo": "Provo un senso pervasivo di insicurezza e vulnerabilità, come se un pericolo indefinito incombesse su di me."},
    {"id": 32, "titolo": "Diminuita trasparenza della coscienza", "testo": "Mi sento come annebbiato nella testa, come se non fossi mai pienamente sveglio o lucido."},
    {"id": 33, "titolo": "Diminuzione dell'iniziativa", "testo": "Faccio fatica a iniziare qualsiasi attività, come se ogni azione richiedesse uno sforzo eccessivo."},
    {"id": 34, "titolo": "Ipoedonia", "testo": "Non riesco più a provare piacere per le cose che prima mi piacevano."},
    {"id": 35, "titolo": "Diminuita vitalità", "testo": "Mi sento privo di energia, sia fisica che mentale, senza una vera ragione."},
    # 3. Esperienze corporee
    {"id": 36, "titolo": "Cambiamento morfologico", "testo": "Ho la sensazione che alcune parti del mio corpo abbiano dimensioni diverse dal solito, più piccole, più grandi o schiacciate."},
    {"id": 37, "titolo": "Fenomeni legati allo specchio", "testo": "Mi guardo spesso allo specchio per controllare se il mio aspetto è cambiato, o addirittura per assicurarmi di esistere davvero."},
    {"id": 38, "titolo": "Depersonalizzazione somatica", "testo": "Percepisco una parte del mio corpo come estranea, isolata dal resto o come se non fosse davvero mia."},
    {"id": 39, "titolo": "Discordanza psicofisica", "testo": "Sento una distanza tra la mia mente e il mio corpo, come se fossero due entità scollegate."},
    {"id": 40, "titolo": "Disintegrazione corporea", "testo": "Ho la sensazione che una parte del mio corpo si stia dissolvendo o perdendo consistenza."},
    {"id": 41, "titolo": "Spazializzazione delle esperienze corporee", "testo": "Riesco ad avvertire fisicamente i miei organi interni e le loro attività."},
    {"id": 42, "titolo": "Esperienze cenestesiche", "testo": "Ho sensazioni fisiche strane e insolite nel corpo, come scosse elettriche, pressioni o cambiamenti di temperatura senza causa apparente."},
    {"id": 43, "titolo": "Disturbi motori", "testo": "Mi capita di percepire movimenti nel mio corpo che gli altri non vedono, o di bloccarmi improvvisamente durante un'azione."},
    {"id": 44, "titolo": "Esperienza mimetica/risonanza del movimento", "testo": "Ho la sensazione che i movimenti di altre persone o oggetti siano in qualche modo collegati ai miei movimenti."},
    # 4. Demarcazione/Transitivismo
    {"id": 45, "titolo": "Confusione con l'altro", "testo": "Mi sento come invaso o mischiato con gli altri, al punto da non capire più se i pensieri e i sentimenti che provo siano davvero miei."},
    {"id": 46, "titolo": "Confusione con la propria immagine riflessa", "testo": "Mi capita di non riconoscermi nella mia immagine riflessa allo specchio o in una fotografia, come se non fossi io."},
    {"id": 47, "titolo": "Contatto corporeo vissuto come minaccia", "testo": "Il contatto fisico con un'altra persona mi provoca ansia estrema, o mi fa sentire come se stessi scomparendo."},
    {"id": 48, "titolo": "Stato d'animo di passività", "testo": "Mi sento oppresso da qualcosa di indefinito e minaccioso, come se fossi in balìa di un pericolo che non so descrivere."},
    {"id": 49, "titolo": "Altri fenomeni di transitivismo", "testo": "Mi sento come trasparente, come se gli altri potessero leggere dentro di me senza che ci sia una vera barriera tra noi."},
    # 5. Riorientamento Esistenziale
    {"id": 50, "titolo": "Fenomeni primari di autoriferimento", "testo": "Ho la sensazione che ciò che accade intorno a me sia in qualche modo originato da me o riferito direttamente a me."},
    {"id": 51, "titolo": "Sentimento di centralità", "testo": "Ho la sensazione, anche solo per un momento, di essere al centro dell'universo."},
    {"id": 52, "titolo": "Il proprio campo esperienziale come unica realtà", "testo": "Ho la sensazione che solo ciò che percepisco io sia davvero reale, come se il resto non esistesse."},
    {"id": 53, "titolo": "Sentimento di straordinario potere creativo/insight", "testo": "Ho la sensazione di avere un potere creativo straordinario o un accesso privilegiato a dimensioni nascoste della realtà."},
    {"id": 54, "titolo": "Il mondo come apparente o illusorio", "testo": "Ho la sensazione che il mondo che vivo non sia veramente reale, ma solo apparente o illusorio."},
    {"id": 55, "titolo": "Idee magiche", "testo": "Penso che certi eventi possano essere collegati tra loro da una causa non fisica o magica."},
    {"id": 56, "titolo": "Cambiamento esistenziale o intellettuale", "testo": "Mi ritrovo spesso a riflettere in modo nuovo e intenso su temi esistenziali, filosofici o religiosi."},
    {"id": 57, "titolo": "Grandiosità solipsistica", "testo": "Penso di possedere conoscenze o capacità straordinarie che mi rendono superiore agli altri."},
]
items = {it["titolo"]: it["testo"] for it in items}
item_keys = list(items.keys())
id_to_title = {i + 1: titolo for i, titolo in enumerate(item_keys)}
hyp_list = list(items.values())
print(f"{len(item_keys)} item EASE caricati")

# DOMINI EASE (macro-categorie), analoghi alle subscale IPASE
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

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"NLI EASE su {text_col}"):
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

# SOGLIA DI RILEVANZA: come per l'IPASE, la distribuzione dello score NLI per singolo
# item e' bimodale (maggioranza di metafore irrilevanti, score ~0, coda di metafore
# fortemente pertinenti, score ~1). Ovunque si aggreghi uno score di ITEM (non di
# dominio, gia' una media su piu' item e quindi meno bimodale) si usa la proporzione di
# metafore con score > RELEVANCE_THRESHOLD invece della media grezza continua.
RELEVANCE_THRESHOLD = 0.5
item_cols_rel = [f"{c}__rel" for c in item_cols]
df_all[item_cols_rel] = (df_all[item_cols] > RELEVANCE_THRESHOLD).astype(int)

# MACRO-DOMINI EASE a livello di singola metafora
for name, titles in domain_titles.items():
    titles_present = [t for t in titles if t in df_all.columns]
    df_all[name] = df_all[titles_present].mean(axis=1)

df_all["EASE_Mean_Score"] = df_all[item_cols].mean(axis=1)

df_all.to_csv(f"{OUTPUT_DIR}/Metaphor_ALL_NLI_EASE.csv", index=False)
df_all.to_excel(f"{OUTPUT_DIR}/Metaphor_ALL_NLI_EASE.xlsx", index=False)

print("df_all EASE:", df_all.shape)
print(df_all[["File", "Group"] + domain_names + ["EASE_Mean_Score"]].head())

domain_colors = {
    "Cognitività e flusso della coscienza": "#e41a1c",
    "Consapevolezza del Sé e Presenza": "#377eb8",
    "Esperienze corporee": "#4daf4a",
    "Demarcazione/Transitivismo": "#984ea3",
    "Riorientamento Esistenziale": "#ff7f00",
}
item_to_domain = {t: name for name, titles in domain_titles.items() for t in titles}

# AGGREGAZIONE A LIVELLO PAZIENTE: item come proporzione di metafore del paziente con
# score > RELEVANCE_THRESHOLD; domini e punteggio medio complessivo restano una media
# continua (sono gia' composti su piu' item, meno bimodali, vedi nota sopra)
df_patient_ease_items = (
    df_all.groupby(["File", "Group"])[item_cols_rel]
    .mean()
    .rename(columns=dict(zip(item_cols_rel, item_cols)))
)
df_patient_ease_domains = df_all.groupby(["File", "Group"])[domain_names + ["EASE_Mean_Score"]].mean()
df_patient_ease = df_patient_ease_items.join(df_patient_ease_domains).reset_index()
df_patient_ease.to_csv(f"{OUTPUT_DIR}/EASE_patient_avg.csv", index=False)
print("EASE - rating medi per paziente:", df_patient_ease.shape)

# VERSIONE SENZA SOGLIA: media grezza continua per gli item, invece della proporzione
df_patient_ease_raw = (
    df_all.groupby(["File", "Group"])[item_cols + domain_names + ["EASE_Mean_Score"]]
    .mean()
    .reset_index()
)
df_patient_ease_raw.to_csv(f"{OUTPUT_DIR}/EASE_patient_avg_raw.csv", index=False)
print("EASE - rating medi per paziente, senza soglia:", df_patient_ease_raw.shape)

"""## Descrittive degli score EASE"""

# DESCRITTIVE: item, proporzione di metafore rilevanti (su tutte le metafore, nessuna
# aggregazione per paziente), ordinate per media (= proporzione) decrescente
ease_item_stats = df_all[item_cols_rel].agg(["mean", "median", "std", "min", "max"]).T
ease_item_stats = ease_item_stats.rename(index=dict(zip(item_cols_rel, item_cols)))
ease_item_stats["domain"] = ease_item_stats.index.map(item_to_domain)
ease_item_stats = ease_item_stats.sort_values("mean", ascending=False)
ease_item_stats.to_csv(f"{OUTPUT_DIR}/EASE_descriptive_items.csv")
print("\nDescrittive item EASE (top 10 per media):")
print(ease_item_stats.head(10))

# VERSIONE SENZA SOGLIA: descrittive su media grezza continua
ease_item_stats_raw = df_all[item_cols].agg(["mean", "median", "std", "min", "max"]).T
ease_item_stats_raw["domain"] = ease_item_stats_raw.index.map(item_to_domain)
ease_item_stats_raw = ease_item_stats_raw.sort_values("mean", ascending=False)
ease_item_stats_raw.to_csv(f"{OUTPUT_DIR}/EASE_descriptive_items_raw.csv")

# DESCRITTIVE: domini (+ punteggio medio complessivo), su tutte le metafore
ease_domain_stats = df_all[domain_names + ["EASE_Mean_Score"]].agg(
    ["mean", "median", "std", "min", "max"]
).T
ease_domain_stats.to_csv(f"{OUTPUT_DIR}/EASE_descriptive_domains.csv")
print("\nDescrittive domini EASE:")
print(ease_domain_stats)


def add_domain_brackets(ax, labels, domain_means, x_start):
    """Disegna, a destra delle barre, una parentesi quadra per ogni gruppo contiguo di
    item dello stesso dominio EASE, con accanto il mean score del dominio (cosi' non
    serve un secondo grafico solo per i domini)."""
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
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.2f}",
                va="center", ha="left", fontsize=13)

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


means_items_ease = pd.Series({c: df_all[f"{c}__rel"].mean() for c in item_cols})

# la parentesi di dominio nel barplot mostra la media delle proporzioni dei suoi item
# (coerente con la definizione usata a livello paziente in df_patient_ease), non la
# media del composito continuo per-metafora
means_domains_dict_ease = {
    name: means_items_ease[[t for t in titles if t in means_items_ease.index]].mean()
    for name, titles in domain_titles.items()
}

item_bar_colors = [domain_colors.get(item_to_domain.get(lbl), "#4c72b0") for lbl in item_cols]
make_mean_barplot_ease(
    means_items_ease.values, means_items_ease.index.tolist(),
    title=f"Proporzione di metafore rilevanti per item EASE (score > {RELEVANCE_THRESHOLD}); mean score di dominio tra parentesi",
    filepath=f"{OUTPUT_DIR}/EASE_barplot_items.png",
    xlabel=f"Proporzione metafore (score > {RELEVANCE_THRESHOLD})",
    bar_colors=item_bar_colors,
    domain_means=means_domains_dict_ease,
)

# VERSIONE SENZA SOGLIA: media grezza continua su tutte le metafore
means_items_ease_raw = df_all[item_cols].mean().reindex(item_cols)
means_domains_dict_ease_raw = {
    name: means_items_ease_raw[[t for t in titles if t in means_items_ease_raw.index]].mean()
    for name, titles in domain_titles.items()
}
make_mean_barplot_ease(
    means_items_ease_raw.values, means_items_ease_raw.index.tolist(),
    title="Mean score per item EASE, senza soglia (media grezza su tutte le metafore); mean score di dominio tra parentesi",
    filepath=f"{OUTPUT_DIR}/EASE_barplot_items_raw.png",
    xlabel="Mean score (0-1)",
    bar_colors=item_bar_colors,
    domain_means=means_domains_dict_ease_raw,
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
ax.set_title("Item EASE - distribuzione (a livello di metafora)", fontsize=15, wrap=True)
ax.grid(axis="x", alpha=0.3)
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=domain_colors[name], label=name) for name in domain_names]
ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9, title="Dominio EASE")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/EASE_boxplot_items_distribution.png", dpi=200)
plt.show()
plt.close(fig)

"""## Correlazione EASE x IPASE (a livello paziente)

Richiede che NLI_Analysis.py sia gia' stato eseguito (usa
output/IPASE_macrocategories_by_patient.csv, prodotto dalla sezione 6.1).
"""

ipase_patient_path = f"{OUTPUT_DIR}/IPASE_macrocategories_by_patient.csv"
if not os.path.exists(ipase_patient_path):
    raise FileNotFoundError(
        f"{ipase_patient_path} non trovato: esegui prima NLI_Analysis.py (sezione 6.1) "
        "per generare i punteggi IPASE a livello paziente."
    )

df_patient_ipase = pd.read_csv(ipase_patient_path)
ipase_domain_names = ["Cognition", "Self-Awareness and Presence", "Consciousness", "Somatization", "Demarcation/Transitivism"]

df_ease_ipase = df_patient_ease[["File", "Group"] + domain_names + ["EASE_Mean_Score"]].merge(
    df_patient_ipase[["File"] + ipase_domain_names + ["IPASE_Mean_Score"]], on="File", how="inner",
)
df_ease_ipase.to_csv(f"{OUTPUT_DIR}/EASE_IPASE_patient_merged.csv", index=False)
print(f"\nPazienti con entrambi i punteggi EASE e IPASE: {df_ease_ipase['File'].nunique()}")

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
        # p_value non arrotondato a poche cifre: con correlazioni forti e N~100 il
        # p-value reale puo' essere anche < 1e-30, e un round(p, 4) lo schiaccerebbe
        # a 0.0 mostrando "p=0.000" (fuorviante, sembra un bug ma non lo e').
        rows.append({"Variable": col, "r": round(r, 3), "p_value": p, "N": len(pair), "Significant": p < ALPHA})
    return pd.DataFrame(rows)


def apply_bonferroni_ease(corr_tables, alpha=ALPHA):
    """Correzione di Bonferroni per confronti multipli: la famiglia di test e' l'intera
    griglia (tutte le celle domini EASE x domini IPASE mostrate insieme). Sovrascrive
    'Significant' con la soglia corretta alpha/n_test; ritorna la soglia corretta."""
    n_tests = sum(len(df) for df in corr_tables.values())
    alpha_corrected = alpha / n_tests if n_tests > 0 else alpha
    for df in corr_tables.values():
        df["Significant"] = df["p_value"] < alpha_corrected
    return alpha_corrected


ease_ipase_corr_tables = {}
for ipase_col in ipase_col_labels:
    ease_ipase_corr_tables[ipase_col] = compute_correlation_ease(df_ease_ipase, ipase_col, ease_row_labels)
alpha_bonf_ease = apply_bonferroni_ease(ease_ipase_corr_tables)
print(f"Bonferroni EASE x IPASE: alpha corretto = {alpha_bonf_ease:.2e} "
      f"(n_test={len(ease_row_labels) * len(ipase_col_labels)})")

for ipase_col, df_corr in ease_ipase_corr_tables.items():
    safe_ipase_col = re.sub(r"[^A-Za-z0-9]+", "_", ipase_col).strip("_")
    df_corr.to_csv(f"{OUTPUT_DIR}/correlation_EASE_domains_vs_{safe_ipase_col}.csv", index=False)


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
    ax.set_ylabel("Domini EASE", fontsize=12)
    ax.text(0.5, -0.12, "Domini IPASE", ha="center", va="center", fontsize=12, transform=ax.transAxes)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("r (p < 0.05)", fontsize=10)

    plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.show()
    plt.close(fig)


plot_ease_ipase_grid(
    ease_ipase_corr_tables, ease_row_labels,
    title="Correlazione domini EASE x domini IPASE (Spearman, a livello paziente)",
    filepath=f"{OUTPUT_DIR}/correlation_grid_EASE_vs_IPASE.png",
)

"""### Stessa correlazione, versione senza soglia (media grezza continua)"""

ipase_patient_path_raw = f"{OUTPUT_DIR}/IPASE_macrocategories_by_patient_raw.csv"
if not os.path.exists(ipase_patient_path_raw):
    raise FileNotFoundError(
        f"{ipase_patient_path_raw} non trovato: esegui prima NLI_Analysis.py (sezione 6.1) "
        "per generare i punteggi IPASE a livello paziente senza soglia."
    )

df_patient_ipase_raw = pd.read_csv(ipase_patient_path_raw)
df_ease_ipase_raw = df_patient_ease_raw[["File", "Group"] + domain_names + ["EASE_Mean_Score"]].merge(
    df_patient_ipase_raw[["File"] + ipase_domain_names + ["IPASE_Mean_Score"]], on="File", how="inner",
)
df_ease_ipase_raw.to_csv(f"{OUTPUT_DIR}/EASE_IPASE_patient_merged_raw.csv", index=False)

ease_ipase_corr_tables_raw = {}
for ipase_col in ipase_col_labels:
    ease_ipase_corr_tables_raw[ipase_col] = compute_correlation_ease(df_ease_ipase_raw, ipase_col, ease_row_labels)
alpha_bonf_ease_raw = apply_bonferroni_ease(ease_ipase_corr_tables_raw)
print(f"Bonferroni EASE x IPASE, senza soglia: alpha corretto = {alpha_bonf_ease_raw:.2e} "
      f"(n_test={len(ease_row_labels) * len(ipase_col_labels)})")

for ipase_col, df_corr in ease_ipase_corr_tables_raw.items():
    safe_ipase_col = re.sub(r"[^A-Za-z0-9]+", "_", ipase_col).strip("_")
    df_corr.to_csv(f"{OUTPUT_DIR}/correlation_EASE_domains_vs_{safe_ipase_col}_raw.csv", index=False)

plot_ease_ipase_grid(
    ease_ipase_corr_tables_raw, ease_row_labels,
    title="Correlazione domini EASE x domini IPASE, senza soglia (Spearman, a livello paziente)",
    filepath=f"{OUTPUT_DIR}/correlation_grid_EASE_vs_IPASE_raw.png",
)
