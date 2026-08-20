"""
Genera le coppie item IPASE + frase metaforica da far validare ai clinici,
a partire dagli score NLI gia' calcolati in output/Metaphor_ALL_NLI_IPASE.csv.

Disegno:
- Pool iniziale di N_TOTAL_INIT coppie (item, metafora), campionate stratificando
  per score NLI (5 bin da 0.0 a 1.0) cosi' il range di punteggi coperto e' ampio
  e la correlazione punteggio-modello vs punteggio-clinico ha varianza sufficiente.
- Una quota CORE (OVERLAP_FRAC, di default 20%) viene annotata da TUTTI gli
  annotatori (presenti e futuri) -> serve per calcolare l'accordo inter-rater
  (es. ICC, kappa pesato).
- Il resto (pool "personal") viene diviso in blocchi esclusivi, uno per
  annotatore.
- Un ledger (annotation/output/ledger_master_pairs.csv) tiene traccia di tutte
  le coppie gia' distribuite, cosi' i comandi sono idempotenti e ripetibili.

USO

  Creazione batch iniziale per due annotatori:
    python build_annotation_batches.py init Annotatore1 Annotatore2

  Aggiungere un annotatore in futuro (riceve lo stesso CORE + un nuovo
  blocco personal, mai visto da nessun altro annotatore):
    python build_annotation_batches.py add-annotator Annotatore3

  Le opzioni --n-total, --overlap, --n-personal, --seed permettono di
  cambiare i default (vedi --help).
"""

import argparse
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_CSV = os.path.join(BASE_DIR, "..", "output", "Metaphor_ALL_NLI_IPASE.csv")
OUT_DIR = os.path.join(BASE_DIR, "output")
LEDGER_PATH = os.path.join(OUT_DIR, "ledger_master_pairs.csv")

SEED = 42
N_TOTAL_INIT = 400
OVERLAP_FRAC = 0.20
SCORE_BIN_EDGES = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
SCORE_BIN_LABELS = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

ANNOTATOR_CSV_COLUMNS = [
    "ID_coppia",
    "ID_paziente_anon",
    "Item_ID",
    "Item_Testo",
    "Frase_metafora",
    "Punteggio_{annot}",
    "Note",
]

# SCALA IPASE — 57 item (titolo: testo), identica a NLI_Analysis.py
IPASE_ITEMS = [
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
ITEM_BY_TITLE = {it["titolo"]: it for it in IPASE_ITEMS}

# righe di calibrazione fisse, identiche per tutti gli annotatori (dal manuale, §5.1-5.5)
CALIBRATION_ROWS = [
    {"ID_coppia": "ESEMPIO-1", "ID_paziente_anon": "—", "Item_ID": 1,
     "Item_Testo": "Ho come la sensazione che i miei pensieri siano generati da qualcun altro.",
     "Frase_metafora": "Era come se dei viaggiatori misteriosi bussassero alla porta della mia coscienza senza annunciarsi.",
     "Note": "Esempio di calibrazione — punteggio atteso: 4 (vedi manuale §5.1)"},
    {"ID_coppia": "ESEMPIO-2", "ID_paziente_anon": "—", "Item_ID": 38,
     "Item_Testo": "Sento di non essere più la stessa persona che sono sempre stato/a.",
     "Frase_metafora": "Quindi io in questi giorni sento che sta avvenendo una trasformazione da carpa koi, quindi un me addomesticato che si fa fare tutto quello che deve fare.",
     "Note": "Esempio di calibrazione — punteggio atteso: 3 (vedi manuale §5.2)"},
    {"ID_coppia": "ESEMPIO-3", "ID_paziente_anon": "—", "Item_ID": 46,
     "Item_Testo": "È come se stia svanendo dall'esistenza.",
     "Frase_metafora": "È un incubo senza risveglio.",
     "Note": "Esempio di calibrazione — punteggio atteso: 2 (vedi manuale §5.3)"},
    {"ID_coppia": "ESEMPIO-4", "ID_paziente_anon": "—", "Item_ID": 28,
     "Item_Testo": "Quando penso, è come se i miei pensieri venissero messi per scritto.",
     "Frase_metafora": "sentivo anche la radio in testa",
     "Note": "Esempio di calibrazione — punteggio atteso: 1 (vedi manuale §5.4)"},
    {"ID_coppia": "ESEMPIO-5", "ID_paziente_anon": "—", "Item_ID": 19,
     "Item_Testo": "Spesso mi guardo allo specchio per vedere se sono cambiato/a.",
     "Frase_metafora": "Il vento portava voci",
     "Note": "Esempio di calibrazione — punteggio atteso: 0 (vedi manuale §5.5)"},
]


def load_long_pool():
    """Trasforma Metaphor_ALL_NLI_IPASE.csv (wide, 1 riga = 1 metafora) in
    formato long (1 riga = 1 coppia item-metafora) con lo score NLI relativo."""
    df = pd.read_csv(SRC_CSV).reset_index(drop=True)
    df["Metaphor_row_id"] = df.index

    item_titles = [c for c in df.columns if c in ITEM_BY_TITLE]
    assert len(item_titles) == 57, f"attesi 57 item IPASE, trovati {len(item_titles)}"

    id_vars = ["Metaphor_row_id", "Metaphor_Text", "File", "Group", "TimePoint",
               "TARGET", "SOURCE1", "SOURCE2", "vehicle1", "vehicle2", "vehicle3"]
    long = df.melt(id_vars=id_vars, value_vars=item_titles,
                    var_name="Item_Titolo", value_name="NLI_score")

    long["Item_ID"] = long["Item_Titolo"].map(lambda t: ITEM_BY_TITLE[t]["id"])
    long["Item_Testo"] = long["Item_Titolo"].map(lambda t: ITEM_BY_TITLE[t]["testo"])
    long["ID_paziente_anon"] = long["File"]
    long["Frase_metafora"] = long["Metaphor_Text"]
    long["Pair_key"] = long["Metaphor_row_id"].astype(str) + "__" + long["Item_ID"].astype(str)
    long["Score_bin"] = pd.cut(long["NLI_score"], bins=SCORE_BIN_EDGES, right=False,
                                labels=SCORE_BIN_LABELS)
    return long


def stratified_sample(pool, n, rng):
    """Campiona n righe da pool, stratificando per Score_bin (numerosita' pari
    per bin, con fallback casuale se un bin non basta a coprire la quota)."""
    per_bin = n // len(SCORE_BIN_LABELS)
    parts = []
    for lbl in SCORE_BIN_LABELS:
        sub = pool[pool["Score_bin"] == lbl]
        k = min(per_bin, len(sub))
        if k > 0:
            parts.append(sub.sample(n=k, random_state=int(rng.integers(0, 1_000_000))))
    sample = pd.concat(parts) if parts else pool.iloc[0:0]

    remaining = n - len(sample)
    if remaining > 0:
        rest = pool.drop(sample.index)
        sample = pd.concat([sample, rest.sample(n=remaining, random_state=int(rng.integers(0, 1_000_000)))])

    return sample.sample(frac=1, random_state=int(rng.integers(0, 1_000_000)))


def load_ledger():
    if os.path.exists(LEDGER_PATH):
        return pd.read_csv(LEDGER_PATH, dtype={"Item_ID": int})
    return None


def save_ledger(ledger):
    os.makedirs(OUT_DIR, exist_ok=True)
    ledger.to_csv(LEDGER_PATH, index=False)


def next_pair_ids(start, n):
    return [f"C{i:03d}" for i in range(start, start + n)]


def annotators_from_ledger(ledger):
    names = set()
    for cell in ledger["Annotatori_assegnati"].dropna():
        names.update(cell.split(","))
    return names


def write_annotator_csv(rows_df, annot_name):
    score_col = f"Punteggio_{annot_name}"
    out_rows = list(CALIBRATION_ROWS)
    for _, r in rows_df.iterrows():
        out_rows.append({
            "ID_coppia": r["Pair_ID"],
            "ID_paziente_anon": r["ID_paziente_anon"],
            "Item_ID": int(r["Item_ID"]),
            "Item_Testo": r["Item_Testo"],
            "Frase_metafora": r["Frase_metafora"],
            "Note": "",
        })
    out_df = pd.DataFrame(out_rows)
    out_df[score_col] = ""
    out_df = out_df[["ID_coppia", "ID_paziente_anon", "Item_ID", "Item_Testo",
                      "Frase_metafora", score_col, "Note"]]
    path = os.path.join(OUT_DIR, f"{annot_name}.csv")
    out_df.to_csv(path, index=False)
    print(f"scritto {path} ({len(out_df)} righe, di cui {len(CALIBRATION_ROWS)} di calibrazione)")


def cmd_init(annotator_names, n_total, overlap_frac, seed):
    if load_ledger() is not None:
        raise SystemExit(f"{LEDGER_PATH} esiste gia' — usa 'add-annotator' per aggiungere annotatori, "
                          f"non rilanciare 'init' (sovrascriverebbe le coppie gia' distribuite).")

    rng = np.random.default_rng(seed)
    long = load_long_pool()

    master = stratified_sample(long, n_total, rng).reset_index(drop=True)
    master["Pair_ID"] = next_pair_ids(1, len(master))

    core_n = round(n_total * overlap_frac)
    core = stratified_sample(master, core_n, rng)
    core_ids = set(core["Pair_ID"])
    master["Set"] = np.where(master["Pair_ID"].isin(core_ids), "core", "personal")

    personal = master[master["Set"] == "personal"].sample(frac=1, random_state=int(rng.integers(0, 1_000_000)))
    n_annot = len(annotator_names)
    chunk = len(personal) // n_annot
    personal_assign = {}
    for i, name in enumerate(annotator_names):
        part = personal.iloc[i * chunk:(i + 1) * chunk] if i < n_annot - 1 else personal.iloc[i * chunk:]
        for pid in part["Pair_ID"]:
            personal_assign[pid] = name

    def assigned(row):
        if row["Set"] == "core":
            return ",".join(annotator_names)
        return personal_assign.get(row["Pair_ID"], "")

    master["Annotatori_assegnati"] = master.apply(assigned, axis=1)
    save_ledger(master)

    for name in annotator_names:
        mask = (master["Set"] == "core") | (master["Annotatori_assegnati"] == name)
        write_annotator_csv(master[mask], name)

    print(f"\nledger: {LEDGER_PATH}")
    print(f"totale coppie: {len(master)} | core (condivise): {core_n} | personal per annotatore: {chunk}")


def cmd_add_annotator(name, n_personal, seed):
    ledger = load_ledger()
    if ledger is None:
        raise SystemExit("nessun ledger trovato — lancia prima 'init' con gli annotatori iniziali.")
    if name in annotators_from_ledger(ledger):
        raise SystemExit(f"'{name}' e' gia' presente nel ledger.")

    rng = np.random.default_rng(seed if seed is not None else abs(hash(name)) % (2**32))
    long = load_long_pool()

    used_keys = set()
    for _, r in ledger.iterrows():
        used_keys.add(str(r["Metaphor_row_id"]) + "__" + str(int(r["Item_ID"])))
    available = long[~long["Pair_key"].isin(used_keys)]

    new_personal = stratified_sample(available, n_personal, rng).reset_index(drop=True)
    start_n = ledger["Pair_ID"].str.slice(1).astype(int).max() + 1
    new_personal["Pair_ID"] = next_pair_ids(start_n, len(new_personal))
    new_personal["Set"] = "personal"
    new_personal["Annotatori_assegnati"] = name

    ledger.loc[ledger["Set"] == "core", "Annotatori_assegnati"] = (
        ledger.loc[ledger["Set"] == "core", "Annotatori_assegnati"] + "," + name
    )

    keep_cols = ledger.columns
    ledger = pd.concat([ledger, new_personal[keep_cols]], ignore_index=True)
    save_ledger(ledger)

    mask = (ledger["Set"] == "core") | (ledger["Annotatori_assegnati"] == name)
    write_annotator_csv(ledger[mask], name)
    print(f"\n'{name}' aggiunto: {len(new_personal)} coppie personal nuove + {(ledger['Set']=='core').sum()} core condivise.")
    print("Nota: le CSV degli annotatori gia' esistenti non sono state toccate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="crea il pool iniziale e i CSV per i primi annotatori")
    p_init.add_argument("annotators", nargs="+", help="nomi annotatori, es: Annotatore1 Annotatore2")
    p_init.add_argument("--n-total", type=int, default=N_TOTAL_INIT)
    p_init.add_argument("--overlap", type=float, default=OVERLAP_FRAC)
    p_init.add_argument("--seed", type=int, default=SEED)

    p_add = sub.add_parser("add-annotator", help="aggiunge un nuovo annotatore (CORE condiviso + nuove coppie personal)")
    p_add.add_argument("name")
    p_add.add_argument("--n-personal", type=int, default=160)
    p_add.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    if args.cmd == "init":
        cmd_init(args.annotators, args.n_total, args.overlap, args.seed)
    elif args.cmd == "add-annotator":
        cmd_add_annotator(args.name, args.n_personal, args.seed)
