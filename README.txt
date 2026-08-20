README - Analisi metafore NLI + PANSS


STRUTTURA CARTELLA

progetto/
├── NLI_Analysis.ipynb
├── NLI_Analysis.py
├── data/
│   ├── Schizo_Metaphors.xlsx
│   ├── Database_Voices.xlsx
│   └── Database_Delusions_14042026.xlsx
└── output/
    └── (csv, xlsx, png dei risultati)


FILE

NLI_Analysis.ipynb   notebook
NLI_Analysis.py       stesso codice, versione script


REQUIREMENTS

pandas, numpy, torch, transformers, tqdm, scipy, matplotlib, openpyxl

Il codice scarica da HF il modello NLI (mDeBERTa) al primo avvio.


FILE DI INPUT (cartella data)

- Schizo_Metaphors.xlsx: tutte le metafore (Delusion e Voices)
- Database_Voices.xlsx: dati e PANSS dei pazienti Voices
- Database_Delusions_14042026.xlsx: dati e PANSS dei pazienti Delusion


ESECUZIONE

Da script: python NLI_Analysis.py

Se non vuoi ricalcolare tutti gli score NLI da zero, trovi gia' nella
cartella "output" i file generati in precedenza.


OUTPUT (cartella output, creata in automatico)

- il file con tutte le metafore classificate (csv e xlsx)
- i punteggi medi per paziente
- le tabelle delle correlazioni con il PANSS
- i grafici (barplot, boxplot, heatmap)
- le tabelle riassuntive (numero pazienti, numero metafore, PANSS medio)
- output.zip: tutto il contenuto della cartella compresso
