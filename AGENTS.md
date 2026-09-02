# Repository Agent Instructions

## Project Purpose

- The main project builds canonical NMR datasets and develops an NMR foundation model using combined `1H + 13C` spectra.
- Extracting embeddings from published models is a secondary benchmarking task, not the architecture of the new foundation model.

This is an extreem syntesis of the content pf the project, for your contetx:  

---

Ho esaminato esclusivamente i 10 file Markdown direttamente in `contex/`, escludendo `NMR/` e le altre sottocartelle.

#### Sintesi generale

Il progetto mira a costruire un foundation model self-supervised per spettri NMR combinati `¹H + ¹³C`, usando una rappresentazione riusabile dei picchi e valutandone il valore per property prediction. In parallelo, vuole produrre un “cookbook” ingegneristico riproducibile: dataset versionati, pipeline, test, benchmark, checkpoint e documentazione.

L’idea centrale è separare nettamente:

```text
dataset canonico → adattatore specifico del modello → batch nativo → encoder/embedding
```

Il dataset canonico non deve incorporare tokenizzazione, padding o scelte dipendenti da un checkpoint; queste appartengono ai processor dei singoli modelli.

#### Fondamenti NMR e motivazione AI

L’NMR osserva nuclei con spin, in particolare `¹H` e `¹³C`, e fornisce vincoli strutturali: ambiente elettronico tramite chemical shift, numero relativo di protoni tramite integrazione, e connettività/geometria locale tramite splitting e costanti di accoppiamento `J`.

- `¹H NMR`: segnali, shift, integrazione e molteplicità sono ricchi di informazione, ma possono sovrapporsi.
- `¹³C NMR`: complementa il protone, contando ambienti carboniosi; le intensità non sono normalmente quantitative.
- Solvente, campo magnetico, temperatura e condizioni sperimentali influenzano lo spettro: simulato ed esperimentale non sono quindi intercambiabili.

L’AI per NMR copre ricostruzione, denoising, peak picking, previsione spettro da molecola, elucidazione strutturale e analisi di miscele. La lacuna interessante è una rappresentazione NMR generale, pre-addestrata e riusabile: la maggior parte dei lavori resta concentrata su compiti supervisionati specifici.

#### Paesaggio dei dataset e strategia

Le fonti si dividono in tre gruppi.

- Dataset simulati enormi ma “shift-only”, soprattutto SimNMR-PubChem / NMR-Solver: circa 106 milioni di molecole. Offrono shift H e C, ma non molteplicità, range, linewidth o `J`.
- Dataset simulati ricchi ma più piccoli, come MST-NMR/MSD: circa 794 mila record, con tabelle di picchi e più attributi.
- Dataset sperimentali da letteratura, come NMRexp e NMRSpec/NMRTrans: più realistici ma incompleti e variabili.

Non esiste oggi un dataset aperto che sia contemporaneamente enorme, sperimentale, appaiato `¹H + ¹³C`, ricco di picchi e accoppiamenti, con tracce dense e metadati completi.

La scelta proposta è una rappresentazione per risonanza/picco, non per traccia densa né solo per shift. Ogni picco protonico può avere shift, integrazione, molteplicità, `J` e intervallo di shift; per carbonio lo shift è l’attributo essenziale. È cruciale distinguere:

- padding: nessun picco in quella posizione del batch;
- missing: attributo assente alla fonte;
- masked: attributo presente ma nascosto per il pretraining.

Il modello dovrebbe partire da milioni di esempi shift-only, poi incorporare dati ricchi simulati e sperimentali, con attribute dropout per tollerare annotazioni mancanti. Le viste simulate e sperimentali della stessa molecola vanno mantenute e possono diventare coppie positive per un allineamento contrastivo leggero, senza cancellare le differenze fisiche di dominio.

#### Schema canonico e conversione

Cinque converter producono Parquet schema v2 da NMRPeak/NMRexp, MST-NMR, NMRTrans/NMRSpec, NMRGym e SimNMR-PubChem. La conversione è deliberatamente conservativa:

- un record sorgente entra una sola volta;
- nessuna deduplicazione, split inventato, filtraggio per modello, padding o inferenza di informazioni NMR assenti;
- `record_id` è l’unico identificatore, deterministico;
- i file separati rappresentano gli split; non esiste una colonna split;
- Parquet usa liste/struct tipizzate e metadata di schema, sorgente, converter e RDKit.

Le strutture vengono normalizzate da RDKit: si conserva lo SMILES sorgente, ma si ricalcolano SMILES canonico, formula e lista ordinata degli atomi. Le coordinate upstream vengono escluse. Per SimNMR, solo 63 record con metadati chimici non coerentemente derivabili sono stati esclusi, con report JSONL.

Nei dati SimNMR, gli shift atomici sono raggruppati mediante la classe di equivalenza fornita, mai tramite semplice vicinanza di shift. Per i protoni questo conserva sia integrazione sia shift individuali dei membri. NMRPeak e NMRTrans sono già a livello di risonanza e non devono ricevere classi di equivalenza inferite.

Lo schema conserva sia `multiplicity_raw` sia `multiplicity` normalizzata. Il vocabolario compatto è quello di NMRTrans: 21 categorie fisiche più `<unk>`. Tre alias sono lossless: `p→quint`, `hept→sept`, `brd→bd`. `null` significa informazione non disponibile; `<unk>` significa annotazione presente ma non rappresentabile. Non si deve trasformare automaticamente un valore sconosciuto in `m`.

#### Qualità, filtraggio e raccolta finale

Il filtro comune è un passaggio derivato e non modifica mai le fonti canoniche. Elimina record senza picchi, valori non finiti, shift fuori da limiti fisici, oltre 60 picchi per modalità, più di 6 `J` per picco, `J` negativi, integrazioni fornite ma non positive e strutture a frammenti multipli. Non applica filtri di drug-likeness.

La deduplicazione è prudente: solo stessa struttura canonica e liste di shift H/C esattamente uguali, senza arrotondamenti. Record della stessa molecola ma con spettri differenti restano: possono rappresentare repliche, condizioni sperimentali o vista simulata/esperimentale.

La raccolta pubblicata e già materializzata con la pipeline DVC contiene:

| Componente | Record | Ruolo |
|---|---:|---|
| Rich | 1.941.847 | pretraining ricco dopo trasferimento SimNMR, filtro e disgiunzione dai cinque property dataset |
| Rich benchmark test | 87.897 | disgiunto da rich train esteso, SimNMR e NMRGym per `smiles_canonical` esatto |
| SimNMR-PubChem | 105.509.616 | grande pool shift-only simulato |
| NMRGym | 265.095 | fonte shift-only sperimentale separata |

La pipeline DVC mantenuta usa exact `smiles_canonical` per tutti gli overlap
NMR–NMR. Prima sposta dal benchmark al train gli spettri rich le cui molecole
sono in SimNMR; poi rimuove dal test residuo le molecole presenti nel train
esteso o in NMRGym. Dopo il filtro comune, prepara quattro endpoint TDC e
Sangster logP usando full InChIKey, risolve le label ripetute e rimuove i match
dal pretraining.
Proprietà molecolari e analytics vengono generate soltanto sui file finali.

I raw source sono centralizzati in `datasets/raw/`, identificati da pointer
`.dvc` e conservati nel remote Nebius. `dvc.yaml` definisce il DAG esplicito,
`params.yaml` i parametri condivisi e `datasets/raw/sources.yaml` la provenance
umana. I Parquet canonici, gli intermedi e gli audit di rimozione sono output
ricostruibili con `push: false`; dataset finali, coorti ADMET, sidecar di
proprietà, analytics e report strutturati usano il normale push. La pipeline è
stata materializzata end-to-end: i finali correnti contengono 1.941.847 record
rich, 87.897 benchmark, 105.509.616 SimNMR e 265.095 NMRGym.
`dvc.lock` e i report sono l'autorità per hash e conteggi; le cifre di release
precedenti nelle note sono solo confronti storici esplicitamente etichettati.

Nel pool ricco, MST-NMR e NMRTrans sono completamente appaiati H+C; NMRexp è l’unica fonte con record monomodali. Le distribuzioni tra train/validation e benchmark risultano simili, pur con differenze chimiche tra le sorgenti.

#### Benchmark di embedding

La pipeline estrae embedding solo da record con `¹H` e `¹³C` entrambi non vuoti. I quattro baseline correnti sono:

| Modello | Embedding | Ruolo |
|---|---:|---|
| NMRPeak-R | 768-D | encoder BART su peak table ricche |
| NMRTrans | 1024-D | media masked degli stati locali H e C |
| UltraNMR | 768-D | Transformer shift-only su liste espanse |
| NMR-Solver | 256-D | controllo fisso Gaussian/grid, non appreso |

NMRPeak-R sfrutta range, integrazione, molteplicità e `J`; UltraNMR e NMR-Solver espandono un picco protonico ripetendone lo shift secondo l’integrazione. NMRTrans usa set transformer separati per H e C; il suo output PMA è instabile rispetto al padding, quindi viene mantenuta come embedding primaria la media masked degli stati locali, non PMA.

DiffNMR è un candidato futuro per feature continue ricche, ma non è ancora integrato. I confronti devono usare identici molecole, label, split e metriche, pur lasciando a ogni checkpoint il proprio input nativo. Vanno confrontati anche fingerprint Morgan, descrittori RDKit, un encoder molecolare e combinazioni molecola+NMR.

#### Molteplicità

La normalizzazione copre bene le fonti sperimentali: dopo i tre alias, solo il 2,391% dei picchi NMRPeak resta fuori dal vocabolario NMRTrans; circa 1,344% in NMRexp e 3,523% in MST-NMR. MST-NMR è più complesso, con più categorie di splitting fini; NMRexp e NMRSpec hanno distribuzioni molto simili. Ciò giustifica un campo raw per fedeltà alla fonte e uno canonico per interoperabilità.

#### Benchmark ADMET

L’obiettivo downstream è verificare se un encoder NMR congelato contiene informazione chimica utile. Il test principale è un linear probe; un piccolo MLP è secondario.

Le coorti selezionate sono:

| Endpoint | Train/validation | Test | Tipo |
|---|---:|---:|---|
| AqSolDB solubility | 3.777 | 770 | regressione |
| LD50 Zhu | 2.748 | 574 | regressione |
| Ames | 2.527 | 455 | classificazione |

Il matching usa full InChIKey RDKit, non soltanto connectivity key, per non unire silenziosamente stereoisomeri. I record abbinati alle proprietà vengono rimossi dal pretraining ricco. Le etichette duplicate discordanti LD50 sono escluse anziché mediate; le repliche coerenti vengono consolidate. CSV delle etichette e Parquet degli spettri condividono lo stesso insieme di `record_id` unici.

---

## Documentation Guide

- `README.md` gives the repository overview and published-model boundaries.
- `datasets/README.md` explains raw, canonical, intermediate, and cleaned
  directory ownership, DVC policies, and reproduction.
- `dvc.yaml`, `params.yaml`, and `datasets/raw/sources.yaml` define the runnable
  data graph, tunable values, and source provenance.
- `contex/NMR foundations and AI.md` covers NMR fundamentals and the broader AI research context.
- `contex/Datasets.md` explains the dataset landscape, schema rationale,
  multiplicity harmonization, canonical pipeline, and training strategy.
- `contex/Canonicalization_Implementation_Notes.md` is the authority for the
  implemented schema, conversions, ranges, missing values, provenance, and IDs.
- `contex/Multiplicity analysis.md` records vocabulary evidence, lossless aliases,
  and measured coverage.
- `contex/Dataset_Filtering_and_Processing.md` distinguishes source curation,
  benchmark selection, model preparation, and canonical conversion.
- `contex/Dataset Analysis.md` records measured dataset properties and processor
  compatibility.
- `contex/Embedding_Pipeline_Architecture.md` defines component responsibilities,
  embedding flow, extraction behavior, and current limitations.
- `contex/Baseline NMR Encoders.md` explains benchmark models, inputs, pooling,
  and comparison choices.
- `contex/Properties Dataset.md` covers downstream property benchmarks and
  leakage-safe evaluation.
- `contex/NMR/DL Methods/` contains model- and dataset-specific notes, paper summaries, implementation observations, and links to relevant work.

## Repository Navigation

- Before changing the data DAG, read `README.md`, `datasets/README.md`,
  `scripts/data/README.md`, `dvc.yaml`, and `params.yaml`; regenerate stage
  definitions with `scripts/data/generate_dvc_pipeline.sh`.
- Before changing canonical schema or shared data code, read `contex/Datasets.md`,
  `contex/Canonicalization_Implementation_Notes.md`, and
  `contex/Multiplicity analysis.md`.
- Before converting a dataset, also read
  `contex/Dataset_Filtering_and_Processing.md`, `contex/Dataset Analysis.md`, and
  `scripts/data/canonicalize/README.md`.
- Before changing model benchmarks, read `contex/Embedding_Pipeline_Architecture.md`,
  `contex/Baseline NMR Encoders.md`, and the relevant note under
  `contex/NMR/DL Methods/`.
- Before changing embedding extraction, also read the selected model's upstream
  README under `models/`.
- Put reusable schema, validation, and dataset code directly in `scripts/data/`.
- Put source-specific conversion code in `scripts/data/canonicalize/`.
- Put derived-data and benchmark-preparation utilities in
  `scripts/data/postprocess/`.
- Put published-model processors and embedding wrappers in `scripts/model_benchmarks/`.

## Architecture and Boundaries

- Canonical data code must remain model-independent and reusable for future foundation-model training.
- Keep the intended flow:

  `canonical dataset -> model-specific processor/collator -> model-specific batch -> lightweight embedder or trainable model`

- Do not place reusable canonical dataset logic inside the benchmarking package.
- Processors own validation, model-native conversion, padding, and collation.
  Embedders own checkpoint loading, encoder execution, and pooling.
- Reuse official preprocessing, checkpoint-loading, and model code where practical.
- Keep model-family imports lazy. Different families may require separate Python environments and should run one family per process.
- The current published-model benchmark scope requires combined `1H + 13C` input.
- Follow the canonicalization notes rather than restating or guessing schema rules.
  Preserve missing values versus empty collections and raw versus normalized
  annotations. Do not silently invent missing NMR information.
- Report assumptions, incompatible records, and unsupported cases clearly. Do not
  hide them with implicit clipping, filtering, coercion, or lossy mappings.
- Do not process, rewrite, move, merge, filter, or convert real datasets unless
  explicitly requested.
- Do not download data, modify large checkpoints, or edit cloned repositories under `models/` unless explicitly required.

## Code Style

- Write clear beginner-to-intermediate research Python that a strong university student can read and modify.
- Prefer simple functions, small classes, explicit control flow, descriptive names,
  limited type hints, and short comments for non-obvious model-specific logic.
- A little duplication is acceptable when it makes model paths easier to follow.
- Keep useful checks and clear errors with the minimum required complexity.
- Avoid deep inheritance, advanced generics or protocols, decorators,
  metaprogramming, excessive factories or registries, configuration layers,
  one-use abstractions, and clever compact code.
- Do not design a general framework for hypothetical future requirements.

## Tests

- Run tests from the repository root with `scripts` on `PYTHONPATH` and use the
  environment appropriate to the model family. Use `nmr-env main` for tests and scripts. Use model-specific environments
  (see `scripts/envs_scr/README.md`) only for tasks that require the model
  submodules.

```bash
PYTHONPATH=scripts python -m unittest discover -s scripts/data/canonicalize/tests -v
PYTHONPATH=scripts python -m unittest discover -s scripts/data/postprocess/tests -v
PYTHONPATH=scripts python -m unittest discover -s scripts/model_benchmarks/tests -v
PYTHONPATH=scripts python -m unittest scripts.model_benchmarks.tests.test_nmrtrans_mask -v
```

- Keep testing lightweight and focused on canonical dataset loading, multiplicity
  normalization, model processors, checkpoint loading and embedding extraction,
  and the NMRTrans padding/PMA diagnostic.
- Tests must use small fixtures and must not scan or rewrite real datasets.
- Model dependency or checkpoint tests may skip when the required local environment
  or asset is unavailable. Do not require exhaustive tests for small helpers unless
  the task specifically needs them.
