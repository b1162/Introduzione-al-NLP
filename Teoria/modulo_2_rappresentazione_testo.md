# Modulo 2: Rappresentazione del Testo e Word Embeddings

## Introduzione alla rappresentazione del testo

La rappresentazione del testo è un concetto fondamentale nel Natural Language Processing (NLP) che riguarda il modo in cui trasformiamo il linguaggio naturale, comprensibile agli esseri umani, in un formato che possa essere elaborato efficacemente dai computer. Questo processo di trasformazione è essenziale perché i computer non "comprendono" naturalmente il linguaggio umano: hanno bisogno di rappresentazioni numeriche o vettoriali per poter eseguire calcoli e applicare algoritmi.

La qualità della rappresentazione del testo influenza direttamente l'efficacia di qualsiasi sistema NLP. Una rappresentazione inadeguata può limitare significativamente le prestazioni anche degli algoritmi più sofisticati, mentre una rappresentazione efficace può consentire anche a modelli relativamente semplici di ottenere risultati sorprendenti.

Nel corso dell'evoluzione dell'NLP, sono stati sviluppati diversi approcci per rappresentare il testo, ciascuno con i propri punti di forza e limitazioni. In questo modulo, esploreremo le principali tecniche di rappresentazione del testo, con particolare attenzione ai word embeddings, che hanno rivoluzionato il campo negli ultimi anni.

> **Concetti Chiave**
> 
> - La rappresentazione del testo trasforma il linguaggio naturale in formati comprensibili per i computer
> - La qualità della rappresentazione influenza direttamente le prestazioni dei sistemi NLP
> - L'evoluzione delle tecniche di rappresentazione riflette il progresso generale nel campo dell'NLP

## Rappresentazioni tradizionali del testo

Prima di addentrarci nel mondo dei word embeddings, è importante comprendere le tecniche tradizionali di rappresentazione del testo che hanno dominato il campo per decenni e che, in alcuni contesti, continuano ad essere utilizzate efficacemente.

### One-Hot Encoding

Una delle rappresentazioni più semplici è il one-hot encoding, dove ogni parola del vocabolario è rappresentata da un vettore di dimensione pari alla dimensione del vocabolario, con un 1 nella posizione corrispondente alla parola e 0 in tutte le altre posizioni.

Ad esempio, in un vocabolario di 10.000 parole, la parola "hotel" potrebbe essere rappresentata da un vettore con un 1 nella posizione 3.542 e 0 in tutte le altre 9.999 posizioni.

Questa rappresentazione è concettualmente semplice ma presenta significative limitazioni:

- **Alta dimensionalità**: Per vocabolari realistici (tipicamente decine o centinaia di migliaia di parole), i vettori diventano estremamente grandi e sparsi.
- **Nessuna informazione semantica**: Non cattura alcuna relazione tra le parole; ogni parola è equidistante da tutte le altre, indipendentemente dal significato.
- **Nessuna generalizzazione**: Non c'è modo di inferire informazioni su parole rare o mai viste durante l'addestramento.

### Bag-of-Words (BoW)

Il modello Bag-of-Words rappresenta un documento come un vettore che conta la frequenza di ciascuna parola del vocabolario nel documento, ignorando l'ordine delle parole.

Ad esempio, la frase "il gatto insegue il topo" potrebbe essere rappresentata come un vettore dove le posizioni corrispondenti a "il", "gatto", "insegue" e "topo" contengono rispettivamente 2, 1, 1 e 1, mentre tutte le altre posizioni sono 0.

Nonostante la sua semplicità, il BoW è stato ampiamente utilizzato in applicazioni come la classificazione di documenti e l'information retrieval. Tuttavia, presenta limitazioni significative:

- **Perdita dell'ordine delle parole**: "Il gatto insegue il topo" e "Il topo insegue il gatto" avrebbero la stessa rappresentazione.
- **Perdita di contesto**: Non cattura il significato contestuale delle parole.
- **Alta dimensionalità**: Come il one-hot encoding, soffre di vettori sparsi di grandi dimensioni.

### Term Frequency-Inverse Document Frequency (TF-IDF)

TF-IDF è un'evoluzione del BoW che pesa le parole non solo in base alla loro frequenza in un documento (TF), ma anche in base alla loro rarità nell'intero corpus di documenti (IDF). L'idea è che parole che appaiono frequentemente in un documento ma raramente negli altri documenti sono probabilmente più informative e caratterizzanti.

La formula base per TF-IDF è:

TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)

dove:
- TF(t, d) è la frequenza del termine t nel documento d
- IDF(t, D) è il logaritmo del rapporto tra il numero totale di documenti e il numero di documenti contenenti il termine t

TF-IDF migliora il BoW in termini di capacità discriminativa, ma condivide molte delle sue limitazioni fondamentali riguardo alla mancanza di informazioni semantiche e contestuali.

### N-grams

Gli N-grams estendono il concetto di BoW considerando sequenze di N parole consecutive invece di parole singole. Ad esempio, i bi-grams (N=2) della frase "il gatto insegue il topo" sarebbero "il gatto", "gatto insegue", "insegue il", "il topo".

Questo approccio cattura parzialmente l'ordine locale delle parole e alcune relazioni contestuali, ma la dimensionalità aumenta esponenzialmente con N, rendendo impraticabile l'uso di N-grams lunghi.

> **Applicazione Aziendale: Analisi dei Brevetti con TF-IDF**
> 
> Le aziende tecnologiche utilizzano TF-IDF per analizzare brevetti e identificare innovazioni rilevanti:
> - Estrazione dei termini più distintivi da migliaia di documenti brevettuali
> - Identificazione di tecnologie emergenti in specifici settori
> - Monitoraggio delle attività di brevettazione dei concorrenti
> - Valutazione di potenziali acquisizioni basata su portfolio brevetti
> 
> Esempio: IBM utilizza analisi TF-IDF avanzata per monitorare il panorama brevettuale in settori strategici come l'intelligenza artificiale, identificando rapidamente nuove tendenze tecnologiche e potenziali opportunità di mercato.

## Word Embeddings: Rappresentare le Parole come Vettori

### Cos'è un word embedding

Un word embedding è una rappresentazione vettoriale di una parola in uno spazio multidimensionale continuo. In termini più semplici, si tratta di trasformare ogni parola in un vettore di numeri reali, dove la posizione della parola nello spazio vettoriale cattura relazioni semantiche e sintattiche con altre parole.

A differenza delle rappresentazioni one-hot, i word embeddings sono:

- **Densi**: Utilizzano vettori di dimensione relativamente piccola (tipicamente da 100 a 300 dimensioni) con valori reali in ogni dimensione.
- **Semanticamente significativi**: Parole con significati simili si trovano vicine nello spazio vettoriale.
- **Capaci di catturare relazioni**: La geometria dello spazio degli embeddings può riflettere relazioni semantiche e sintattiche tra parole.

La proprietà più sorprendente dei word embeddings è che catturano analogie e relazioni semantiche attraverso operazioni vettoriali. L'esempio classico è:

vector("re") - vector("uomo") + vector("donna") ≈ vector("regina")

Questa proprietà emerge naturalmente durante l'addestramento e riflette le relazioni semantiche presenti nei dati di addestramento.

### Principi di funzionamento

I word embeddings si basano sull'ipotesi distribuzionale della semantica, sintetizzata dalla famosa citazione del linguista J.R. Firth: "You shall know a word by the company it keeps" (Conoscerai una parola dalla compagnia che frequenta).

In pratica, i word embeddings vengono appresi osservando i contesti in cui le parole appaiono in grandi corpora di testo. Parole che appaiono in contesti simili tendono ad avere significati simili e, di conseguenza, embeddings simili.

I modelli di word embedding vengono tipicamente addestrati su obiettivi predittivi, come:

- Predire una parola dato il suo contesto (CBOW - Continuous Bag of Words)
- Predire il contesto data una parola (Skip-gram)
- Predire la probabilità di co-occorrenza di coppie di parole (GloVe)

Durante l'addestramento, i vettori delle parole vengono gradualmente aggiustati per minimizzare l'errore di predizione, portando a rappresentazioni che catturano implicitamente relazioni semantiche e sintattiche.

### Principali tecniche di word embedding

#### Word2Vec

Sviluppato da ricercatori di Google nel 2013, Word2Vec è stato uno dei primi e più influenti algoritmi di word embedding. Esistono due varianti principali:

- **Continuous Bag of Words (CBOW)**: Predice una parola target dato il contesto circostante (le parole vicine).
- **Skip-gram**: Predice il contesto circostante data una parola target.

Entrambi i modelli utilizzano una rete neurale shallow con un singolo strato nascosto. Durante l'addestramento, la rete impara a predire parole basandosi sul loro contesto, e i pesi dello strato nascosto diventano i vettori di embedding.

Word2Vec ha dimostrato sorprendenti proprietà algebriche: operazioni vettoriali sui word embeddings possono riflettere relazioni semantiche. Oltre all'esempio "re - uomo + donna ≈ regina", altre analogie catturate includono relazioni di capitale-paese, tempo verbale, comparativi e superlativi.

#### GloVe (Global Vectors for Word Representation)

Sviluppato da ricercatori di Stanford nel 2014, GloVe combina i vantaggi dei metodi basati su contesto locale (come Word2Vec) con quelli basati su statistiche globali di co-occorrenza. GloVe costruisce una matrice di co-occorrenza delle parole e poi addestra un modello per predire il logaritmo delle probabilità di co-occorrenza.

La funzione obiettivo di GloVe è progettata per catturare sia informazioni locali (come Word2Vec) che globali (statistiche di co-occorrenza sull'intero corpus). Questo approccio ibrido permette a GloVe di catturare meglio alcune relazioni semantiche rispetto a Word2Vec, soprattutto per parole che co-occorrono raramente ma hanno relazioni semantiche significative.

#### FastText

Sviluppato da Facebook AI Research nel 2016, FastText estende Word2Vec considerando i caratteri n-gram all'interno delle parole. Invece di apprendere vettori per parole intere, FastText apprende vettori per frammenti di parole (n-gram di caratteri) e rappresenta una parola come la somma dei vettori dei suoi n-gram.

Questo approccio offre due vantaggi principali:

- **Gestione di parole fuori vocabolario**: Può generare embeddings per parole mai viste durante l'addestramento (out-of-vocabulary) basandosi sui loro n-gram.
- **Migliore gestione di lingue morfologicamente ricche**: Funziona meglio per lingue con molte forme flesse (come l'italiano, il tedesco o il finlandese) dove le parole possono avere molte varianti.

Ad esempio, anche se la parola "imprenditorialità" non fosse presente nel corpus di addestramento, FastText potrebbe generare un embedding ragionevole basandosi sui suoi n-gram, molti dei quali sarebbero condivisi con parole come "imprenditore" o "imprenditoriale".

> **Concetti Chiave**
> 
> - I word embeddings rappresentano le parole come vettori densi in uno spazio multidimensionale
> - Parole semanticamente simili hanno vettori simili nello spazio degli embeddings
> - Word2Vec utilizza un approccio predittivo basato sul contesto locale
> - GloVe combina informazioni locali e globali di co-occorrenza
> - FastText considera i caratteri n-gram, migliorando la gestione di parole rare o morfologicamente complesse

## Visualizzazione e interpretazione dei word embeddings

Una delle sfide nell'utilizzo dei word embeddings è la loro interpretabilità. Con centinaia di dimensioni, è difficile comprendere intuitivamente cosa rappresenti ciascuna dimensione o come le parole siano organizzate nello spazio vettoriale.

### Tecniche di riduzione della dimensionalità

Per visualizzare gli embeddings, si utilizzano tecniche di riduzione della dimensionalità che proiettano i vettori ad alta dimensionalità in uno spazio bidimensionale o tridimensionale, preservando il più possibile le relazioni di similarità.

Le tecniche più comuni includono:

- **Principal Component Analysis (PCA)**: Identifica le direzioni di massima varianza nei dati e proietta i vettori su queste direzioni principali.
- **t-Distributed Stochastic Neighbor Embedding (t-SNE)**: Preserva meglio le relazioni di vicinanza locale, rendendo visibili cluster di parole semanticamente correlate.
- **Uniform Manifold Approximation and Projection (UMAP)**: Un'alternativa più recente a t-SNE che può preservare sia la struttura locale che quella globale.

Queste visualizzazioni rivelano spesso cluster semantici interessanti: parole relative a numeri, colori, professioni, o paesi tendono a formare gruppi distinti nello spazio degli embeddings.

### Interpretazione delle dimensioni

Sebbene le singole dimensioni dei word embeddings non abbiano generalmente un'interpretazione semantica chiara (a differenza, ad esempio, delle dimensioni in un'analisi fattoriale), è possibile identificare direzioni significative nello spazio degli embeddings.

Ad esempio, si può identificare una "direzione di genere" calcolando la differenza tra coppie di parole come "uomo"-"donna", "re"-"regina", "attore"-"attrice". Proiettando altre parole su questa direzione, si può quantificare quanto siano associate a concetti maschili o femminili.

Analogamente, si possono identificare direzioni corrispondenti a concetti come formalità/informalità, positività/negatività, o temporalità (passato/futuro).

### Valutazione degli embeddings

La qualità dei word embeddings può essere valutata attraverso vari benchmark:

- **Test di analogia**: Valutano la capacità degli embeddings di risolvere analogie del tipo "a sta a b come c sta a ?".
- **Test di similarità**: Misurano la correlazione tra similarità coseno degli embeddings e giudizi umani di similarità semantica.
- **Valutazione estrinseca**: Misura le prestazioni degli embeddings quando utilizzati in compiti downstream come classificazione di testi o named entity recognition.

È importante notare che diversi embeddings possono eccellere in diversi tipi di valutazione, e la scelta dell'embedding più appropriato dipende dall'applicazione specifica.

> **Applicazione Aziendale: Ricerca Semantica nei Documenti Legali**
> 
> Gli studi legali e i dipartimenti legali aziendali utilizzano word embeddings per migliorare la ricerca nei documenti:
> - Ricerca di concetti legali simili anche quando espressi con terminologia diversa
> - Identificazione di precedenti rilevanti basata sulla similarità semantica
> - Organizzazione automatica di grandi archivi di documenti legali
> - Suggerimento di clausole contrattuali pertinenti durante la redazione di nuovi contratti
> 
> Esempio: Kira Systems utilizza tecnologie basate su word embeddings per analizzare contratti e documenti legali, permettendo ai professionisti di estrarre informazioni rilevanti da migliaia di documenti in una frazione del tempo richiesto dalla revisione manuale.

## Applicazioni dei word embeddings

I word embeddings hanno trasformato numerose applicazioni di NLP, migliorando significativamente le prestazioni in vari compiti. Vediamo alcune delle applicazioni più rilevanti in diversi settori.

### Miglioramento della ricerca semantica

I word embeddings permettono di implementare sistemi di ricerca che comprendono il significato delle query, non solo le parole esatte. Questo approccio, noto come ricerca semantica, migliora significativamente la pertinenza dei risultati.

Nel settore dell'e-commerce, la ricerca semantica basata su embeddings permette ai clienti di trovare prodotti pertinenti anche quando utilizzano sinonimi, descrizioni generiche o termini correlati. Ad esempio, una ricerca per "abbigliamento invernale" potrebbe restituire risultati per "cappotti", "sciarpe" e "guanti", anche se questi termini specifici non sono menzionati nella query.

Nel settore dell'editoria digitale, la ricerca semantica migliora l'esperienza degli utenti permettendo loro di trovare articoli rilevanti basati su concetti piuttosto che su semplici corrispondenze di parole chiave. Questo aumenta il tempo di permanenza sul sito e migliora l'engagement.

### Categorizzazione avanzata dei documenti

I word embeddings migliorano significativamente l'accuratezza dei sistemi di classificazione dei testi, permettendo di categorizzare correttamente documenti anche quando utilizzano terminologie diverse o non standard.

Nel settore assicurativo, i sistemi basati su embeddings possono categorizzare automaticamente le richieste di risarcimento in base al tipo di incidente, alla copertura applicabile e alla gravità, anche quando descritte con linguaggio non tecnico dai clienti. Questo accelera il processo di gestione delle richieste e migliora l'accuratezza dell'instradamento.

Nel settore dell'analisi dei social media, gli embeddings permettono di categorizzare i post in base a temi, sentiment o intenti, facilitando il monitoraggio della reputazione del brand e l'identificazione di tendenze emergenti.

### Sistemi di raccomandazione basati su contenuto

I word embeddings possono essere utilizzati per creare rappresentazioni vettoriali di documenti, prodotti o contenuti, permettendo di calcolare similarità semantiche e generare raccomandazioni pertinenti.

Nel settore dei media e dell'intrattenimento, piattaforme di streaming utilizzano embeddings per analizzare le descrizioni dei contenuti, le recensioni e i metadati, generando raccomandazioni basate sulla similarità semantica. Questo complementa gli approcci basati sul collaborative filtering, migliorando la diversità e la pertinenza delle raccomandazioni.

Nel settore del recruiting, i sistemi di matching tra curriculum e offerte di lavoro utilizzano embeddings per comprendere le competenze, le esperienze e i requisiti oltre le semplici corrispondenze di parole chiave, migliorando la qualità delle candidature e riducendo il tempo di assunzione.

### Analisi del sentiment più accurata

I word embeddings permettono ai modelli di analisi del sentiment di generalizzare meglio, riconoscendo il tono emotivo anche in frasi che utilizzano parole non presenti nel dataset di addestramento.

Nel settore finanziario, l'analisi del sentiment basata su embeddings viene applicata a notizie economiche, report di analisti e social media per prevedere movimenti di mercato o valutare la percezione di specifiche aziende o settori.

Nel settore della ristorazione e dell'ospitalità, l'analisi del sentiment avanzata permette di estrarre insights granulari dalle recensioni dei clienti, identificando aspetti specifici (cibo, servizio, atmosfera, pulizia) che generano feedback positivi o negativi.

### Rilevamento di temi emergenti

Analizzando i cluster di parole nello spazio degli embeddings, è possibile identificare temi o problemi emergenti nelle comunicazioni, anche prima che questi vengano esplicitamente etichettati o categorizzati.

Nel settore farmaceutico, l'analisi di forum di pazienti e social media attraverso embeddings può identificare precocemente effetti collaterali non documentati o utilizzi off-label di farmaci, fornendo informazioni preziose per la farmacovigilanza.

Nel settore del marketing, l'analisi di conversazioni online può identificare nuove tendenze di consumo, preoccupazioni emergenti o opportunità di mercato non ancora evidenti attraverso i tradizionali studi di mercato.

> **Concetti Chiave**
> 
> - La ricerca semantica utilizza embeddings per trovare contenuti concettualmente rilevanti
> - La categorizzazione avanzata dei documenti beneficia della generalizzazione offerta dagli embeddings
> - I sistemi di raccomandazione possono utilizzare embeddings per calcolare similarità semantiche
> - L'analisi del sentiment diventa più robusta grazie alla capacità di generalizzazione degli embeddings
> - Il rilevamento di temi emergenti sfrutta la struttura dello spazio degli embeddings

## Sentence Embeddings e Document Embeddings

Mentre i word embeddings rappresentano singole parole, molte applicazioni richiedono rappresentazioni di frasi, paragrafi o interi documenti. Vediamo come i concetti di embedding possono essere estesi oltre il livello della singola parola.

### Da word embeddings a sentence embeddings

Il modo più semplice per ottenere un sentence embedding è aggregare i word embeddings delle parole che compongono la frase, tipicamente attraverso media o somma pesata. Sebbene questo approccio perda informazioni sull'ordine delle parole, può essere sorprendentemente efficace per molte applicazioni.

Approcci più sofisticati includono:

- **Smooth Inverse Frequency (SIF)**: Calcola una media pesata dei word embeddings, dove i pesi sono inversamente proporzionali alla frequenza delle parole, e poi rimuove la componente principale comune a tutte le frasi.
- **Doc2Vec**: Un'estensione di Word2Vec che apprende direttamente embeddings per documenti insieme a embeddings per parole.
- **Universal Sentence Encoder (USE)**: Un modello pre-addestrato che genera embeddings di frasi ottimizzati per la similarità semantica.

### Modelli avanzati per sentence embeddings

Modelli più recenti basati su architetture Transformer hanno portato a significativi miglioramenti nella qualità dei sentence embeddings:

- **BERT Sentence Embeddings**: Utilizzando la rappresentazione del token [CLS] o la media dei token di output di BERT, si possono ottenere embeddings di alta qualità per frasi o paragrafi.
- **Sentence-BERT (SBERT)**: Una modificazione di BERT specificamente ottimizzata per generare sentence embeddings semanticamente significativi, con prestazioni state-of-the-art in compiti di similarità semantica.
- **SimCSE (Simple Contrastive Learning of Sentence Embeddings)**: Utilizza tecniche di apprendimento contrastivo per migliorare ulteriormente la qualità degli embeddings di frasi.

Questi modelli avanzati catturano molto meglio il significato complessivo delle frasi rispetto alla semplice aggregazione di word embeddings, considerando l'ordine delle parole, le dipendenze sintattiche e il contesto.

### Applicazioni dei sentence embeddings

I sentence embeddings hanno numerose applicazioni pratiche:

- **Clustering semantico di documenti**: Raggruppare automaticamente documenti simili per contenuto, facilitando l'organizzazione di grandi archivi documentali.
- **Rilevamento di duplicati e near-duplicates**: Identificare contenuti sostanzialmente simili anche quando espressi con parole diverse.
- **Sistemi di risposta a domande**: Trovare passaggi di testo che semanticamente rispondono a una domanda, anche senza corrispondenze esatte di parole chiave.
- **Riassunto estrattivo**: Selezionare le frasi più rappresentative di un documento basandosi sulla similarità con l'intero documento.

Nel settore bancario, i sentence embeddings vengono utilizzati per analizzare e categorizzare comunicazioni con i clienti, identificare potenziali problemi o opportunità, e automatizzare risposte a richieste comuni.

Nel settore della ricerca accademica, i sentence embeddings facilitano la scoperta di letteratura rilevante oltre le semplici corrispondenze di parole chiave, permettendo ai ricercatori di identificare connessioni tra campi apparentemente non correlati.

> **Applicazione Aziendale: Analisi dei Feedback dei Dipendenti**
> 
> Le aziende utilizzano sentence embeddings per analizzare i feedback dei dipendenti:
> - Clustering di risposte aperte in sondaggi sulla soddisfazione dei dipendenti
> - Identificazione di temi ricorrenti nelle valutazioni delle performance
> - Monitoraggio del sentiment e del morale aziendale nel tempo
> - Rilevamento precoce di problemi organizzativi o di leadership
> 
> Esempio: Microsoft utilizza tecniche avanzate di sentence embedding per analizzare i feedback dei dipendenti nei sondaggi interni, identificando rapidamente aree di preoccupazione e misurando l'impatto delle iniziative di miglioramento della cultura aziendale.

## Sfide e limitazioni dei word embeddings

Nonostante i loro numerosi vantaggi, i word embeddings tradizionali presentano alcune limitazioni significative che è importante considerare quando si progettano applicazioni NLP.

### Polisemia e ambiguità

I word embeddings tradizionali (non contestuali) assegnano un singolo vettore a ogni parola, indipendentemente dal contesto. Questo è problematico per parole polisemiche, che hanno significati diversi in contesti diversi.

Ad esempio, la parola "calcio" può riferirsi a uno sport, a un elemento chimico, o all'azione di colpire con il piede. Un word embedding tradizionale rappresenterebbe tutti questi significati con un unico vettore, creando una rappresentazione "media" che potrebbe non essere ottimale per nessuno dei significati specifici.

Nel settore della traduzione automatica, questa limitazione può portare a traduzioni errate quando una parola nella lingua di origine ha più possibili traduzioni nella lingua di destinazione, a seconda del contesto.

### Bias e stereotipi

I word embeddings apprendono dai dati su cui vengono addestrati, e questi dati spesso riflettono bias e stereotipi presenti nella società. Ad esempio, è stato dimostrato che embeddings addestrati su corpora di testo generici tendono a associare professioni come "ingegnere" o "programmatore" più strettamente con termini maschili, e professioni come "infermiere" o "segretario" più strettamente con termini femminili.

Questi bias possono propagarsi e amplificarsi nelle applicazioni downstream, portando a risultati discriminatori. Ad esempio, un sistema di screening dei curriculum basato su embeddings potrebbe penalizzare inconsapevolmente candidati di un certo genere per determinate posizioni.

Nel settore dei media e della pubblicità, sistemi di generazione di contenuti o di targeting pubblicitario basati su embeddings biased potrebbero perpetuare stereotipi dannosi o escludere certi gruppi demografici.

### Dipendenza dalla qualità e quantità dei dati

La qualità degli embeddings dipende fortemente dalla qualità e quantità dei dati di addestramento. Embeddings addestrati su corpora generici potrebbero non catturare adeguatamente la terminologia specifica di un dominio o di un'azienda.

Nel settore medico, ad esempio, termini specialistici o acronimi potrebbero essere assenti o mal rappresentati in embeddings generici, richiedendo l'addestramento di embeddings specifici su letteratura medica.

Inoltre, lingue o dialetti con risorse limitate (pochi dati disponibili) tendono ad avere embeddings di qualità inferiore rispetto a lingue ampiamente rappresentate come l'inglese.

### Mancanza di comprensione profonda

Mentre i word embeddings catturano relazioni semantiche superficiali, non forniscono una vera "comprensione" del linguaggio nel senso umano del termine. Non catturano relazioni logiche complesse, ragionamento o conoscenza del mondo reale.

Ad esempio, un sistema basato su embeddings potrebbe riconoscere che "Parigi" e "Francia" sono correlati, ma non necessariamente "comprendere" che Parigi è la capitale della Francia o cosa significhi essere una capitale.

Nel settore del customer care, questa limitazione può manifestarsi nell'incapacità di sistemi basati su embeddings tradizionali di comprendere richieste complesse che richiedono ragionamento multi-step o conoscenza contestuale.

### Evoluzione verso embeddings contestuali

Per superare molte di queste limitazioni, la ricerca si è spostata verso embeddings contestuali, dove la rappresentazione di una parola dipende dal contesto specifico in cui appare.

Modelli come ELMo, BERT e GPT generano embeddings dinamici che variano in base al contesto della frase, catturando meglio il significato specifico di parole polisemiche e relazioni contestuali più complesse.

Ad esempio, in BERT, la parola "banca" avrebbe rappresentazioni diverse nelle frasi "Ho depositato i soldi in banca" e "Mi sono seduto sulla banca del parco", riflettendo i suoi diversi significati.

Questi modelli contestuali rappresentano l'evoluzione naturale dei word embeddings e saranno esplorati più in dettaglio nei moduli successivi.

> **Concetti Chiave**
> 
> - I word embeddings tradizionali non gestiscono bene la polisemia, assegnando un unico vettore a parole con significati multipli
> - Gli embeddings possono ereditare e amplificare bias presenti nei dati di addestramento
> - La qualità degli embeddings dipende dalla qualità e quantità dei dati disponibili
> - Gli embeddings catturano relazioni semantiche ma non forniscono una vera comprensione o ragionamento
> - Gli embeddings contestuali rappresentano l'evoluzione che supera molte di queste limitazioni

## Implementazione pratica dei word embeddings

L'implementazione pratica dei word embeddings nelle applicazioni reali richiede considerazioni su vari aspetti, dalla scelta del modello appropriato all'integrazione con i sistemi esistenti.

### Scelta del modello di embedding

La scelta del modello di embedding più appropriato dipende da vari fattori:

- **Dominio applicativo**: Per applicazioni in domini specialistici (medico, legale, finanziario), embeddings addestrati su testi di dominio spesso performano meglio di embeddings generici.
- **Lingua**: Per lingue diverse dall'inglese, è importante verificare la disponibilità e la qualità degli embeddings pre-addestrati.
- **Risorse computazionali**: Modelli più complessi come BERT richiedono significativamente più risorse rispetto a Word2Vec o GloVe.
- **Compito specifico**: Alcuni modelli performano meglio per certi compiti; ad esempio, FastText è particolarmente adatto per lingue morfologicamente ricche o applicazioni con molti termini tecnici o neologismi.

### Utilizzo di embeddings pre-addestrati vs addestramento custom

In molti casi, è possibile utilizzare embeddings pre-addestrati disponibili pubblicamente, come:

- Word2Vec addestrato su Google News
- GloVe addestrato su Wikipedia e Gigaword
- FastText addestrato su Wikipedia, Common Crawl o altri corpora

Questi modelli pre-addestrati offrono una soluzione rapida e efficace per molte applicazioni generiche.

Tuttavia, l'addestramento di embeddings custom può essere vantaggioso quando:

- Si lavora con terminologia di dominio molto specifica
- Si hanno grandi quantità di dati proprietari rilevanti
- Si necessita di ottimizzare gli embeddings per un compito specifico

L'addestramento custom richiede più risorse ma può portare a miglioramenti significativi nelle prestazioni per applicazioni specializzate.

### Integrazione nei pipeline NLP

L'integrazione degli embeddings nei pipeline NLP può avvenire in vari modi:

- **Feature engineering**: Utilizzare embeddings come features di input per modelli di machine learning tradizionali (SVM, Random Forest, etc.)
- **Layer di embedding in reti neurali**: Inizializzare il primo layer di una rete neurale con embeddings pre-addestrati, con possibilità di fine-tuning durante l'addestramento
- **Calcolo di similarità**: Utilizzare direttamente la similarità coseno tra embeddings per applicazioni come ricerca semantica o clustering

Nel settore retail, ad esempio, un sistema di ricerca prodotti potrebbe utilizzare embeddings per calcolare la similarità tra query dell'utente e descrizioni dei prodotti, migliorando la pertinenza dei risultati rispetto a approcci basati su semplice matching di parole chiave.

### Tecniche di debiasing

Per mitigare i bias negli embeddings, sono state sviluppate varie tecniche di debiasing:

- **Hard debiasing**: Identifica direzioni di bias nello spazio degli embeddings (es. direzione di genere) e neutralizza esplicitamente queste componenti per parole che non dovrebbero essere gender-specific.
- **Soft debiasing**: Utilizza tecniche di regolarizzazione durante l'addestramento per ridurre l'apprendimento di associazioni biased.
- **Augmentation dei dati**: Arricchisce i dati di addestramento con esempi che contrastano gli stereotipi esistenti.

Nel settore delle risorse umane, l'applicazione di tecniche di debiasing agli embeddings utilizzati nei sistemi di screening dei curriculum può contribuire a processi di selezione più equi e diversificati.

> **Applicazione Aziendale: Ottimizzazione della Supply Chain**
> 
> Le aziende manifatturiere utilizzano word embeddings per analizzare dati della supply chain:
> - Categorizzazione semantica di componenti e materiali oltre le nomenclature standard
> - Identificazione di fornitori alternativi per componenti simili durante carenze
> - Analisi di report di qualità e non conformità per identificare pattern ricorrenti
> - Ottimizzazione dell'inventario basata su similarità funzionale dei componenti
> 
> Esempio: Siemens utilizza tecnologie basate su embeddings per analizzare la documentazione tecnica e i dati della supply chain, identificando opportunità di standardizzazione dei componenti e riducendo la complessità della catena di fornitura.

## Conclusione

I word embeddings hanno rivoluzionato il campo del Natural Language Processing, fornendo rappresentazioni vettoriali dense e semanticamente significative che catturano relazioni linguistiche in modo sorprendentemente efficace. Dalla loro introduzione con Word2Vec nel 2013, hanno trasformato praticamente ogni aspetto dell'NLP, migliorando significativamente le prestazioni in compiti come la classificazione dei testi, l'analisi del sentiment, la ricerca semantica e molti altri.

In questo modulo, abbiamo esplorato il concetto di rappresentazione del testo, dalle tecniche tradizionali come Bag-of-Words e TF-IDF fino ai moderni word embeddings. Abbiamo approfondito i principi di funzionamento di Word2Vec, GloVe e FastText, e abbiamo esaminato come queste rappresentazioni possano essere visualizzate e interpretate. Abbiamo discusso numerose applicazioni pratiche in vari settori aziendali, evidenziando come i word embeddings stiano trasformando processi e creando valore in contesti diversi.

Abbiamo anche riconosciuto le limitazioni dei word embeddings tradizionali, in particolare riguardo alla polisemia, ai bias e alla mancanza di comprensione profonda. Queste limitazioni hanno spinto l'evoluzione verso modelli contestuali come BERT e GPT, che saranno esplorati nei moduli successivi.

La comprensione dei word embeddings fornisce una base essenziale per apprezzare i modelli più avanzati che dominano l'NLP contemporaneo. Mentre i Large Language Models e i Transformer hanno portato le capacità NLP a nuovi livelli, i principi fondamentali della rappresentazione vettoriale del linguaggio introdotti dai word embeddings rimangono centrali nel funzionamento di questi sistemi più sofisticati.

Nel prossimo modulo, esploreremo la classificazione del testo e l'analisi del sentiment, due applicazioni fondamentali che beneficiano significativamente dell'uso dei word embeddings e che hanno numerose applicazioni pratiche nel mondo aziendale.

> **Riepilogo del Modulo**
> 
> - La rappresentazione del testo è fondamentale per trasformare il linguaggio in formati comprensibili per i computer
> - Le tecniche tradizionali come BoW e TF-IDF hanno limitazioni significative nella cattura di relazioni semantiche
> - I word embeddings rappresentano le parole come vettori densi che catturano relazioni semantiche e sintattiche
> - Word2Vec, GloVe e FastText sono le principali tecniche di word embedding, ciascuna con punti di forza specifici
> - I sentence embeddings estendono il concetto oltre le singole parole, permettendo di rappresentare frasi e documenti
> - Le applicazioni spaziano dalla ricerca semantica all'analisi del sentiment, dalla categorizzazione dei documenti ai sistemi di raccomandazione
> - Nonostante le limitazioni, i word embeddings hanno trasformato l'NLP e posto le basi per i modelli contestuali più avanzati

## Riferimenti e Approfondimenti

- Mikolov, T., et al. (2013). Distributed Representations of Words and Phrases and their Compositionality. Advances in Neural Information Processing Systems, 26.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP).
- Bojanowski, P., et al. (2017). Enriching Word Vectors with Subword Information. Transactions of the Association for Computational Linguistics, 5.
- Bolukbasi, T., et al. (2016). Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings. Advances in Neural Information Processing Systems, 29.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP).
