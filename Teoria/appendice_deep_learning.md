# Appendice: Introduzione al Deep Learning

## 1. Cos'è il Deep Learning?

Il Deep Learning, o apprendimento profondo, è una branca specifica del Machine Learning (Apprendimento Automatico) che, a sua volta, è un sottocampo dell'Intelligenza Artificiale (IA). La caratteristica distintiva del Deep Learning è l'utilizzo di **reti neurali artificiali** con molteplici strati (da cui il termine "profondo") per analizzare e apprendere da grandi quantità di dati.

A differenza dei metodi di Machine Learning tradizionali, che spesso richiedono un intervento umano significativo per l'ingegnerizzazione delle feature (cioè, per selezionare e trasformare le variabili di input più rilevanti), i modelli di Deep Learning sono in grado di **apprendere automaticamente le rappresentazioni gerarchiche dei dati**. Questo significa che, partendo da dati grezzi, il modello impara a identificare pattern e feature a diversi livelli di astrazione, dai più semplici ai più complessi.

Ad esempio, in un compito di riconoscimento di immagini, i primi strati di una rete neurale profonda potrebbero imparare a riconoscere bordi e angoli, gli strati intermedi potrebbero combinare queste feature per identificare forme più complesse come occhi o nasi, e gli strati finali potrebbero assemblare queste informazioni per riconoscere un volto intero.

## 2. Perché il Deep Learning è diventato così popolare?

Sebbene i concetti fondamentali delle reti neurali esistano da decenni, il Deep Learning ha vissuto un'esplosione di popolarità e successo solo negli ultimi anni. Questo è dovuto principalmente alla convergenza di tre fattori chiave:

1.  **Grandi quantità di dati (Big Data)**: L'era digitale ha portato a una disponibilità senza precedenti di dati (testi, immagini, video, ecc.), che sono essenziali per addestrare modelli di Deep Learning complessi e performanti. Più dati ha a disposizione un modello, meglio può generalizzare e apprendere pattern complessi.
2.  **Potenza computazionale (GPU)**: L'addestramento di reti neurali profonde richiede una notevole potenza di calcolo. Lo sviluppo di Unità di Elaborazione Grafica (GPU) ad alte prestazioni, originariamente progettate per i videogiochi, ha reso possibile addestrare modelli molto grandi in tempi ragionevoli. Le GPU sono particolarmente adatte per le operazioni matematiche (principalmente moltiplicazioni di matrici) che sono alla base del funzionamento delle reti neurali.
3.  **Miglioramenti algoritmici**: Sono stati sviluppati nuovi algoritmi, architetture di rete (come i Transformer), funzioni di attivazione e tecniche di ottimizzazione che hanno reso l'addestramento delle reti profonde più stabile ed efficiente, superando problemi storici come il "vanishing gradient" (scomparsa del gradiente).

## 3. Concetti di Base del Deep Learning

### Neuroni Artificiali e Strati (Layers)

L'unità fondamentale di una rete neurale è il **neurone artificiale** (o nodo). Un neurone riceve uno o più input, esegue una somma pesata di questi input, aggiunge un termine di bias e poi passa il risultato attraverso una **funzione di attivazione**.

I neuroni sono organizzati in **strati (layers)**:

*   **Strato di input (Input Layer)**: Riceve i dati grezzi.
*   **Strati nascosti (Hidden Layers)**: Sono gli strati intermedi dove avvengono le elaborazioni principali. Una rete è considerata "profonda" se ha più strati nascosti.
*   **Strato di output (Output Layer)**: Produce il risultato finale del modello (es. una classificazione, una previsione numerica).

### Funzioni di Attivazione

Le funzioni di attivazione introducono la **non-linearità** nel modello, permettendo alla rete di apprendere relazioni complesse che vanno oltre le semplici trasformazioni lineari. Senza funzioni di attivazione non lineari, una rete neurale profonda, indipendentemente dal numero di strati, si comporterebbe come un singolo strato lineare.

Alcune funzioni di attivazione comuni includono:

*   **Sigmoide**: Mappa l'input in un valore tra 0 e 1. Usata storicamente, ma meno comune negli strati nascosti moderni a causa del problema del vanishing gradient.
    ```
    σ(x) = 1 / (1 + e^(-x))
    ```
*   **ReLU (Rectified Linear Unit)**: Restituisce l'input se è positivo, altrimenti zero. È molto efficiente dal punto di vista computazionale e aiuta a mitigare il vanishing gradient.
    ```
    ReLU(x) = max(0, x)
    ```
*   **Tanh (Tangente Iperbolica)**: Mappa l'input in un valore tra -1 e 1.
*   **Softmax**: Usata tipicamente nello strato di output per problemi di classificazione multi-classe, produce una distribuzione di probabilità sui possibili output.

### Reti Neurali Feedforward (Feedforward Neural Networks - FNN)

Le reti neurali feedforward, note anche come Multi-Layer Perceptrons (MLP), sono il tipo più semplice di rete neurale artificiale. L'informazione si muove in una sola direzione: dagli strati di input, attraverso gli strati nascosti, fino allo strato di output. Non ci sono cicli o connessioni all'indietro (durante la fase di inferenza).

## 4. Addestramento delle Reti Neurali

L'addestramento di una rete neurale è il processo attraverso cui il modello impara a mappare gli input agli output desiderati, aggiustando i suoi **pesi** e **bias**.

### Funzione di Perdita (Loss Function)

Una **funzione di perdita** (o funzione di costo) misura quanto le previsioni del modello si discostano dai valori reali (target) durante l'addestramento. L'obiettivo dell'addestramento è minimizzare questa funzione di perdita.

Esempi di funzioni di perdita:

*   **Mean Squared Error (MSE)**: Usata comunemente per problemi di regressione.
*   **Cross-Entropy Loss (Entropia Incrociata)**: Usata per problemi di classificazione.

### Backpropagation e Discesa del Gradiente (Gradient Descent)

L'algoritmo di **backpropagation** (retropropagazione dell'errore) è il cuore dell'addestramento delle reti neurali. Calcola il gradiente della funzione di perdita rispetto a ciascun peso e bias nella rete. Questo gradiente indica la direzione in cui i pesi dovrebbero essere modificati per ridurre la perdita.

La **discesa del gradiente** è un algoritmo di ottimizzazione che utilizza questi gradienti per aggiornare iterativamente i pesi e i bias del modello, muovendosi gradualmente verso un minimo della funzione di perdita.

Il **learning rate** (tasso di apprendimento) è un iperparametro cruciale che determina la dimensione dei passi effettuati durante la discesa del gradiente.

### Ottimizzatori (Optimizers)

Esistono diverse varianti della discesa del gradiente, chiamate ottimizzatori, che cercano di rendere l'addestramento più veloce e stabile. Alcuni ottimizzatori popolari includono:

*   **SGD (Stochastic Gradient Descent)**: Aggiorna i pesi utilizzando un singolo esempio di addestramento (o un piccolo batch) alla volta.
*   **Adam (Adaptive Moment Estimation)**: Un ottimizzatore sofisticato che adatta il learning rate per ciascun parametro e combina i vantaggi di altri ottimizzatori come AdaGrad e RMSProp.

## 5. Architetture Comuni di Deep Learning (Cenni)

Oltre alle FNN, esistono architetture specializzate per diversi tipi di dati e compiti:

### Reti Neurali Convoluzionali (Convolutional Neural Networks - CNN)

Le CNN sono particolarmente efficaci nell'elaborazione di dati con una struttura a griglia, come le immagini. Utilizzano strati convoluzionali per applicare filtri che rilevano pattern locali (es. bordi, texture). Nel NLP, le CNN possono essere usate per compiti come la classificazione di testi, catturando pattern locali nelle sequenze di parole.

### Reti Neurali Ricorrenti (Recurrent Neural Networks - RNN)

Le RNN sono progettate per elaborare dati sequenziali, come testi o serie temporali. Hanno connessioni cicliche che permettono all'informazione di persistere da un passo temporale all'altro, creando una sorta di "memoria". Varianti come LSTM (Long Short-Term Memory) e GRU (Gated Recurrent Unit) sono state sviluppate per affrontare meglio le dipendenze a lungo termine. Come discusso nel corso, i Transformer hanno largamente superato le RNN in molti compiti NLP, ma le RNN rimangono importanti concettualmente.

## 6. Deep Learning per l'NLP (Cenni)

Il Deep Learning ha rivoluzionato l'NLP, portando a progressi significativi in quasi tutti i suoi sottocampi.

*   **Word Embeddings**: Tecniche come Word2Vec, GloVe e FastText (già trattate nel corso) utilizzano reti neurali per apprendere rappresentazioni vettoriali dense delle parole che catturano relazioni semantiche.
*   **Modelli Sequence-to-Sequence (Seq2Seq)**: Architetture composte da un encoder e un decoder (spesso basate su RNN o Transformer) utilizzate per compiti come la traduzione automatica, il riassunto e la generazione di risposte.
*   **Architettura Transformer**: Come ampiamente discusso nel corso, i Transformer, basati sul meccanismo di self-attention, sono diventati lo standard de facto per la maggior parte dei compiti NLP avanzati, alimentando modelli come BERT e GPT.

## 7. Strumenti e Framework Popolari

L'implementazione di modelli di Deep Learning è facilitata da numerosi framework open-source:

*   **TensorFlow (Google)**: Un ecosistema completo per il machine learning, con un focus sul deep learning. Keras è un'API di alto livello integrata in TensorFlow che semplifica la costruzione di reti neurali.
*   **PyTorch (Facebook/Meta)**: Un framework popolare, noto per la sua flessibilità e l'approccio "pythonico", particolarmente apprezzato nella comunità di ricerca.
*   **Hugging Face Transformers**: Una libreria che fornisce migliaia di modelli pre-addestrati (principalmente Transformer) e strumenti per l'NLP, costruita su PyTorch e TensorFlow.

## 8. Considerazioni Etiche e Limitazioni

Come per l'NLP in generale, anche il Deep Learning solleva importanti questioni etiche:

*   **Bias**: I modelli possono apprendere e amplificare i bias presenti nei dati di addestramento.
*   **Trasparenza e Interpretabilità**: Le reti neurali profonde sono spesso considerate "scatole nere", rendendo difficile capire come prendono le loro decisioni.
*   **Costi computazionali ed energetici**: L'addestramento di modelli di grandi dimensioni richiede risorse significative.
*   **Uso improprio**: Possibilità di generare disinformazione (deepfakes, testo generato artificialmente) o per scopi di sorveglianza.

È fondamentale approcciare lo sviluppo e l'applicazione del Deep Learning con consapevolezza e responsabilità.

## Conclusione dell'Appendice

Questa appendice ha fornito un'introduzione ai concetti fondamentali del Deep Learning. Comprendere queste basi è utile per apprezzare appieno le capacità e il funzionamento dei modelli NLP avanzati, come i Transformer, che sono al centro di questo corso. Il Deep Learning è un campo in rapida evoluzione, e le tecniche qui descritte sono solo la punta dell'iceberg di un mondo vasto e affascinante.
