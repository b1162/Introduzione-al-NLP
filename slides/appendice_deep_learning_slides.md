---
marp: true
theme: default
paginate: true
backgroundColor: #fff
header: "Appendice: Introduzione al Deep Learning"
footer: "Corso di Natural Language Processing"
---

<!-- _class: lead -->
# Appendice: Introduzione al Deep Learning 🤖🧠
## Un Tuffo nel Cuore dell'Intelligenza Artificiale Moderna

<!--
Speaker Notes:
Benvenuti a questa appendice dedicata al Deep Learning. In questo modulo vi spiegherò i concetti fondamentali che stanno alla base di molte delle rivoluzioni che abbiamo visto nell'NLP e in altri campi dell'IA. Anche se il corso si concentra sull'NLP, una comprensione di base del Deep Learning è cruciale per capire appieno come funzionano modelli potenti come i Transformer, BERT e GPT. Preparatevi per un viaggio affascinante nel mondo delle reti neurali profonde! 
Voglio che pensiate a questo come a un viaggio dentro una tecnologia che sta cambiando il mondo, dalla traduzione automatica ai sistemi di riconoscimento vocale, fino alle auto a guida autonoma. Vi guiderò passo passo, con esempi semplici, per farvi capire come queste reti funzionano davvero e perché sono così potenti.
-->

---

## Indice dell'Appendice

1.  Cos'è il Deep Learning?
2.  Perché è diventato così popolare?
3.  Concetti di Base: Neuroni, Strati, Funzioni di Attivazione
4.  Reti Neurali Feedforward (FNN)
5.  Addestramento: Funzione di Perdita, Backpropagation, Ottimizzatori
6.  Architetture Comuni (CNN, RNN - cenni)
7.  Deep Learning per l'NLP (cenni)
8.  Strumenti e Framework Popolari
9.  Considerazioni Etiche e Limitazioni

<!--
Speaker Notes:
Ecco una panoramica degli argomenti che tratteremo. Inizierò con una definizione di Deep Learning e il motivo della sua recente ascesa. Poi, ci addentreremo nei mattoni fondamentali: neuroni, strati e funzioni di attivazione. Vi spiegherò come le reti vengono addestrate e vi darò uno sguardo ad alcune architetture comuni, prima di concludere con un accenno al suo impatto sull'NLP, gli strumenti disponibili e le importanti considerazioni etiche.
Durante il percorso, cercherò di fornire esempi concreti e di mantenere tutto il più chiaro possibile, così che anche chi non ha mai studiato informatica o matematica possa seguire con facilità.
-->

---

## 1. Cos'è il Deep Learning?

-   Branca del **Machine Learning (ML)**, che è un sottocampo dell'**Intelligenza Artificiale (IA)**.
-   Utilizza **Reti Neurali Artificiali** con **molteplici strati** (da cui "profondo").
-   Capace di apprendere **rappresentazioni gerarchiche dei dati** automaticamente da dati grezzi.
    -   *Esempio immagini*: bordi -> forme -> oggetti.


<!--
Speaker Notes:
Iniziamo definendo il Deep Learning. Immaginate il Deep Learning come un sottoinsieme specializzato del Machine Learning. La sua caratteristica distintiva è l'uso di reti neurali artificiali che hanno molti strati – ecco da dove viene il termine "profondo". La vera magia del Deep Learning sta nella sua capacità di imparare da sola le caratteristiche rilevanti dai dati, senza che dobbiamo dirgli esplicitamente cosa cercare. Per esempio, se gli forniamo molte immagini di gatti, imparerà da sola a riconoscere prima i bordi, poi le forme come orecchie e baffi, e infine l'intero concetto di "gatto".
Quindi, se avete mai usato un assistente vocale come Siri o Alexa, o un traduttore automatico come Google Translate, sappiate che dietro ci sono proprio queste reti profonde che analizzano e comprendono il linguaggio o le immagini. È come se la rete imparasse a vedere e capire il mondo in modo simile a come fa il nostro cervello, ma attraverso calcoli matematici. Questo rende il Deep Learning uno degli strumenti più potenti e versatili dell'intelligenza artificiale moderna.
-->

---

## 2. Perché il Deep Learning è diventato così popolare?

Convergenza di tre fattori chiave:

1.  📊 **Grandi quantità di dati (Big Data)**: Essenziali per addestrare modelli complessi.
2.  💻 **Potenza computazionale (GPU)**: Ha reso l'addestramento di modelli grandi fattibile.
3.  💡 **Miglioramenti algoritmici**: Nuove architetture, funzioni di attivazione, tecniche di ottimizzazione.

<!--
Speaker Notes:
Vi starete chiedendo perché il Deep Learning sia esploso solo di recente, nonostante le idee di base esistano da tempo. Vi spiego: la risposta sta nella perfetta tempesta creata da tre elementi. Primo, l'enorme quantità di dati che generiamo ogni giorno – i cosiddetti Big Data. Pensate ai dati che ogni giorno generiamo: foto, post, video… tutto questo alimenta questi modelli e li rende sempre più precisi. Secondo, l'avvento di hardware molto potente, in particolare le GPU (le schede grafiche), che possono eseguire i calcoli necessari molto velocemente. Questo ha permesso di addestrare modelli molto grandi in tempi ragionevoli. Terzo, importanti progressi negli algoritmi stessi, che hanno reso l'addestramento di queste reti più efficiente e stabile, come nuove funzioni di attivazione e tecniche di ottimizzazione.
In pratica, questi tre fattori insieme hanno reso possibile ciò che prima era solo teoria o esperimenti limitati. È un po’ come avere una ricetta segreta, gli ingredienti giusti e un forno potente: solo così si può sfornare una torta perfetta!
-->

---

## 3. Concetti di Base: Neuroni Artificiali

-   Unità fondamentale: **neurone artificiale** (o nodo).
-   Riceve input, esegue una **somma pesata**, aggiunge un **bias**.
-   Passa il risultato attraverso una **funzione di attivazione**.


<!--
Speaker Notes:
Ora entriamo nei dettagli tecnici, partendo dal componente base: il neurone artificiale. Pensatelo come una piccola unità di calcolo. Riceve diversi segnali di input, ognuno con un certo "peso" o importanza. Il neurone somma questi input pesati, aggiunge un valore fisso chiamato bias (che aiuta a regolare l'output), e poi applica una funzione di attivazione. Questa funzione decide se e come il neurone "si attiva" e passa l'informazione allo strato successivo.
Potete immaginare il neurone artificiale come una specie di rubinetto che regola quanto un’informazione passa allo strato successivo, a seconda del peso e della funzione di attivazione. Se il segnale è abbastanza forte, il rubinetto si apre e lascia passare l'informazione; se è debole, resta chiuso o apre poco. Questo meccanismo permette alla rete di decidere quali segnali sono importanti e quali no, proprio come fa il nostro cervello con le informazioni che riceve.
-->

---

## 3. Concetti di Base: Strati (Layers)

I neuroni sono organizzati in strati:

-   **Strato di Input (Input Layer)**: Riceve i dati grezzi.
-   **Strati Nascosti (Hidden Layers)**: Elaborazioni intermedie. Più strati nascosti = rete "profonda".
-   **Strato di Output (Output Layer)**: Produce il risultato finale.

![bg right:40% 90%](images/03_strati_rete_neurale.png)

<!--
Speaker Notes:
Questi neuroni non lavorano da soli, ma sono organizzati in strati. Abbiamo lo strato di input, che è la porta d'ingresso dei nostri dati. Poi ci sono uno o più strati nascosti, dove avviene la maggior parte dell'elaborazione e dell'apprendimento. Se ci sono molti strati nascosti, parliamo di una rete "profonda". Infine, c'è lo strato di output, che ci dà il risultato finale del modello, come una previsione o una classificazione.
Pensate a questi strati come a una catena di montaggio: l'input entra, viene lavorato da ogni stazione (strato nascosto) che aggiunge valore o trasforma l'informazione, e alla fine esce il prodotto finito, cioè la risposta della rete. Ogni strato ha un ruolo specifico e insieme permettono alla rete di imparare rappresentazioni sempre più astratte e utili dei dati.
-->

---

## 3. Concetti di Base: Funzioni di Attivazione

Introducono la **non-linearità**, permettendo di apprendere relazioni complesse.

-   **Sigmoide**: `σ(x) = 1 / (1 + e^(-x))` (output tra 0 e 1)
    *(Nota: Immagine illustrativa non disponibile al momento)*
-   **ReLU (Rectified Linear Unit)**: `ReLU(x) = max(0, x)` (efficiente, mitiga vanishing gradient)
-   **Tanh**: Output tra -1 e 1
-   **Softmax**: Per classificazione multi-classe (output come distribuzione di probabilità)


<!--
Speaker Notes:
Le funzioni di attivazione sono cruciali. Senza di esse, anche una rete con molti strati si comporterebbe come un semplice modello lineare. Queste funzioni introducono la non-linearità, che è fondamentale per permettere alla rete di imparare pattern complessi e non solo relazioni dirette. La ReLU (immagine in alto a destra) è molto popolare oggi perché è semplice ed efficiente. La Tanh (immagine in basso a sinistra) e la Softmax (immagine in basso a destra) sono altre funzioni comuni, quest'ultima tipicamente usata nello strato finale per problemi di classificazione multi-classe.
Per la Sigmoide, purtroppo, non abbiamo un'immagine al momento, ma il suo grafico è una curva a "S".
Immaginate la funzione di attivazione come un filtro o un interruttore che decide come trasformare il segnale in ingresso. A seconda della funzione scelta, il neurone può "accendersi" solo se il segnale supera una certa soglia, o può modulare l'output in modi diversi. Questo è fondamentale per permettere alla rete di apprendere relazioni non banali e di rappresentare dati complessi come immagini, suoni o testi.
-->

---

## 4. Reti Neurali Feedforward (FNN / MLP)

-   Tipo più semplice di rete neurale artificiale.
-   Informazione si muove in **una sola direzione**: input -> hidden -> output.
-   Nessun ciclo o connessione all'indietro (in inferenza).
-   Anche note come **Multi-Layer Perceptrons (MLP)**.

![bg right:40% 90%](images/05_fnn_mlp.gif)

<!--
Speaker Notes:
Le reti neurali feedforward, o MLP, sono la forma più basilare di rete neurale. Il nome "feedforward" significa che l'informazione fluisce sempre in avanti, dallo strato di input, attraverso gli strati nascosti, fino allo strato di output. Non ci sono loop o connessioni che tornano indietro, almeno non durante la fase in cui il modello fa una previsione (la fase di inferenza).
Pensate a questa rete come a una linea di montaggio molto semplice dove il prodotto passa sempre in avanti, senza tornare indietro o ripassare da una stazione precedente. Questo rende il processo più semplice da capire e da calcolare, ma limita la capacità della rete di gestire dati sequenziali o dipendenze nel tempo, che richiedono architetture più complesse.
L'immagine animata qui mostra chiaramente questo flusso unidirezionale, così potete visualizzare come l'informazione si sposta attraverso la rete.
-->

---

## 5. Addestramento: Funzione di Perdita (Loss Function)

-   Misura quanto le previsioni del modello si discostano dai valori reali (target).
-   Obiettivo dell'addestramento: **minimizzare la funzione di perdita**.

Esempi:
-   **Mean Squared Error (MSE)**: Per problemi di regressione (prevedere un numero).
    `MSE = (1/n) * Σ(y_true - y_pred)^2`
-   **Cross-Entropy Loss**: Per problemi di classificazione (prevedere una categoria).


<!--
Speaker Notes:
Come fa una rete neurale a imparare? Tutto inizia con la funzione di perdita. Questa funzione è come un insegnante che dice al modello quanto è sbagliata la sua previsione rispetto alla risposta corretta. L'obiettivo di tutto il processo di addestramento è trovare i pesi e i bias che rendono questa perdita il più piccola possibile.
Immaginate di voler insegnare a un bambino a lanciare una palla in un cestino. Ogni volta che sbaglia, gli dite quanto lontano è andata la palla dal bersaglio. La funzione di perdita fa esattamente questo per la rete neurale: quantifica l'errore.
Per esempio, se stiamo cercando di prevedere il prezzo di una casa, potremmo usare l'errore quadratico medio (MSE), che penalizza molto gli errori grandi. Se stiamo classificando email come spam o non spam, useremmo la cross-entropy, che misura la differenza tra la probabilità prevista e quella reale.
Durante l'addestramento, la rete cerca di ridurre questa perdita, migliorando così le sue previsioni.
-->

---

## 5. Addestramento: Backpropagation e Discesa del Gradiente

-   **Backpropagation**: Algoritmo per calcolare il **gradiente** della funzione di perdita rispetto a ogni peso/bias.
    -   Il gradiente indica la direzione per ridurre la perdita.
-   **Discesa del Gradiente (Gradient Descent)**: Algoritmo di ottimizzazione che aggiorna iterativamente i pesi/bias usando i gradienti.
    -   **Learning Rate**: Dimensione dei passi durante l'aggiornamento.

*(Nota: Immagine illustrativa per Backpropagation/Discesa del Gradiente non disponibile al momento)*

<!--
Speaker Notes:
La backpropagation è l'algoritmo magico che permette alle reti di imparare. Dopo aver calcolato la perdita, la backpropagation calcola quanto ogni singolo peso e bias nella rete ha contribuito a quell'errore. Questo contributo è chiamato gradiente.
Potete immaginare il gradiente come una freccia che indica la direzione in cui dobbiamo muoverci per migliorare, cioè per ridurre l'errore.
Una volta che abbiamo i gradienti, usiamo un algoritmo chiamato discesa del gradiente. Immaginate di essere su una montagna e voler scendere al punto più basso: il gradiente vi dice qual è la direzione della discesa più ripida. La discesa del gradiente fa piccoli passi in quella direzione, aggiornando i pesi, fino a raggiungere (sperabilmente) un buon minimo della funzione di perdita. Il learning rate controlla quanto grandi sono questi passi.
Se i passi sono troppo grandi, rischiamo di saltare il minimo; se sono troppo piccoli, l'addestramento sarà lento. Quindi scegliere un buon learning rate è importante.
Questo meccanismo è alla base di come la rete "impara" dai dati, correggendo continuamente i suoi parametri per migliorare le previsioni.
-->

---

## 5. Addestramento: Ottimizzatori (Optimizers)

Varianti della discesa del gradiente per un addestramento più veloce e stabile.

-   **SGD (Stochastic Gradient Descent)**: Usa un singolo esempio (o un piccolo batch) alla volta.
-   **Adam (Adaptive Moment Estimation)**: Adatta il learning rate per ciascun parametro. Molto popolare ed efficace.
-   Altri: AdaGrad, RMSProp, Adadelta.

<!--
Speaker Notes:
La discesa del gradiente semplice ha delle varianti, chiamate ottimizzatori, che cercano di rendere il processo di apprendimento più efficiente. L'SGD, o discesa stocastica del gradiente, aggiorna i pesi usando solo un piccolo sottoinsieme di dati alla volta, il che può rendere l'addestramento più veloce e aiutare a evitare minimi locali scarsi.
Adam è un ottimizzatore molto popolare oggi perché spesso funziona bene su una vasta gamma di problemi senza richiedere troppa messa a punto manuale degli iperparametri. Adam combina i vantaggi di altri metodi, adattando dinamicamente il learning rate per ogni parametro e accelerando la convergenza.
Pensate agli ottimizzatori come a diversi modi di scalare una montagna: alcuni sono più diretti ma rischiosi, altri più lenti ma sicuri. Scegliere l'ottimizzatore giusto può fare una grande differenza nella velocità e qualità dell'apprendimento.
-->

---

## 6. Architetture Comuni (Cenni)

Oltre alle FNN, architetture specializzate:

-   🧠 **Reti Neurali Convoluzionali (CNN)**:
    -   Efficaci per dati a griglia (es. immagini).
    -   Usano strati convoluzionali (filtri) per pattern locali.
    -   In NLP: classificazione testi (pattern locali di parole).
    ![bg right:40% 90%](images/08_cnn_architecture.png)


<!--
Speaker Notes:
Oltre alle reti feedforward, ci sono architetture specializzate. Le CNN (immagine in alto) sono le regine del riconoscimento di immagini. Usano operazioni chiamate convoluzioni, che sono come dei piccoli filtri che scorrono sull'immagine per trovare pattern come bordi o texture.
Nell'NLP, possono essere usate per trovare pattern in piccole sequenze di parole, ad esempio per riconoscere frasi o espressioni ricorrenti.
Le RNN (immagine in basso), invece, sono state progettate specificamente per le sequenze, come il testo. Hanno una sorta di memoria interna che gli permette di tener conto di ciò che è venuto prima nella sequenza.
LSTM e GRU sono versioni più sofisticate di RNN che gestiscono meglio le dipendenze a lunga distanza. Anche se i Transformer le hanno superate in molti compiti NLP, è importante conoscerle.
In pratica, queste architetture sono strumenti diversi per problemi diversi: le CNN sono ottime per dati con struttura spaziale, mentre le RNN sono adatte per dati sequenziali e temporali.
-->

---

-   🔄 **Reti Neurali Ricorrenti (RNN)**:
    -   Per dati sequenziali (testi, serie temporali).
    -   Connessioni cicliche (memoria).
    -   Varianti: LSTM, GRU (per dipendenze a lungo termine).
    !![bg right:40% 90%](images/09_rnn_architecture.png)


<!--
Speaker Notes:
Oltre alle reti feedforward, ci sono architetture specializzate. Le CNN (immagine in alto) sono le regine del riconoscimento di immagini. Usano operazioni chiamate convoluzioni, che sono come dei piccoli filtri che scorrono sull'immagine per trovare pattern come bordi o texture. Nell'NLP, possono essere usate per trovare pattern in piccole sequenze di parole. Le RNN (immagine in basso), invece, sono state progettate specificamente per le sequenze, come il testo. Hanno una sorta di memoria interna che gli permette di tener conto di ciò che è venuto prima nella sequenza. LSTM e GRU sono versioni più sofisticate di RNN che gestiscono meglio le dipendenze a lunga distanza. Anche se i Transformer le hanno superate in molti compiti NLP, è importante conoscerle.
Immaginate le RNN come a una catena in cui ogni anello ricorda ciò che è successo prima, così da poter usare questa memoria per decidere cosa fare dopo. Questo è fondamentale per capire il contesto in una frase o in una serie temporale.
-->

---

## 7. Deep Learning per l'NLP (Cenni)

Il Deep Learning ha rivoluzionato l'NLP:

-   **Word Embeddings**: Word2Vec, GloVe, FastText (reti neurali per rappresentazioni vettoriali di parole).
-   **Modelli Sequence-to-Sequence (Seq2Seq)**: Encoder-Decoder per traduzione, riassunto.
-   **Architettura Transformer**: Self-attention, standard per NLP avanzato (BERT, GPT).

<!--
Speaker Notes:
L'impatto del Deep Learning sull'NLP è stato enorme. Tecniche come Word2Vec, che abbiamo già visto, usano reti neurali per creare quelle rappresentazioni dense di parole (embeddings) che catturano il significato. I modelli Sequence-to-Sequence, spesso composti da un encoder e un decoder, sono la base per la traduzione automatica e il riassunto.
E, naturalmente, l'architettura Transformer, che è il cuore di questo corso, è un prodotto del Deep Learning e ha portato a modelli incredibilmente potenti come BERT e GPT.
Questi modelli hanno rivoluzionato il modo in cui le macchine comprendono e generano il linguaggio naturale, permettendo applicazioni che fino a poco tempo fa sembravano fantascienza, come chatbot intelligenti, assistenti personali e molto altro.
Vi guiderò attraverso questi concetti più avanti nel corso, ma è importante capire fin da subito quanto il Deep Learning sia centrale in tutto questo.
-->
