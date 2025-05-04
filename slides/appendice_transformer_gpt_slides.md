---
marp: true
theme: default
paginate: true
footer: "Corso NLP - Appendice: Transformer e GPT"
style: |
  section {
    font-size: 1.5em;
  }
  h1 {
    color: #0078D7;
  }
  h2 {
    color: #00BCF2;
  }
  code {
    background-color: #f0f0f0;
    padding: 0.2em 0.4em;
    border-radius: 3px;
  }
---

# Architettura Transformer e Implementazione di GPT from Scratch 🚀

![bg right:40% 80%](slides/transformer/images/transformer_blackbox.avif)

<!-- 
In questa presentazione, esploreremo l'architettura Transformer, che ha rivoluzionato il campo del Natural Language Processing. Introdotta nel 2017 con il paper "Attention is All You Need", questa architettura ha superato le limitazioni delle reti neurali ricorrenti (RNN) e convoluzionali (CNN), diventando la base per modelli come BERT e GPT.

Vedremo in dettaglio come funziona questa architettura e implementeremo un modello GPT semplificato da zero utilizzando PyTorch. Questo vi darà una comprensione profonda non solo teorica ma anche pratica di come funzionano questi potenti modelli.

Domanda per coinvolgere la classe: Chi di voi ha già utilizzato modelli basati su Transformer come ChatGPT o BERT? Per quali applicazioni?
-->

---

## Perché i Transformer hanno rivoluzionato l'NLP? 🤔

- **Parallelizzazione completa** vs elaborazione sequenziale delle RNN
- **Self-attention**: cattura dipendenze a lungo termine direttamente
- **Rappresentazioni posizionali** esplicite
- **Architettura scalabile** per modelli sempre più grandi

<!-- 
I Transformer hanno rappresentato una svolta epocale nel campo dell'NLP per diverse ragioni fondamentali.

Prima dei Transformer, le RNN elaboravano il testo parola per parola, in modo sequenziale, rendendo l'addestramento lento e difficile da parallelizzare. I Transformer invece elaborano l'intera sequenza contemporaneamente, permettendo una parallelizzazione massiccia sui GPU.

Il meccanismo di self-attention permette a ogni parola di "guardare" direttamente a tutte le altre parole della sequenza, catturando dipendenze a lungo termine in modo molto più efficace rispetto alle RNN, che spesso soffrivano del problema del "vanishing gradient" con sequenze lunghe.

Poiché non c'è ricorrenza, le informazioni sulla posizione vengono aggiunte esplicitamente attraverso encoding posizionali, permettendo al modello di comprendere l'ordine delle parole.

Infine, l'architettura si è dimostrata incredibilmente scalabile, permettendo di costruire modelli sempre più grandi e potenti, come dimostrato dalla serie GPT.

Domanda per la classe: Quali limitazioni delle RNN avete sperimentato nei vostri progetti di NLP? Come pensate che i Transformer possano superare queste limitazioni?
-->

---

## Quiz: Cosa rende speciale l'architettura Transformer? 🧩

<div style="display: flex; justify-content: space-between;">
<div style="width: 48%;">

**A)** Utilizza solo reti convoluzionali  
**B)** Processa sequenze elemento per elemento  

</div>
<div style="width: 48%;">

**C)** Si basa sul meccanismo di self-attention  
**D)** Richiede meno dati per l'addestramento  

</div>
</div>

<!-- 
Questo è un momento interattivo per verificare la comprensione iniziale della classe sull'architettura Transformer.

La risposta corretta è C: Si basa sul meccanismo di self-attention.

Spiegazione delle opzioni:
A) Falso - I Transformer non utilizzano reti convoluzionali, ma si basano principalmente sul meccanismo di attention.
B) Falso - A differenza delle RNN, i Transformer processano l'intera sequenza in parallelo, non elemento per elemento.
C) Vero - Il meccanismo di self-attention è il cuore dell'architettura Transformer.
D) Falso - I Transformer in realtà brillano con grandi quantità di dati, e i modelli più potenti richiedono enormi dataset di addestramento.

Dopo aver raccolto le risposte, posso chiedere: "Perché pensate che il meccanismo di self-attention sia così importante per l'elaborazione del linguaggio naturale?"
-->
---

## Componenti principali dell'architettura Transformer 🧱

![bg right:50% 90%](slides/transformer/images/encoder_decoder_structure.avif)

- **Encoder**: comprende il contesto di input
- **Decoder**: genera output sequenziale
- **Self-Attention**: il cuore del modello
- **Feed-Forward Networks**: elaborazione per posizione
- **Layer Normalization e Residual Connections**: stabilità

<!-- 
L'architettura Transformer originale è composta da due parti principali: l'encoder e il decoder.

L'encoder comprende il contesto di input, elaborando l'intera sequenza e creando rappresentazioni contestuali per ogni token. È utilizzato principalmente in modelli come BERT.

Il decoder genera output sequenziale in modo autoregressive (una parola alla volta), utilizzando sia l'attenzione sui propri output precedenti che l'attenzione sull'output dell'encoder. È la base di modelli come GPT.

Il meccanismo di self-attention è il cuore del Transformer, permettendo a ogni posizione di interagire con tutte le altre posizioni.

Le reti feed-forward vengono applicate indipendentemente a ciascuna posizione dopo l'attention, aggiungendo non-linearità e aumentando la capacità rappresentativa.

Layer normalization e residual connections sono tecniche che facilitano l'addestramento di reti profonde, migliorando la stabilità e la velocità di convergenza.

Domanda per la classe: Quali di questi componenti pensate sia più cruciale per il successo dei Transformer, e perché?
-->

---

## Componenti principali dell'architettura Transformer 🧱


| Componente             | Descrizione                                     | Scopo                                                        |
|------------------------|-------------------------------------------------|--------------------------------------------------------------|
| Attenzione Multi-Head  | Meccanismo per concentrarsi su diverse parti dell'input | Cattura le dipendenze tra diverse posizioni nella sequenza    |
| Reti Feed-Forward      | Strati completamente connessi per posizione     | Trasforma gli output dell'attenzione, aggiungendo complessità |
| Codifica Posizionale   | Aggiunge informazioni posizionali agli embedding | Fornisce il contesto dell'ordine della sequenza al modello    |
| Normalizzazione Layer  | Normalizza gli input di ogni sotto-strato       | Stabilizza l'addestramento, migliora la convergenza           |
| Connessioni Residue    | Scorciatoie tra gli strati                      | Aiuta nell'addestramento di reti più profonde riducendo i problemi di gradiente |
| Dropout                | Azzera casualmente alcune connessioni della rete | Previene l'overfitting regolarizzando il modello              |


---

## Embedding

![alt text](images/embedding.avif)

---

## Self-Attention: Il cuore del Transformer ❤️

![bg right:40% 90%](images/Multi_Head_Attention.avif)

### Query, Key e Value

Per ogni elemento della sequenza:
- **Query (Q)**: cosa l'elemento "sta cercando"
- **Key (K)**: cosa l'elemento "offre" agli altri
- **Value (V)**: contenuto informativo dell'elemento

<!-- 
Il meccanismo di self-attention è il vero cuore pulsante dell'architettura Transformer. Permette al modello di pesare l'importanza di diverse parole in una sequenza quando elabora una parola specifica.

Per ogni elemento della sequenza, vengono calcolati tre vettori attraverso trasformazioni lineari dell'input:

La Query rappresenta ciò che l'elemento "sta cercando" - potete pensarla come una domanda che l'elemento pone al resto della sequenza.

La Key rappresenta ciò che l'elemento "offre" agli altri - è come un'etichetta che descrive il contenuto dell'elemento.

Il Value rappresenta il contenuto informativo effettivo dell'elemento - è ciò che viene effettivamente trasmesso quando l'attenzione si concentra su quell'elemento.

Un'analogia utile è quella di una biblioteca: la Query è la richiesta di un libro, la Key è il titolo o l'indice del libro, e il Value è il contenuto effettivo del libro.

Domanda per stimolare la discussione: Come pensate che questo meccanismo aiuti il modello a catturare relazioni complesse tra parole distanti in una frase?
-->

---

## Calcolo dell'attenzione in 4 passaggi 🧮

1. **Calcolo dei punteggi**: `Scores = Q * K^T`
2. **Scaling**: `Scores_scaled = Scores / √d_k`
3. **Softmax**: `Weights = softmax(Scores_scaled)`
4. **Aggregazione pesata**: `Output = Weights * V`

![bg right:40% 90%](slides/transformer/images/softmax_adjusted_scores.avif)

<!-- 
Il calcolo dell'attenzione avviene in quattro passaggi fondamentali:

Primo, calcoliamo i punteggi di attenzione moltiplicando ogni query per tutte le key (trasposta). Questo ci dice quanto ogni elemento dovrebbe prestare attenzione a tutti gli altri elementi.

Secondo, scaliamo questi punteggi dividendo per la radice quadrata della dimensione delle key. Questo è un trucco importante per stabilizzare i gradienti durante l'addestramento, evitando che i valori diventino troppo grandi.

Terzo, applichiamo la funzione softmax per normalizzare i punteggi, trasformandoli in pesi di attenzione che sommano a 1. Questo ci dà una distribuzione di probabilità su tutti gli elementi della sequenza.

Infine, aggreghiamo i valori (V) pesandoli secondo i pesi di attenzione calcolati. Questo ci dà l'output finale dell'operazione di attenzione.

In forma matriciale, l'intera operazione può essere espressa come:
Attention(Q, K, V) = softmax(Q * K^T / √d_k) * V

Domanda tecnica per la classe: Perché pensate sia necessario lo scaling dividendo per la radice quadrata della dimensione? Cosa succederebbe senza questo passaggio?

Il Multi-Head Attention è un'estensione potente del meccanismo di self-attention. Invece di eseguire una singola operazione di attenzione, il Transformer utilizza multiple "teste" di attenzione in parallelo.

Ecco come funziona:
1. Le matrici Q, K e V vengono proiettate h volte (tipicamente 8 o 16) con diverse matrici di proiezione
2. Per ogni "testa", viene calcolata l'attenzione separatamente
3. Gli output delle diverse teste vengono concatenati
4. Il risultato concatenato viene proiettato linearmente per ottenere l'output finale

Questo approccio permette al modello di catturare diverse relazioni e pattern contemporaneamente. Ogni testa può specializzarsi nell'identificare diversi tipi di relazioni: alcune potrebbero focalizzarsi sulla sintassi, altre sulla semantica, altre ancora su relazioni a lungo termine.

È come avere più persone che leggono lo stesso testo, ognuna concentrandosi su aspetti diversi, e poi combinando le loro intuizioni.

Domanda per la classe: Quali tipi diversi di relazioni linguistiche pensate che le diverse teste di attenzione potrebbero imparare a riconoscere?
-->

---

## Positional Encoding: Aggiungere informazioni sulla posizione 📍

![bg right 90%](images/positional-encoding.avif)


<!-- 
Poiché il Transformer processa l'intera sequenza in parallelo, non ha una nozione intrinseca dell'ordine delle parole. Per risolvere questo problema, vengono aggiunti encoding posizionali agli embedding di input.

La formulazione originale utilizza funzioni sinusoidali di diverse frequenze. Per ogni posizione nella sequenza e ogni dimensione nell'embedding, viene calcolato un valore utilizzando funzioni seno e coseno.

Queste funzioni sono state scelte per alcune proprietà interessanti:
- Permettono al modello di generalizzare a sequenze più lunghe di quelle viste durante l'addestramento
- Mantengono informazioni sulla distanza relativa tra posizioni
- Hanno un pattern che il modello può imparare a interpretare

Nel grafico, possiamo vedere come ogni riga rappresenti una dimensione dell'encoding posizionale, e come i valori cambino in modo regolare ma distintivo per ogni posizione.

Domanda per stimolare la riflessione: Perché pensate che sia importante utilizzare funzioni con diverse frequenze per l'encoding posizionale? Cosa succederebbe se usassimo semplicemente numeri sequenziali?
-->

---

## Feed-Forward Networks e Layer Normalization 🔄

![bg right 70%](images/encoder.avif)

```python
# Feed-Forward Network
FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2

# Layer Normalization
LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
```

<!-- 
Dopo ogni blocco di attenzione, il Transformer applica una rete feed-forward a ciascuna posizione indipendentemente. Questa è essenzialmente una trasformazione a due strati con attivazione ReLU (o GELU nei modelli più recenti).

La rete feed-forward introduce non-linearità nel modello, aumenta la capacità rappresentativa e permette trasformazioni specifiche per posizione. Potete pensarla come un elaborazione "verticale" dopo l'elaborazione "orizzontale" dell'attenzione.

Per facilitare l'addestramento di reti profonde, il Transformer utilizza due tecniche importanti:

Layer Normalization: Normalizza gli input a ciascun sub-layer, calcolando media e varianza per ogni esempio individualmente. Questo aiuta a stabilizzare l'addestramento.

Residual Connections: Aggiungono l'input di un sub-layer al suo output (x + Sublayer(x)). Questo permette ai gradienti di fluire più facilmente attraverso la rete durante il backpropagation, combattendo il problema del vanishing gradient.

Domanda tecnica: Quali vantaggi offre la Layer Normalization rispetto alla Batch Normalization in questo contesto di elaborazione sequenziale?
-->

---

## Architettura del Decoder (utilizzata in GPT) 🧠

![bg right:40% 90%](slides/transformer/images/decoder_structure.png)

- **Masked Self-Attention**: vede solo le posizioni precedenti
- **Stack di blocchi identici**:
  - Masked multi-head self-attention
  - Feed-forward network
  - Layer normalization e residual connections
- **Linear Layer e Softmax finale**

<!-- 
I modelli GPT utilizzano solo la parte decoder del Transformer, con alcune modifiche. L'architettura del decoder include:

Masked Self-Attention: A differenza dell'encoder, il decoder utilizza masked self-attention, dove ogni posizione può attendere solo a posizioni precedenti (non future). Questo è essenziale per i modelli autoregressive come GPT, che generano testo una parola alla volta.

Stack di blocchi identici: Ogni blocco contiene masked multi-head self-attention, feed-forward network, layer normalization e residual connections. Nei modelli GPT moderni, questi blocchi possono essere molto numerosi (fino a 96 o più nei modelli più grandi).

Linear Layer e Softmax finale: Trasforma le rappresentazioni finali in distribuzioni di probabilità sul vocabolario, permettendo al modello di predire la parola successiva.

La caratteristica chiave del decoder è la sua natura autoregressive: durante l'addestramento, predice la parola successiva dato il contesto precedente, e durante la generazione, utilizza le parole già generate come contesto per predire la prossima.

Domanda per la classe: Perché pensate che sia necessario mascherare l'attenzione nel decoder per impedire di vedere le posizioni future? Cosa succederebbe se non lo facessimo?
-->

---

## Implementazione di un modello GPT from scratch 💻

![bg right:30% 90%](https://pytorch.org/assets/images/pytorch-logo.png)

Obiettivi:
- Comprendere ogni componente dell'architettura
- Implementare un modello GPT semplificato
- Utilizzare PyTorch per l'implementazione

<!-- 
Nella seconda parte della presentazione, passeremo all'implementazione pratica di un modello GPT semplificato utilizzando PyTorch. Questo ci permetterà di comprendere in modo concreto come funziona l'architettura.

L'obiettivo non è creare un modello pronto per la produzione, ma piuttosto fornire una comprensione pratica di come funziona l'architettura. Implementeremo ogni componente del modello, dalla self-attention alla generazione di testo.

Utilizzeremo PyTorch, una delle librerie più popolari per il deep learning, che offre un'API flessibile e intuitiva per la costruzione di modelli neurali.

Questa implementazione ci permetterà di vedere come i concetti teorici che abbiamo discusso si traducono in codice concreto, e come i diversi componenti interagiscono tra loro per formare un modello completo.

Domanda per la classe: Chi ha già esperienza con PyTorch? Chi ha già implementato modelli di deep learning da zero?
-->

---

## Setup e configurazione del modello 🛠️

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTConfig:
    def __init__(self, vocab_size, block_size, n_embd=768, 
                 n_layer=12, n_head=12, dropout=0.1):
        self.vocab_size = vocab_size  # dimensione del vocabolario
        self.block_size = block_size  # lunghezza massima della sequenza
        self.n_embd = n_embd          # dimensione degli embedding
        self.n_layer = n_layer        # numero di blocchi del decoder
        self.n_head = n_head          # numero di teste di attenzione
        self.dropout = dropout        # probabilità di dropout
```

<!-- 
Iniziamo con le importazioni necessarie e la definizione della classe di configurazione del nostro modello GPT.

Importiamo math per alcune operazioni matematiche, torch come libreria principale, nn per i moduli neurali e F per le funzioni di attivazione e altre utility.

La classe GPTConfig ci permette di definire tutti i parametri principali del nostro modello:
- vocab_size: la dimensione del vocabolario, ovvero quante parole o token diversi il modello può riconoscere
- block_size: la lunghezza massima della sequenza che il modello può elaborare
- n_embd: la dimensione degli embedding, ovvero quanto sono "ricche" le rappresentazioni vettoriali
- n_layer: il numero di blocchi del decoder impilati uno sull'altro
- n_head: il numero di teste di attenzione in ogni blocco
- dropout: la probabilità di dropout, una tecnica di regolarizzazione

Questi parametri determinano la capacità e le dimensioni del modello. Per riferimento, GPT-2 Small ha 12 layer, 12 teste e 768 dimensioni di embedding, mentre GPT-3 arriva a 96 layer, 96 teste e 12288 dimensioni.

Domanda tecnica: Come pensate che la scelta di questi iperparametri influenzi le capacità e le prestazioni del modello?
-->

---

## Implementazione del Self-Attention 👁️

```python
class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # causal mask
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, 
                             config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd
```

<!-- 
Qui implementiamo la classe SelfAttention, che è il cuore del nostro modello GPT.

Iniziamo verificando che la dimensione degli embedding sia divisibile per il numero di teste, in modo da poter dividere equamente le dimensioni tra le teste.

Creiamo tre trasformazioni lineari per calcolare key, query e value a partire dall'input. Queste sono semplicemente matrici di pesi che trasformano l'input in tre diversi spazi vettoriali.

Aggiungiamo anche una proiezione di output che combinerà i risultati delle diverse teste di attenzione.

Per la regolarizzazione, utilizziamo due layer di dropout: uno per i pesi di attenzione e uno per l'output residuale.

Un elemento cruciale è la maschera causale (causal mask), che assicura che ogni posizione possa vedere solo le posizioni precedenti. Questa è implementata come una matrice triangolare inferiore di uni.

Infine, memorizziamo il numero di teste e la dimensione degli embedding per uso futuro.

Domanda per approfondire: Perché è importante che la dimensione degli embedding sia divisibile per il numero di teste? Come influisce questo sulla struttura del multi-head attention?
-->
---

## Implementazione del forward pass del Self-Attention 🔄

```python
def forward(self, x):
    B, T, C = x.size() # batch size, sequence length, embedding dimensionality
    
    # calculate query, key, values for all heads in batch
    k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    
    # causal self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    att = self.attn_dropout(att)
    y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    
    # output projection
    y = self.resid_dropout(self.proj(y))
    return y
```

<!-- 
Qui implementiamo il metodo forward del nostro modulo SelfAttention, che definisce come l'informazione fluisce attraverso questo componente.

Iniziamo estraendo le dimensioni dell'input: B è la dimensione del batch, T è la lunghezza della sequenza, e C è la dimensionalità degli embedding.

Calcoliamo key, query e value per tutte le teste in parallelo. Notiamo la manipolazione delle dimensioni: dividiamo C in n_head parti uguali, e riorganizziamo il tensore per separare le teste.

Poi calcoliamo i punteggi di attenzione moltiplicando q per k trasposto. Scaliamo dividendo per la radice quadrata della dimensione delle key, come discusso in precedenza.

Applichiamo la maschera causale, sostituendo con -infinito i valori che corrispondono a posizioni future, in modo che dopo il softmax diventino zero.

Applichiamo softmax per normalizzare i punteggi in pesi di attenzione, seguita da dropout per regolarizzazione.

Moltiplichiamo i pesi per i valori per ottenere l'output pesato, e riorganizziamo il tensore per combinare i risultati di tutte le teste.

Infine, applichiamo una proiezione lineare e dropout all'output.

Domanda tecnica: Cosa succede esattamente quando sostituiamo con -infinito i valori mascherati? Perché questo è necessario per implementare l'attenzione causale?
-->

---

## Implementazione del Feed-Forward Network 🔄

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # Utilizziamo GELU invece di ReLU come in GPT
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)
```

<!-- 
Qui implementiamo il modulo FeedForward, che è la rete feed-forward applicata dopo ogni blocco di attenzione.

La struttura è semplice: una rete sequenziale con due trasformazioni lineari e un'attivazione non lineare nel mezzo.

La prima trasformazione espande la dimensionalità da n_embd a 4*n_embd, introducendo più parametri e capacità rappresentativa.

Utilizziamo l'attivazione GELU (Gaussian Error Linear Unit) invece di ReLU. GELU è stata introdotta più recentemente e ha dimostrato di funzionare meglio in molti modelli di linguaggio, inclusi i modelli GPT.

La seconda trasformazione riporta la dimensionalità a n_embd, e infine applichiamo dropout per regolarizzazione.

Questa struttura a "collo di bottiglia" (espansione seguita da compressione) è comune in molte architetture di deep learning e permette alla rete di apprendere trasformazioni complesse.

Domanda per la classe: Perché pensate che si utilizzi un fattore di espansione di 4 nella prima trasformazione lineare? Come potrebbe influire questo sulla capacità del modello?
-->

---

## Implementazione di un blocco del Transformer 🧱

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x):
        # Utilizziamo la normalizzazione pre-layer come in GPT-2
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
```

<!-- 
Qui combiniamo self-attention e feed-forward in un blocco completo del Transformer.

Ogni blocco contiene:
- Un layer di normalizzazione seguito da self-attention
- Un altro layer di normalizzazione seguito da feed-forward
- Connessioni residuali attorno a entrambi i sub-layer

Notiamo che utilizziamo la normalizzazione pre-layer (applicando layer norm prima di ogni sub-layer) invece della normalizzazione post-layer utilizzata nel paper originale. Questa è una modifica introdotta in GPT-2 che ha dimostrato di migliorare la stabilità dell'addestramento.

Le connessioni residuali (x + ...) sono cruciali per l'addestramento di reti profonde, permettendo ai gradienti di fluire più facilmente attraverso la rete.

Questo blocco è l'unità fondamentale che verrà ripetuta più volte per formare il modello completo. Nei modelli GPT moderni, possono esserci decine o centinaia di questi blocchi impilati.

Domanda per stimolare la riflessione: Perché pensate che la normalizzazione pre-layer funzioni meglio della normalizzazione post-layer nei modelli profondi? Quali vantaggi potrebbe offrire durante l'addestramento?
-->

---

## Implementazione del modello GPT completo (Parte 1) 🏗️

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # input embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        
        # transformer blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        
        # final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # language modeling head
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # initialize weights
        self.apply(self._init_weights)
```

<!-- 
Qui iniziamo l'implementazione del modello GPT completo.

Iniziamo definendo gli embedding:
- tok_emb: l'embedding dei token, che mappa ogni token del vocabolario a un vettore di dimensione n_embd
- pos_emb: l'embedding posizionale, che viene appreso durante l'addestramento invece di utilizzare le funzioni sinusoidali fisse del paper originale
- drop: un layer di dropout applicato dopo la somma degli embedding

Poi definiamo i blocchi del transformer, creando n_layer istanze del blocco che abbiamo definito prima e combinandole in un modulo Sequential.

Aggiungiamo un layer di normalizzazione finale dopo tutti i blocchi.

Infine, definiamo la "testa" di language modeling, che è semplicemente una trasformazione lineare che mappa le rappresentazioni finali alle probabilità sul vocabolario.

Notiamo anche che includiamo un metodo per inizializzare i pesi del modello, che verrà applicato a tutti i sotto-moduli.

Domanda per la classe: Perché pensate che in GPT si utilizzi un embedding posizionale appreso invece delle funzioni sinusoidali fisse proposte nel paper originale? Quali vantaggi e svantaggi potrebbe avere questo approccio?
-->

---

## Implementazione del modello GPT completo (Parte 2) 🏗️

```python
def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def forward(self, idx, targets=None):
    B, T = idx.size()
    assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
    
    # forward the GPT model
    token_embeddings = self.tok_emb(idx) # (B, T, n_embd)
    position_embeddings = self.pos_emb[:, :T, :] # (1, T, n_embd)
    x = self.drop(token_embeddings + position_embeddings)
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.head(x) # (B, T, vocab_size)
    
    # if we are given some desired targets also calculate the loss
    loss = None
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    return logits, loss
```

<!-- 
Continuiamo con l'implementazione del modello GPT, definendo il metodo di inizializzazione dei pesi e il forward pass.

Il metodo _init_weights inizializza i pesi delle trasformazioni lineari e degli embedding con una distribuzione normale con media 0 e deviazione standard 0.02, mentre i bias vengono inizializzati a zero. Per i layer di normalizzazione, i pesi vengono inizializzati a 1 e i bias a 0.

Nel metodo forward, prima verifichiamo che la lunghezza della sequenza non superi il block_size configurato.

Poi calcoliamo gli embedding dei token e gli embedding posizionali, li sommiamo e applichiamo dropout.

Facciamo passare il risultato attraverso tutti i blocchi del transformer, seguiti dal layer di normalizzazione finale.

Infine, applichiamo la testa di language modeling per ottenere i logits, che rappresentano le probabilità non normalizzate per ogni token del vocabolario.

Se vengono forniti dei target, calcoliamo anche la loss utilizzando la cross-entropy, che è la funzione di perdita standard per i problemi di classificazione multi-classe come il language modeling.

Domanda tecnica: Perché utilizziamo la cross-entropy come funzione di perdita per il language modeling? Come si collega questa scelta alla natura probabilistica del task?
-->

---

## Implementazione della generazione di testo 📝

```python
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Generate new tokens beyond the context provided in idx.
    
    Args:
        idx: (B, T) tensor of indices in the current context
        max_new_tokens: number of new tokens to generate
        temperature: softmax temperature, higher values increase diversity
        top_k: if set, only sample from the top k most probable tokens
    
    Returns:
        (B, T+max_new_tokens) tensor of indices
    """
    for _ in range(max_new_tokens):
        # crop context if needed
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        # forward pass
        logits, _ = self.forward(idx_cond)
        # focus on the last time step
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)
            
    return idx
```

<!-- 
Qui implementiamo il metodo generate, che ci permette di utilizzare il modello per generare nuovo testo in modo autoregressive.

Il metodo prende in input:
- idx: un tensore di indici che rappresenta il contesto iniziale
- max_new_tokens: il numero di nuovi token da generare
- temperature: un parametro che controlla la "creatività" della generazione (valori più alti aumentano la diversità)
- top_k: se specificato, limita il campionamento ai k token più probabili

Il processo di generazione è iterativo:
1. Se il contesto è troppo lungo, lo tagliamo per rispettare il block_size
2. Facciamo un forward pass per ottenere le probabilità di tutti i token
3. Ci concentriamo sull'ultimo time step, che rappresenta la predizione per il token successivo
4. Applichiamo la temperature scaling per controllare la diversità
5. Se specificato, limitiamo le probabilità ai top_k token
6. Convertiamo i logits in probabilità con softmax
7. Campioniamo il prossimo token dalla distribuzione di probabilità
8. Aggiungiamo il token generato alla sequenza e ripetiamo

Questo approccio autoregressive è il cuore dei modelli generativi come GPT: ogni token generato diventa parte del contesto per generare il token successivo.

Domanda per stimolare la discussione: Come pensate che i parametri temperature e top_k influenzino la qualità e la diversità del testo generato? Quali valori usereste per diverse applicazioni?
-->

---

## Esempio di utilizzo del modello 🚀

```python
# Esempio di configurazione per un modello piccolo
config = GPTConfig(
    vocab_size=50257,  # Dimensione del vocabolario GPT-2
    block_size=1024,   # Lunghezza massima della sequenza
    n_embd=768,        # Dimensione degli embedding
    n_layer=12,        # Numero di blocchi
    n_head=12,         # Numero di teste di attenzione
    dropout=0.1        # Dropout
)

# Inizializzazione del modello
model = GPT(config)

# Esempio di input (batch di 2 sequenze di 10 token ciascuna)
x = torch.randint(0, config.vocab_size, (2, 10))
targets = torch.randint(0, config.vocab_size, (2, 10))

# Forward pass
logits, loss = model(x, targets)
print(f"Input shape: {x.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Loss: {loss.item()}")

# Generazione di testo
context = torch.tensor([[464, 1842, 2181, 1447, 373]])  # Esempio di contesto iniziale
generated = model.generate(context, max_new_tokens=20, temperature=0.8, top_k=40)
print(f"Generated sequence shape: {generated.shape}")
```

<!-- 
Qui mostriamo un esempio di come utilizzare il nostro modello GPT implementato.

Iniziamo definendo una configurazione per un modello relativamente piccolo, simile a GPT-2 Small, con:
- Un vocabolario di 50257 token (come GPT-2)
- Una lunghezza massima di sequenza di 1024 token
- Embedding di dimensione 768
- 12 blocchi del transformer
- 12 teste di attenzione per blocco
- Un dropout di 0.1 per regolarizzazione

Poi inizializziamo il modello con questa configurazione.

Per testare il forward pass, creiamo un batch di input e target casuali e li passiamo al modello, ottenendo i logits e la loss.

Infine, mostriamo come generare nuovo testo a partire da un contesto iniziale, specificando:
- Il numero di nuovi token da generare (20)
- La temperatura (0.8, leggermente più creativa del default)
- Il parametro top_k (40, come in molte implementazioni di GPT-2)

Questo esempio mostra l'intero flusso di utilizzo del modello, dall'inizializzazione alla generazione di testo.

Domanda per la classe: Quali applicazioni pratiche potreste immaginare per un modello GPT come quello che abbiamo implementato? Come potrebbe essere adattato per task specifici?
-->

---

## Addestramento del modello 🏋️‍♂️

```python
def train(model, data_loader, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits, loss = model(x, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
```

<!-- 
Qui implementiamo una semplice funzione di addestramento per il nostro modello GPT.

La funzione prende in input:
- Il modello da addestrare
- Un data loader che fornisce batch di dati
- Un ottimizzatore (tipicamente Adam o AdamW)
- Il numero di epoche di addestramento
- Il dispositivo su cui eseguire l'addestramento (CPU o GPU)

Il processo di addestramento è standard:
1. Mettiamo il modello in modalità di addestramento
2. Per ogni epoca, iteriamo sui batch di dati
3. Spostiamo i dati sul dispositivo appropriato
4. Azzeriamo i gradienti dell'ottimizzatore
5. Facciamo un forward pass per ottenere i logits e la loss
6. Facciamo un backward pass per calcolare i gradienti
7. Aggiorniamo i pesi con l'ottimizzatore
8. Teniamo traccia della loss e stampiamo aggiornamenti periodici

Questo è un loop di addestramento molto basilare. In pratica, l'addestramento di modelli GPT di grandi dimensioni richiede molte ottimizzazioni aggiuntive come mixed precision training, gradient accumulation, learning rate scheduling, e altro ancora.

Domanda per approfondire: Quali sfide pensate si incontrino nell'addestrare modelli GPT di grandi dimensioni? Come si potrebbero affrontare queste sfide?
-->

---

## Ottimizzazioni e considerazioni pratiche 🔧

1. **Efficienza computazionale**:
   - Mixed precision training
   - Gradient accumulation
   - Model parallelism

2. **Tokenizzazione**: Byte-Pair Encoding (BPE)

3. **Strategie di addestramento**:
   - Learning rate scheduling
   - Weight decay
   - Gradient clipping

4. **Generazione di testo**:
   - Beam search
   - Nucleus sampling
   - Temperature scaling

<!-- 
Nella pratica, l'implementazione di un modello GPT completo richiederebbe diverse ottimizzazioni e considerazioni aggiuntive.

Per l'efficienza computazionale, tecniche come mixed precision training (utilizzando float16 invece di float32), gradient accumulation (accumulando gradienti su più batch prima di aggiornare i pesi), e model parallelism (distribuendo il modello su più GPU) sono essenziali per addestrare modelli di grandi dimensioni.

La tokenizzazione è un aspetto cruciale. Modelli come GPT-2 e GPT-3 utilizzano Byte-Pair Encoding (BPE), che bilancia efficacemente la rappresentazione di parole comuni e rare, creando un vocabolario di subword tokens.

Per quanto riguarda le strategie di addestramento, tecniche come learning rate scheduling (variare il learning rate durante l'addestramento), weight decay (regolarizzazione L2), e gradient clipping (limitare la norma dei gradienti) sono importanti per la stabilità dell'addestramento.

Infine, per la generazione di testo, strategie avanzate come beam search (mantenere multiple ipotesi di generazione), nucleus sampling (campionare dalla distribuzione troncata ai token più probabili), e temperature scaling (controllare la diversità) possono migliorare significativamente la qualità del testo generato.

Domanda per stimolare la riflessione: Quali di queste ottimizzazioni pensate siano più critiche per il successo pratico dei modelli GPT? Perché?
-->

---

## Confronto tra BERT e GPT 🔍

| Caratteristica | BERT | GPT |
|----------------|------|-----|
| **Architettura** | Encoder | Decoder |
| **Direzionalità** | Bidirezionale | Unidirezionale (left-to-right) |
| **Pre-addestramento** | Masked Language Modeling (MLM) | Predizione della parola successiva |
| **Applicazioni tipiche** | Comprensione (classificazione, NER, QA) | Generazione (completamento, traduzione, riassunto) |

<!-- 
Per completare la nostra comprensione, è utile confrontare le architetture BERT e GPT, che rappresentano due approcci fondamentali ai modelli Transformer.

BERT utilizza principalmente la parte encoder del Transformer e considera il contesto in entrambe le direzioni (precedente e successivo). Durante il pre-addestramento, utilizza Masked Language Modeling (MLM), dove alcune parole vengono mascherate e il modello deve predirle, e Next Sentence Prediction (NSP). BERT eccelle in compiti di comprensione del linguaggio.

GPT utilizza principalmente la parte decoder del Transformer e considera solo il contesto precedente (left-to-right). Durante il pre-addestramento, predice semplicemente la parola successiva (language modeling classico). GPT eccelle in compiti generativi.

Le principali differenze possono essere riassunte così:
1. Direzione del contesto: BERT è bidirezionale, GPT è unidirezionale
2. Obiettivo di pre-addestramento: BERT predice parole mascherate, GPT predice la parola successiva
3. Uso tipico: BERT per comprensione, GPT per generazione
4. Architettura: BERT usa l'encoder, GPT usa il decoder

Entrambi gli approcci hanno i loro punti di forza e sono complementari in molte applicazioni NLP.

Domanda per la classe: In quali scenari applicativi pensate sia più appropriato utilizzare BERT rispetto a GPT, e viceversa? Ci sono casi in cui potrebbero essere utilizzati insieme?
-->

---

## Domande 🤔

![bg right:40% 80%](https://media.giphy.com/media/3o7buirYcmV5nSwIRW/giphy.gif)

<!-- 
Questo è un momento per aprire la discussione e rispondere a qualsiasi domanda che la classe potrebbe avere sull'architettura Transformer o sull'implementazione di GPT.

Alcune possibili domande da anticipare:
- Come si confrontano i Transformer con altre architetture come LSTM in termini di prestazioni e efficienza?
- Quali sono i limiti principali dei modelli Transformer attuali?
- Come si può adattare un modello pre-addestrato come GPT a un task specifico?
- Quali sono le considerazioni etiche nell'utilizzo di modelli generativi potenti come GPT?
- Come si può interpretare o visualizzare cosa ha appreso un modello Transformer?

Questo è anche un buon momento per collegare i concetti teorici che abbiamo discusso con applicazioni pratiche e casi d'uso reali, incoraggiando gli studenti a pensare a come potrebbero utilizzare queste tecnologie nei loro progetti.
-->

---

## Conclusione 🎯

- L'architettura Transformer ha rivoluzionato l'NLP grazie al meccanismo di self-attention
- I modelli GPT utilizzano la parte decoder del Transformer per generazione di testo
- L'implementazione from scratch ci ha permesso di comprendere ogni componente
- Questi modelli sono alla base di sistemi come ChatGPT e altri LLM

### Riferimenti
- Vaswani et al. (2017). "Attention is All You Need"
- Radford et al. (2018). "Improving Language Understanding by Generative Pre-Training"
- Radford et al. (2019). "Language Models are Unsupervised Multitask Learners"

<!-- 
In questa presentazione, abbiamo esplorato in dettaglio l'architettura Transformer, con particolare attenzione ai modelli generativi come GPT.

Abbiamo visto come il meccanismo di self-attention permetta ai modelli di catturare dipendenze a lungo termine in modo efficiente, e come l'architettura complessiva faciliti l'addestramento di modelli profondi attraverso tecniche come layer normalization e residual connections.

Abbiamo anche implementato un modello GPT semplificato from scratch utilizzando PyTorch, esplorando ogni componente dell'architettura e come questi componenti interagiscono per creare un potente modello linguistico generativo.

Questa comprensione dettagliata dell'architettura Transformer e della sua implementazione fornisce una base solida per lavorare con modelli linguistici moderni, sia utilizzando implementazioni esistenti che sviluppando soluzioni personalizzate per applicazioni specifiche.

I modelli che abbiamo discusso oggi sono alla base di sistemi come ChatGPT, Bard, Claude e altri Large Language Models che stanno trasformando il modo in cui interagiamo con la tecnologia.

Domanda finale per stimolare la riflessione: Come pensate che questi modelli evolveranno nei prossimi anni? Quali nuove applicazioni o capacità potrebbero emergere?
-->

---

## Hai mai pensato... 🤔

Se un modello GPT potesse scrivere il suo stesso codice, come si implementerebbe?

<!-- 
Questa è una slide leggera e divertente per stimolare la riflessione e alleggerire l'atmosfera dopo la parte tecnica della presentazione.

La domanda è volutamente provocatoria e filosofica: se un modello GPT potesse scrivere il suo stesso codice, come si implementerebbe? Questo apre a riflessioni interessanti su ricorsione, auto-miglioramento e i limiti dell'intelligenza artificiale.

Potrei sviluppare questo concetto dicendo: "In effetti, oggi i modelli come GPT-4 sono già in grado di generare codice complesso, incluso codice per implementare reti neurali. Stiamo entrando in un'era in cui l'IA può contribuire al proprio sviluppo. Questo solleva domande affascinanti sul futuro dell'intelligenza artificiale e sul ruolo degli sviluppatori umani."

Potrei anche collegare questo alla pratica attuale: "Molti sviluppatori oggi usano già modelli come GitHub Copilot o ChatGPT per aiutarli a scrivere codice. Quanto tempo passerà prima che questi strumenti possano generare implementazioni complete e ottimizzate di modelli complessi come i Transformer?"

Questa slide può generare una discussione interessante sui limiti e le possibilità future dell'IA generativa.
-->

---

## Risorse per approfondire 📚

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Jay Alammar
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Harvard NLP
- [minGPT](https://github.com/karpathy/minGPT) - Andrej Karpathy
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) - Documentazione
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Paper originale

<!-- 
Questa slide fornisce risorse utili per gli studenti che vogliono approfondire l'argomento dopo la lezione.

"The Illustrated Transformer" di Jay Alammar è una risorsa eccellente che spiega visivamente il funzionamento dell'architettura Transformer.

"The Annotated Transformer" di Harvard NLP implementa il Transformer originale con spiegazioni dettagliate.

"minGPT" di Andrej Karpathy è un'implementazione minimalista di GPT in PyTorch, simile a quella che abbiamo visto ma con alcune ottimizzazioni aggiuntive.

La documentazione di Hugging Face Transformers è una risorsa preziosa per chi vuole utilizzare implementazioni pronte all'uso di modelli Transformer.

Il paper originale "Attention Is All You Need" è fondamentale per comprendere le motivazioni e i dettagli dell'architettura.

Potrei aggiungere: "Queste risorse vi permetteranno di approfondire gli aspetti che vi interessano di più e di esplorare implementazioni alternative e ottimizzazioni avanzate. Vi consiglio particolarmente The Illustrated Transformer se preferite spiegazioni visive, e minGPT se volete vedere un'implementazione pulita e ben documentata."
-->