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

<!-- MARP slide deck: Transformer e GPT - Appendice Corso NLP -->

# Transformer e GPT: Dalla teoria all'implementazione 🚀

![bg right:40% 80%](images/transformer_blackbox.avif)

**Obiettivi della presentazione:**
- Capire i concetti chiave dei Transformer
- Analizzare i componenti fondamentali dell'architettura
- Approfondire l'architettura GPT e la sua implementazione
- Vedere esempi pratici di codice e ottimizzazioni
- Confrontare BERT e GPT e discutere applicazioni

> Chi ha già usato modelli come ChatGPT o BERT? In che contesti?

---

## Perché i Transformer hanno rivoluzionato l'NLP?

- Elaborazione **parallela** delle sequenze (non più sequenziale come le RNN)
- **Self-attention**: relazioni dirette tra tutte le parole della sequenza
- Encoding posizionale esplicito per mantenere l’ordine
- **Scalabilità**: modelli sempre più grandi e potenti

> Domanda: Quali problemi delle RNN vengono superati dai Transformer?

---

## Quiz: Cosa rende speciale i Transformer? 🧩

A) Usano solo reti convoluzionali  
B) Processano sequenze elemento per elemento  
C) Si basano sul meccanismo di self-attention  
D) Richiedono meno dati per l’addestramento  

> Qual è la risposta corretta? Perché?
---

## Struttura generale del Transformer

![bg right:50% 90%](images/encoder_decoder_structure.avif)

- **Encoder**: elabora il contesto dell’input (usato in BERT)
- **Decoder**: genera una sequenza in modo autoregressivo (usato in GPT)
- **Self-attention**: permette interazione tra tutte le posizioni
- **Feed-Forward**: trasforma le rappresentazioni posizione per posizione
- **LayerNorm & Residual**: stabilità e profondità

> Secondo voi, qual è il componente più determinante per il successo dei Transformer?

---

## Tabella dei componenti chiave

| Componente              | Descrizione                                   | Scopo principale                                            |
|------------------------ |-----------------------------------------------|-------------------------------------------------------------|
| Multi-Head Attention    | Più "canali" di attenzione                    | Cattura molteplici relazioni tra posizioni                  |
| Feed-Forward Network    | Strati densi per posizione                    | Aumenta la complessità e la capacità del modello            |
| Positional Encoding     | Informazione di posizione                     | Permette di distinguere l’ordine dei token                  |
| Layer Normalization     | Normalizza ogni sotto-strato                  | Stabilizza e accelera l’addestramento                       |
| Residual Connection     | Collegamenti diretti tra strati               | Facilita il flusso del gradiente nelle reti profonde        |
| Dropout                 | Disattiva connessioni in modo casuale         | Riduce l’overfitting                                        |
---

## Embedding: rappresentare i token come vettori

![alt text](images/embedding.avif)

Ogni parola/token viene trasformato in un vettore continuo (embedding) che il modello può elaborare.

---

## Il cuore del Transformer: Self-Attention ❤️

![bg right:40% 90%](images/Multi_Head_Attention.avif)

**Query, Key, Value per ogni token:**
- **Query (Q):** Cosa sta cercando il token?
- **Key (K):** Cosa offre agli altri token?
- **Value (V):** Qual è il contenuto da trasmettere?

> Come aiuta questo meccanismo a cogliere relazioni tra parole anche lontane nella frase?

---

## Calcolo dell'attenzione: i 4 step chiave

1. **Scores = Q × Kᵗ**
2. **Scaling:** Scores / √dₖ
3. **Softmax:** pesi di attenzione
4. **Output:** somma pesata dei Value

![bg right:40% 90%](images/softmax_adjusted_scores.avif)

> Perché dividiamo per √dₖ? Cosa accadrebbe senza scaling?

**Multi-Head Attention:**  
Più "teste" di attenzione in parallelo → ogni testa può imparare relazioni diverse (es. sintattiche, semantiche, a lungo termine).

---

## Positional Encoding: come dare il senso dell’ordine 📍

![bg right 90%](images/positional-encoding.avif)

- Il Transformer non "vede" l’ordine dei token: serve un encoding posizionale.
- Si usano funzioni sinusoidali di diverse frequenze per ogni posizione e dimensione.

> Perché è utile usare funzioni periodiche invece di numeri sequenziali?

---

## Feed-Forward e Layer Normalization

![bg right 70%](images/encoder.avif)

```python
# Feed-Forward Network
FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2

# Layer Normalization
LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
```

- **Feed-Forward:** trasforma ogni posizione in modo indipendente, aggiunge non-linearità.
- **LayerNorm:** normalizza ogni esempio → stabilità anche con sequenze di lunghezza variabile.
- **Residual:** somma input e output di ogni blocco.

> Perché LayerNorm è preferita a BatchNorm nei Transformer?

---

## Decoder Transformer: la base di GPT 🧠

![bg right:40% 90%](images/decoder_structure.png)

- **Masked self-attention:** ogni token vede solo i precedenti
- Stack di blocchi identici: attention, feed-forward, LayerNorm, residual
- Output: layer lineare + softmax (predizione della parola successiva)

> Perché è fondamentale mascherare le posizioni future nel decoder?

---

## Implementazione GPT step-by-step 💻

![bg right:30% 90%](https://pytorch.org/assets/images/pytorch-logo.png)

**Cosa vedremo:**
- Analisi dei componenti chiave in codice PyTorch
- Implementazione di un mini-GPT da zero
- Esempio pratico di forward, loss e generazione

> Chi ha già usato PyTorch o implementato modelli deep learning da zero?

---

## Setup e configurazione: la classe di base

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

## Self-Attention: implementazione in PyTorch

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

## Self-Attention: forward pass

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

## Feed-Forward Network: codice essenziale

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

## Un blocco Transformer: combinare attention e feed-forward

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

## Modello GPT completo: struttura principale

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

## Modello GPT completo: pesi e forward pass

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

## Generazione di testo autoregressiva

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

## Esempio pratico: come usare il modello GPT

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

## Addestrare il modello: ciclo base

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

## Ottimizzazioni pratiche e best practices

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

## BERT vs GPT: differenze chiave

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

## Domande & Discussione finale 🤔

![bg right:40% 80%](https://media.giphy.com/media/3o7buirYcmV5nSwIRW/giphy.gif)

**Hai dubbi sull'architettura o l'implementazione?**
- Differenze con LSTM/CNN?
- Limiti attuali dei Transformer?
- Come adattare GPT a un task specifico?
- Implicazioni etiche dei LLM?
- Come interpretare cosa "vede" un Transformer?

---

## Conclusioni e takeaway 🎯

- Il Transformer ha rivoluzionato l’NLP grazie a self-attention e parallelizzazione
- GPT sfrutta il decoder per la generazione di testo
- L’implementazione da zero aiuta a capire ogni componente
- Questi modelli sono la base di ChatGPT, LLM e molte applicazioni attuali

**Riferimenti principali:**
- Vaswani et al. (2017), "Attention is All You Need"
- Radford et al. (2018, 2019), "Generative Pre-Training"

> Come immagini il futuro dei modelli linguistici? Quali nuove applicazioni potrebbero nascere?

---

## Una domanda per il futuro... 🤔

Se un modello GPT potesse scrivere il suo stesso codice, come si implementerebbe?  

> Riflettiamo: l’IA può contribuire al proprio sviluppo?

---

## Risorse per approfondire 📚

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) (Jay Alammar)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) (Harvard NLP)
- [minGPT](https://github.com/karpathy/minGPT) (Andrej Karpathy)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Attention Is All You Need (paper)](https://arxiv.org/abs/1706.03762)

> Consiglio: partite da "The Illustrated Transformer" per una spiegazione visiva!

---

## Quiz e domande interattive 🎲

**1. Cosa distingue i Transformer dalle RNN?**  
**2. Perché serve il positional encoding?**  
**3. Qual è il ruolo della maschera nel decoder di GPT?**  
**4. In quali task useresti BERT invece di GPT?**  
**5. Come influenzano temperature e top_k la generazione di testo?**

> Discutiamone insieme!