# Appendice: Architettura Transformer e Implementazione di un Modello GPT from Scratch

## Introduzione all'architettura Transformer

L'architettura Transformer, introdotta nel paper "Attention is All You Need" (Vaswani et al., 2017), ha rivoluzionato il campo del Natural Language Processing, superando le limitazioni delle architetture precedenti basate su reti neurali ricorrenti (RNN) e convoluzionali (CNN). In questa appendice, esploreremo in dettaglio il funzionamento dell'architettura Transformer, con particolare attenzione ai modelli generativi come GPT (Generative Pre-trained Transformer), e vedremo come implementare un modello GPT semplificato from scratch utilizzando PyTorch.

L'architettura Transformer si distingue per diverse caratteristiche fondamentali:

1. **Parallelizzazione completa**: A differenza delle RNN che processano sequenze elemento per elemento, il Transformer elabora l'intera sequenza in parallelo, permettendo un addestramento molto più efficiente.

2. **Meccanismo di self-attention**: Permette a ogni posizione nella sequenza di interagire direttamente con tutte le altre posizioni, catturando dipendenze a lungo termine in modo efficace.

3. **Rappresentazioni posizionali**: Poiché non c'è ricorrenza, le informazioni sulla posizione vengono aggiunte esplicitamente attraverso encoding posizionali.

4. **Architettura encoder-decoder**: La versione originale del Transformer include sia un encoder che un decoder, sebbene molti modelli successivi utilizzino solo una delle due componenti (ad esempio, BERT utilizza principalmente l'encoder, mentre GPT utilizza il decoder).

Questa appendice è strutturata in due parti principali:

1. **Comprensione dettagliata dell'architettura Transformer**: Esploreremo ogni componente dell'architettura, con particolare attenzione al meccanismo di self-attention e alla struttura del decoder utilizzata nei modelli GPT.

2. **Implementazione pratica di un modello GPT from scratch**: Svilupperemo passo dopo passo un modello GPT semplificato utilizzando PyTorch, spiegando ogni componente del codice per una comprensione profonda.

## Parte 1: Comprensione dettagliata dell'architettura Transformer

### Il meccanismo di self-attention

Il cuore dell'architettura Transformer è il meccanismo di self-attention, che permette al modello di pesare l'importanza di diverse parole in una sequenza quando elabora una parola specifica.

#### Query, Key e Value

Per ogni elemento della sequenza, vengono calcolati tre vettori attraverso trasformazioni lineari dell'input:

- **Query (Q)**: Rappresenta ciò che l'elemento "sta cercando"
- **Key (K)**: Rappresenta ciò che l'elemento "offre" agli altri
- **Value (V)**: Rappresenta il contenuto informativo dell'elemento

Matematicamente, per un input X:
- Q = X * W_Q
- K = X * W_K
- V = X * W_V

Dove W_Q, W_K e W_V sono matrici di parametri apprendibili.

#### Calcolo dell'attenzione

Il calcolo dell'attenzione avviene in quattro passaggi:

1. **Calcolo dei punteggi di attenzione**: Si calcola il prodotto scalare tra ogni query e tutte le key
   ```
   Scores = Q * K^T
   ```

2. **Scaling**: I punteggi vengono scalati dividendo per la radice quadrata della dimensione delle key (d_k) per stabilizzare i gradienti
   ```
   Scores_scaled = Scores / √d_k
   ```

3. **Softmax**: I punteggi scalati vengono normalizzati attraverso una funzione softmax per ottenere pesi di attenzione che sommano a 1
   ```
   Weights = softmax(Scores_scaled)
   ```

4. **Aggregazione pesata**: I valori (V) vengono pesati secondo i pesi di attenzione
   ```
   Output = Weights * V
   ```

In forma matriciale, l'intera operazione può essere espressa come:
```
Attention(Q, K, V) = softmax(Q * K^T / √d_k) * V
```

#### Multi-Head Attention

Invece di eseguire una singola operazione di attenzione, il Transformer utilizza il multi-head attention, che permette al modello di focalizzarsi su diverse parti dell'input simultaneamente:

1. Le matrici Q, K e V vengono proiettate h volte con diverse matrici di proiezione
2. Per ogni "testa", viene calcolata l'attenzione separatamente
3. Gli output delle diverse teste vengono concatenati
4. Il risultato concatenato viene proiettato linearmente per ottenere l'output finale

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
dove head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)
```

Questo approccio permette al modello di catturare diverse relazioni e pattern contemporaneamente, migliorando significativamente la capacità rappresentativa.

### Positional Encoding

Poiché il Transformer non ha ricorrenza o convoluzione, non ha una nozione intrinseca dell'ordine degli elementi. Per incorporare informazioni sulla posizione, vengono aggiunti encoding posizionali agli embedding di input.

La formulazione originale utilizza funzioni sinusoidali di diverse frequenze:

```
PE(pos, 2i) = sin(pos/10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
```

Dove:
- pos è la posizione nella sequenza
- i è la dimensione nell'embedding
- d_model è la dimensionalità del modello

Questi encoding hanno proprietà interessanti:
- Permettono al modello di generalizzare a sequenze più lunghe di quelle viste durante l'addestramento
- Mantengono informazioni sulla distanza relativa tra posizioni

### Feed-Forward Networks

Ogni blocco di attenzione è seguito da una rete feed-forward applicata indipendentemente a ciascuna posizione:

```
FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2
```

Questa è essenzialmente una trasformazione a due strati con attivazione ReLU, che:
- Introduce non-linearità nel modello
- Aumenta la capacità rappresentativa
- Permette trasformazioni specifiche per posizione

### Layer Normalization e Residual Connections

Per facilitare l'addestramento di reti profonde, il Transformer utilizza:

- **Layer Normalization**: Normalizza gli input a ciascun sub-layer, calcolando media e varianza per ogni esempio individualmente
  ```
  LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
  ```

- **Residual Connections**: Aggiungono l'input di un sub-layer al suo output
  ```
  Output = LayerNorm(x + Sublayer(x))
  ```

Queste tecniche migliorano significativamente la stabilità e la velocità di convergenza durante l'addestramento.

### Architettura del Decoder (utilizzata in GPT)

I modelli GPT utilizzano solo la parte decoder del Transformer, con alcune modifiche. L'architettura del decoder include:

1. **Masked Self-Attention**: A differenza dell'encoder, il decoder utilizza masked self-attention, dove ogni posizione può attendere solo a posizioni precedenti (non future). Questo è essenziale per i modelli autoregressive come GPT.

2. **Stack di blocchi identici**: Ogni blocco contiene:
   - Masked multi-head self-attention
   - Feed-forward network
   - Layer normalization e residual connections

3. **Linear Layer e Softmax finale**: Trasforma le rappresentazioni finali in distribuzioni di probabilità sul vocabolario

La caratteristica chiave del decoder è la sua natura autoregressive: durante l'addestramento, predice la parola successiva dato il contesto precedente, e durante la generazione, utilizza le parole già generate come contesto per predire la prossima.

## Parte 2: Implementazione di un modello GPT from scratch

In questa sezione, implementeremo un modello GPT semplificato from scratch utilizzando PyTorch. L'obiettivo è fornire una comprensione pratica di come funziona l'architettura, piuttosto che creare un modello pronto per la produzione.

### Setup e importazioni

Iniziamo con le importazioni necessarie:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```

### Definizione dei parametri del modello

Definiamo i parametri principali del nostro modello:

```python
class GPTConfig:
    def __init__(self, vocab_size, block_size, n_embd=768, n_layer=12, n_head=12, dropout=0.1):
        self.vocab_size = vocab_size  # dimensione del vocabolario
        self.block_size = block_size  # lunghezza massima della sequenza
        self.n_embd = n_embd          # dimensione degli embedding
        self.n_layer = n_layer        # numero di blocchi del decoder
        self.n_head = n_head          # numero di teste di attenzione
        self.dropout = dropout        # probabilità di dropout
```

### Implementazione del Self-Attention

Implementiamo il meccanismo di self-attention con masking:

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
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

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

Questa implementazione include:
- Proiezioni lineari per query, key e value
- Divisione in multiple teste di attenzione
- Masking causale per assicurare che ogni posizione possa attendere solo a posizioni precedenti
- Scaling dei punteggi di attenzione
- Proiezione finale dell'output

### Implementazione del Feed-Forward Network

Implementiamo la rete feed-forward che segue ogni blocco di attenzione:

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

### Implementazione di un blocco del Transformer

Ora combiniamo self-attention e feed-forward in un blocco completo:

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

Nota che utilizziamo la normalizzazione pre-layer (applicando layer norm prima di ogni sub-layer) invece della normalizzazione post-layer utilizzata nel paper originale. Questa è una modifica introdotta in GPT-2 che ha dimostrato di migliorare la stabilità dell'addestramento.

### Implementazione del modello GPT completo

Infine, mettiamo insieme tutti i componenti per creare il modello GPT completo:

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

Questo modello include:
- Embedding per token e posizione
- Stack di blocchi Transformer
- Layer norm finale
- Testa di language modeling (proiezione lineare al vocabolario)
- Metodo di generazione che implementa campionamento autoregressive

### Esempio di utilizzo

Ecco come potremmo utilizzare il nostro modello GPT per un semplice task di language modeling:

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

### Addestramento del modello

Per completezza, ecco una funzione di addestramento semplificata:

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

### Ottimizzazioni e considerazioni pratiche

Nella pratica, l'implementazione di un modello GPT completo richiederebbe diverse ottimizzazioni e considerazioni aggiuntive:

1. **Efficienza computazionale**: Tecniche come mixed precision training, gradient accumulation, e model parallelism sono essenziali per addestrare modelli di grandi dimensioni.

2. **Tokenizzazione**: Un tokenizer efficace è cruciale. Modelli come GPT-2 e GPT-3 utilizzano Byte-Pair Encoding (BPE) per gestire efficacemente vocabolari di grandi dimensioni.

3. **Strategie di addestramento**: Tecniche come learning rate scheduling, weight decay, e gradient clipping sono importanti per la stabilità dell'addestramento.

4. **Generazione di testo**: Strategie avanzate come beam search, nucleus sampling, e temperature scaling possono migliorare significativamente la qualità del testo generato.

5. **Fine-tuning**: Per applicazioni specifiche, il fine-tuning su dataset mirati può migliorare significativamente le performance.

## Confronto tra BERT e GPT

Per completare la nostra comprensione, è utile confrontare le architetture BERT e GPT, che rappresentano due approcci fondamentali ai modelli Transformer:

### BERT (Bidirectional Encoder Representations from Transformers)

- **Architettura**: Utilizza principalmente la parte encoder del Transformer
- **Bidirezionalità**: Considera il contesto in entrambe le direzioni (precedente e successivo)
- **Pre-addestramento**: Masked Language Modeling (MLM) e Next Sentence Prediction (NSP)
- **Applicazioni**: Eccelle in compiti di comprensione del linguaggio come classificazione, named entity recognition, question answering

### GPT (Generative Pre-trained Transformer)

- **Architettura**: Utilizza principalmente la parte decoder del Transformer
- **Unidirezionalità**: Considera solo il contesto precedente (left-to-right)
- **Pre-addestramento**: Predizione della parola successiva (language modeling classico)
- **Applicazioni**: Eccelle in compiti generativi come completamento di testo, traduzione, riassunto

Le principali differenze possono essere riassunte così:

1. **Direzione del contesto**: BERT è bidirezionale, GPT è unidirezionale
2. **Obiettivo di pre-addestramento**: BERT predice parole mascherate, GPT predice la parola successiva
3. **Uso tipico**: BERT per comprensione, GPT per generazione
4. **Architettura**: BERT usa l'encoder, GPT usa il decoder

Entrambi gli approcci hanno i loro punti di forza e sono complementari in molte applicazioni NLP.

## Conclusione

In questa appendice, abbiamo esplorato in dettaglio l'architettura Transformer, con particolare attenzione ai modelli generativi come GPT. Abbiamo visto come il meccanismo di self-attention permetta ai modelli di catturare dipendenze a lungo termine in modo efficiente, e come l'architettura complessiva faciliti l'addestramento di modelli profondi attraverso tecniche come layer normalization e residual connections.

Abbiamo anche implementato un modello GPT semplificato from scratch utilizzando PyTorch, esplorando ogni componente dell'architettura e come questi componenti interagiscono per creare un potente modello linguistico generativo.

Questa comprensione dettagliata dell'architettura Transformer e della sua implementazione fornisce una base solida per lavorare con modelli linguistici moderni, sia utilizzando implementazioni esistenti che sviluppando soluzioni personalizzate per applicazioni specifiche.

## Riferimenti

- Vaswani, A., et al. (2017). Attention is All You Need. Advances in Neural Information Processing Systems, 30.
- Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training.
- Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI Blog.
- Brown, T. B., et al. (2020). Language Models are Few-Shot Learners. Advances in Neural Information Processing Systems, 33.
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. Proceedings of NAACL-HLT 2019.
