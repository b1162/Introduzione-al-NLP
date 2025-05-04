---
marp: true
theme: default
paginate: true
---

# Elaborazione del Linguaggio Naturale (NLP) 🐍
## Modulo Pratico con NLTK 🛠️

<!-- 
Nota per il presentatore: Slide di titolo. Presenta il modulo come un'immersione pratica nell'NLP usando NLTK in Python. Sottolinea che si vedranno esempi concreti per capire come funziona il preprocessing e l'analisi del testo.
-->

---

# Cosa Impareremo Oggi 🗺️

- **Installazione** ⚙️: Come preparare l'ambiente.
- **Tokenizzazione** 🔪: Dividere il testo in frasi e parole.
- **Stop Words** 🚫: Rimuovere parole comuni.
- **Stemming & Lemmatizzazione** 🌱: Normalizzare le parole.
- **POS Tagging** 🏷️: Identificare le parti del discorso.
- **Chunking** 🧩: Estrarre gruppi sintattici (es. sintagmi nominali).

<!-- 
Nota per il presentatore: Questa è la roadmap del modulo. Elenca i punti chiave che verranno trattati. Sottolinea che si partirà dalle basi (installazione, tokenizzazione) per poi passare a tecniche di analisi più complesse come il POS tagging e il chunking. Ogni punto sarà accompagnato da codice Python funzionante.
-->

---

# Installazione e Configurazione ⚙️

Prima di tutto, installiamo NLTK e scarichiamo le risorse necessarie.

```bash
# 1. Installa NLTK usando pip (in terminale)
$ pip install nltk
```

```python
# 2. Scarica i dati necessari (in uno script Python)
import nltk

# Scarica solo i pacchetti usati in questo modulo
# (esegui una volta sola)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt') # Tokenizer pre-addestrato

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords') # Liste di stop words

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except nltk.downloader.DownloadError:
    nltk.download('averaged_perceptron_tagger') # POS tagger

try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet') # WordNet (per lemmatizzazione)

print("Risorse NLTK pronte!")
```
<!-- 
Nota per il presentatore: Spiega i due passaggi: l'installazione della libreria Python (`pip`) e il download dei modelli/dati specifici di NLTK (`nltk.download`). Il codice Python mostrato usa `try-except` per evitare di riscaricare dati già presenti. Menziona che `nltk.download('all')` scaricherebbe tutto, ma è meglio scaricare solo ciò che serve. Ricorda che questi download sono necessari solo la prima volta.
-->

---

# Tokenizzazione: Frasi e Parole 🔪📄➡️📑

Dividere il testo in unità significative.

```python
from nltk.tokenize import sent_tokenize, word_tokenize

# Testo di esempio (da Dune)
testo = ("Muad'Dib learned rapidly because his first training was in how to learn. "
         "And the first lesson of all was the basic trust that he could learn. "
         "It's shocking to find how many people do not believe they can learn, "
         "and how many more believe learning to be difficult.")

# 1. Tokenizzazione in Frasi
lista_frasi = sent_tokenize(testo)
print("Frasi:")
print(lista_frasi) 
# Output: ['...', '...', '...']

# 2. Tokenizzazione in Parole (include punteggiatura)
lista_parole = word_tokenize(testo)
print("\nParole:")
print(lista_parole)
# Output: ['Muad'Dib', 'learned', 'rapidly', ..., 'difficult', '.'] 
```

<!-- 
Nota per il presentatore: Spiega la differenza tra tokenizzazione a livello di frase (`sent_tokenize`) e parola (`word_tokenize`). Mostra l'output (o parte di esso) per chiarire cosa fa ciascuna funzione. Sottolinea che `word_tokenize` è più sofisticato di un semplice `split()` e tratta la punteggiatura come token separati, cosa utile per analisi successive. L'esempio di Dune rende il concetto più concreto.
-->

---

# Rimozione delle Stop Words 🚫🗣️

Eliminare parole comuni (es. "il", "e", "a") che non aggiungono molto significato.

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

frase = "Sir, I protest. I am not a merry man!" # Esempio da Star Trek
parole = word_tokenize(frase)
print("Parole originali:", parole)

# Ottieni la lista di stop words in inglese
stop_words_eng = set(stopwords.words('english')) 
# print(stop_words_eng) # Mostra alcune stop words

# Filtra le parole: mantieni solo quelle NON stop words
# Confronta in minuscolo per gestire maiuscole/minuscole
parole_filtrate = [w for w in parole if w.lower() not in stop_words_eng]

print("Parole filtrate:", parole_filtrate)
# Output: ['Sir', ',', 'protest', '.', 'merry', 'man', '!']
```

<!-- 
Nota per il presentatore: Spiega il concetto di stop words e perché è **spesso** utile rimuoverle (non sempre!). Mostra come NLTK fornisce liste predefinite per varie lingue (`stopwords.words('english')`). Sottolinea l'uso di `.lower()` per un confronto case-insensitive e l'uso di un `set` per controlli più veloci. Fai notare che la punteggiatura non viene rimossa qui (viene fatto in passaggi successivi se necessario).
-->

---

# Stemming: Alla Radice della Parola 🌱✂️

Ridurre le parole alla loro radice (stem), anche se non è una parola reale. Utile per raggruppare termini correlati.

```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer() # Un algoritmo di stemming comune
testo = ("The crew of the USS Discovery discovered many discoveries. "
         "Discovering is what explorers do.")
parole = word_tokenize(testo)

print("Parola -> Stem (radice)")
for w in parole:
    print(f"{w} -> {stemmer.stem(w)}")

# Output parziale:
# Discovery -> discoveri
# discovered -> discov
# discoveries -> discoveri
# Discovering -> discov
# explorers -> explor 
```

<!-- 
Nota per il presentatore: Spiega che lo stemming è un processo "brutale" che taglia via i suffissi basandosi su regole, senza usare un dizionario. Mostra come parole diverse come "discovered" e "discoveries" possano essere ridotte a radici simili ("discov", "discoveri"). Fai notare che il risultato non è sempre una parola di senso compiuto ("discoveri", "explor"). Utile per information retrieval ma meno per analisi linguistiche precise.
-->

---

# Lemmatizzazione: Alla Forma Base 📚✅

Ridurre le parole alla loro forma base (lemma) usando un dizionario. Produce parole reali.

```python
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer() # Usa il dizionario WordNet

# Esempio 1: Parola singola
print(f"Lemmatizzazione di 'scarves': {lemmatizer.lemmatize('scarves')}") 
# Output: scarf (corretto!) vs stemmer -> scarv

# Esempio 2: Frase intera
frase = "The friends of DeSoto love scarves."
parole = word_tokenize(frase)
lemmi = [lemmatizer.lemmatize(w) for w in parole]
print(f"Lemmi della frase: {lemmi}")
# Output: ['The', 'friend', 'of', 'DeSoto', 'love', 'scarf', '.']
```

<!-- 
Nota per il presentatore: Confronta la lemmatizzazione con lo stemming della slide precedente. Spiega che la lemmatizzazione usa un dizionario (WordNet) per trovare la forma base corretta (lemma). Mostra come "scarves" diventa "scarf" (parola valida) e "friends" diventa "friend". Sottolinea che questo approccio è più accurato dello stemming ma richiede risorse linguistiche (WordNet) e può essere più lento.
-->

---

# Lemmatizzazione con Contesto Grammaticale (POS) 🤔🏷️

La lemmatizzazione migliora specificando la parte del discorso (Part-of-Speech).

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Senza specificare il POS (assume sostantivo 'n' di default)
print(f"lemmatize('worst'): {lemmatizer.lemmatize('worst')}") 
# Output: worst

# Specificando che è un aggettivo ('a')
# NOTA: WordNet usa codici specifici: 'a'=aggettivo, 'v'=verbo, 'n'=nome, 'r'=avverbio
print(f"lemmatize('worst', pos='a'): {lemmatizer.lemmatize('worst', pos='a')}") 
# Output: bad (corretto! 'worst' è il superlativo di 'bad')
```

<!-- 
Nota per il presentatore: Questo è un punto cruciale sulla lemmatizzazione. Spiega che specificare il POS tag (che vedremo nella prossima slide come ottenere) rende il lemmatizzatore molto più preciso. L'esempio di "worst" -> "bad" (se aggettivo) vs "worst" -> "worst" (se assunto nome) è molto chiaro. Anticipa che il prossimo passo sarà capire come ottenere automaticamente questi POS tag.
-->

---

# Part-of-Speech (POS) Tagging 🏷️✍️

Assegnare a ogni parola la sua categoria grammaticale (nome, verbo, aggettivo, ecc.).

```python
from nltk import pos_tag, word_tokenize

# Frase di esempio (Carl Sagan)
frase = "If you wish to make an apple pie from scratch, you must first invent the universe."
parole = word_tokenize(frase)

# Esegui il POS tagging
taggati = pos_tag(parole) # Usa il tagger pre-addestrato di NLTK

print(taggati) 
# Output: [('If', 'IN'), ('you', 'PRP'), ('wish', 'VBP'), ('to', 'TO'), 
#          ('make', 'VB'), ('an', 'DT'), ('apple', 'NN'), ('pie', 'NN'), ...,
#          ('universe', 'NN'), ('.', '.')]
```
<!-- 
Nota per il presentatore: Spiega cos'è il POS tagging e a cosa serve (capire la struttura grammaticale, aiutare la lemmatizzazione, ecc.). Mostra l'output della funzione `pos_tag`, spiegando alcuni tag comuni (NN=Nome singolare, VB=Verbo base, JJ=Aggettivo, PRP=Pronome personale, IN=Preposizione/Congiunzione). Menziona che NLTK usa il tagset Penn Treebank e che `nltk.help.upenn_tagset('NN')` darebbe la descrizione del tag 'NN'.
-->

---

# Chunking: Estrarre Sintagmi 🧩🏗️

Identificare gruppi di parole con significato sintattico (es. sintagmi nominali) usando regole sui POS tag.

```python
import nltk
from nltk.tokenize import word_tokenize

frase = "It's a dangerous business, Frodo, going out your door." # Esempio da LoTR
parole = word_tokenize(frase)
pos = nltk.pos_tag(parole) # Prima facciamo il POS tagging

# Definiamo una grammatica per i Sintagmi Nominali (NP)
# Regola: (Opzionale:DT) seguito da (Zero o più:JJ) seguito da (Uno:NN)
grammatica = "NP: {<DT>?<JJ>*<NN>}" 
parser_chunk = nltk.RegexpParser(grammatica) # Crea il parser

# Applica il parser alla frase taggata
tree = parser_chunk.parse(pos) 
#tree.draw() # Apre una finestra con l'albero (utile in locale)

# Estrai e stampa i chunk NP trovati
print("Chunk NP trovati:")
for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
    chunk_parole = [token for token, tag in subtree.leaves()]
    print("-", " ".join(chunk_parole))
# Output:
# - a dangerous business
# - door 
```
<!-- 
Nota per il presentatore: Spiega il concetto di chunking come un passo intermedio verso il parsing completo. Decodifica la regola della grammatica (`<DT>?<JJ>*<NN>`). Mostra come il codice prima tagga la frase e poi applica la grammatica per estrarre i chunk NP. Spiega perché "your door" non viene preso tutto (perché 'your' è PRP$ e non DT nella nostra regola semplice). Menziona `tree.draw()` come strumento utile per la visualizzazione.
-->

---

# Ricapitolando 🏁

Abbiamo visto come usare NLTK per:

1.  **Installare** e configurare l'ambiente.
2.  **Tokenizzare** testo in frasi e parole (`sent_tokenize`, `word_tokenize`).
3.  **Filtrare Stop Words** (`stopwords.words`).
4.  **Stemming** (es. `PorterStemmer`).
5.  **Lemmatizzazione** (`WordNetLemmatizer`, opz. con `pos=`).
6.  **POS Tagging** (`pos_tag`).
7.  **Chunking** con grammatiche (`RegexpParser`).

Questi sono blocchi fondamentali per costruire applicazioni NLP più complesse! 🧱

<!-- 
Nota per il presentatore: Questa slide riassume i punti chiave e le funzioni NLTK associate a ciascun passaggio. Serve a consolidare quanto appreso. Sottolinea che queste tecniche di preprocessing e analisi sono spesso i primi passi in pipeline NLP più grandi (es. sentiment analysis, topic modeling, chatbot).
-->

---

# Prossimi Passi & Risorse 🚀

- **Sperimenta!** Prova con altri testi, anche in italiano (`stopwords.words('italian')`).
- **Approfondisci NLTK:** NER, Parsing completo, Classificazione...
- 📖 **NLTK Book:** [nltk.org/book/](https://www.nltk.org/book/) (Gratuito!)
- 📄 **Articolo Real Python:** [Link all'articolo originale](https://realpython.com/nltk-nlp-python/)
- ❓ **Domande?**

<!-- 
Nota per il presentatore: Incoraggia i partecipanti a continuare a esplorare. Suggerisci di provare con testi diversi e magari in altre lingue supportate da NLTK (come l'italiano per le stop words). Indica le risorse principali (libro NLTK, articolo originale) per approfondire. Infine, apri la sessione alle domande.
-->
