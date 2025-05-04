---
marp: true
theme: default
paginate: true
---

# Modulo 6: Etica e Responsabilità nell'NLP 🤖⚖️

![bg right:40%](https://source.unsplash.com/random/800x600/?ethics,technology)

### Un viaggio tra opportunità e responsabilità

<!--
In questa presentazione esploreremo le implicazioni etiche del Natural Language Processing, un campo in rapida evoluzione che sta trasformando il modo in cui interagiamo con la tecnologia. 

L'etica nell'NLP non è solo una questione teorica, ma ha implicazioni pratiche significative che influenzano la vita delle persone. Oggi vedremo come i sistemi NLP incorporano valori e priorità, e come questi possono avere conseguenze profonde.

Possibile domanda per la platea: "Chi di voi ha interagito con un sistema di NLP oggi? Forse un assistente vocale, un chatbot, o un traduttore automatico?"
-->

---

## Cosa tratteremo oggi 📋

- Introduzione all'etica nell'NLP
- Bias e fairness nei sistemi NLP
- Privacy e sicurezza dei dati
- Trasparenza e spiegabilità
- Impatto sociale e responsabilità
- Casi di studio etici nell'NLP
- Framework etici e linee guida
- Sviluppo responsabile di sistemi NLP

<!--
Questa è la roadmap della nostra presentazione. Copriremo diversi aspetti dell'etica nell'NLP, partendo dai concetti fondamentali fino ad arrivare a casi di studio concreti e approcci pratici per lo sviluppo responsabile.

È importante notare che questi temi sono interconnessi: ad esempio, i bias influenzano l'impatto sociale, mentre la trasparenza è fondamentale per garantire l'accountability.

Domanda per stimolare la riflessione: "Quale di questi aspetti pensate sia il più critico per il futuro dell'NLP e perché?"
-->

---

## Introduzione all'etica nell'NLP 🧭

- L'NLP media sempre più le nostre interazioni quotidiane 🌐
- I sistemi NLP **non sono strumenti neutri** 🚩
- Incorporano valori, priorità e visioni del mondo 🧠
- Prendono decisioni che un tempo erano esclusivamente umane 🤔

> "Con grande potere derivano grandi responsabilità"

<!--
L'etica nell'NLP è un campo di crescente importanza che affronta le implicazioni morali, sociali e legali delle tecnologie che elaborano e generano linguaggio naturale.

È fondamentale sottolineare che i sistemi NLP non sono strumenti neutri: incorporano valori, priorità e visioni del mondo che possono avere profonde conseguenze. Questi sistemi prendono decisioni che un tempo erano esclusivamente umane, spesso con minore trasparenza e accountability.

La citazione, sebbene famosa da Spider-Man, è particolarmente rilevante qui: man mano che i sistemi NLP diventano più potenti, la nostra responsabilità nel garantire che siano sviluppati e utilizzati eticamente aumenta.

Domanda per la platea: "In quali ambiti della vostra vita quotidiana notate la presenza di sistemi NLP? Avete mai riflettuto sulle implicazioni etiche di queste tecnologie?"
-->
---

## Definizione di Algoretica

L'algoretica è un campo emergente dell'etica che si occupa di studiare e valutare i problemi morali legati allo sviluppo, all'uso e all'impatto degli algoritmi, dei dati e delle pratiche correlate, specialmente nell'ambito dell'intelligenza artificiale (IA) e del Natural Language Processing (NLP). 

Questo disciplina si basa su valori condivisi e principi morali che guidano il comportamento nella società, andando oltre le leggi per concentrarsi su ciò che è considerato "giusto" o "sbagliato" in contesti tecnologici
L'algoretica applicata al NLP richiede particolare attenzione, poiché le tecnologie linguistiche influenzano molti aspetti della vita quotidiana e possono perpetuare bias o causare danni se non gestite correttamente. 

[The urgency of an algorethics](https://link.springer.com/article/10.1007/s44163-023-00056-6)

---

## Definizione di Algoretica

Di seguito alcuni principi chiave e domande etiche da considerare:
- **Contesto della raccolta dati**: Il contesto in cui i dati sono stati raccolti corrisponde al contesto del loro utilizzo? Ad esempio, utilizzare dati raccolti per scopi accademici in un'applicazione commerciale potrebbe violare aspettative etiche
- **Incentivi e bias nella raccolta dati**: I dati sono stati raccolti da persone o sistemi con quote o strutture di incentivazione che potrebbero introdurre distorsioni? 
- **Rappresentatività**: Chi è sottorappresentato o assente nei dati utilizzati per addestrare modelli di NLP? È possibile trovare dati aggiuntivi o utilizzare metodi statistici per rendere i dataset più inclusivi?
- **Consenso e scelte significative**: I dati sono stati raccolti in un ambiente in cui i soggetti avevano scelte significative e consapevoli riguardo alla loro partecipazione?


---

## Perché l'etica nell'NLP è importante? 🔍

- Adozione in contesti **critici e sensibili** 🏥👨‍⚖️
  - Sanità, giustizia, istruzione, finanza
- Impatto su **decisioni significative** che influenzano vite reali 🧬
- Potenziale di **amplificare disuguaglianze** esistenti 📈
- Necessità di bilanciare **innovazione e protezione** 🛡️

<!--
L'etica nell'NLP è particolarmente importante perché questi sistemi vengono sempre più adottati in contesti critici dove le decisioni hanno un impatto significativo sulla vita delle persone.

Per esempio, nell'ambito sanitario, sistemi NLP possono analizzare cartelle cliniche e supportare diagnosi; nel sistema giudiziario, possono influenzare decisioni sulla libertà condizionale; nel settore finanziario, possono determinare l'accesso al credito.

In tutti questi contesti, è essenziale considerare non solo cosa questi sistemi possono fare, ma anche cosa dovrebbero fare e come dovrebbero farlo.

Spunto di riflessione: "Pensate a un sistema NLP che utilizzate regolarmente. Quali potrebbero essere le conseguenze etiche se questo sistema contenesse bias o non fosse trasparente nel suo funzionamento?"
-->

---

# Domanda 🤔

### Quali sistemi NLP utilizzate quotidianamente?
### Avete mai notato comportamenti problematici?

<!--
Questo è un momento interattivo per coinvolgere la platea e raccogliere esperienze personali. È importante ascoltare attentamente le risposte, poiché potrebbero emergere esempi interessanti che possono essere ripresi durante la presentazione.

Esempi di sistemi NLP quotidiani includono:
- Assistenti vocali (Siri, Alexa, Google Assistant)
- Traduttori automatici (Google Translate)
- Correttori grammaticali (Grammarly)
- Chatbot di servizio clienti
- Sistemi di raccomandazione di contenuti

Comportamenti problematici potrebbero includere bias di genere nelle traduzioni, difficoltà con accenti o dialetti regionali, o suggerimenti inappropriati.

Dopo alcune risposte, posso fare un breve riepilogo collegando le esperienze condivise ai temi che tratteremo nelle prossime slide.
-->

---

## Bias e fairness nei sistemi NLP ⚖️

### Origini dei bias:

- **Dati di addestramento** 📊
  - Sottorappresentazione di gruppi
  - Rappresentazioni stereotipate
- **Scelte di progettazione** 🛠️
  - Feature, definizione del problema, metriche
- **Contesto di utilizzo** 🌍
  - Disparità di accesso e performance

<!--
I bias nei sistemi NLP rappresentano una delle sfide etiche più significative. Questi sistemi possono perpetuare o amplificare pregiudizi esistenti nella società, con conseguenze potenzialmente dannose.

I bias possono emergere da diverse fonti:
1. Nei dati di addestramento: se i dati contengono pregiudizi storici o sociali, i modelli tenderanno a replicarli
2. Nelle scelte di progettazione: decisioni su quali caratteristiche includere, come formulare il problema, quali metriche ottimizzare
3. Nel contesto di utilizzo: anche con dati e progettazione impeccabili, i bias possono emergere quando il sistema viene utilizzato in contesti diversi

Un esempio concreto: modelli addestrati su corpora di notizie storiche potrebbero associare più fortemente "dottore" con "uomo" e "infermiere" con "donna", riflettendo e potenzialmente rinforzando stereotipi di genere.

Domanda per stimolare la discussione: "Quali gruppi pensate siano più frequentemente sottorappresentati nei dati di addestramento dei sistemi NLP?"
-->
---

## Manifestazioni di bias nei sistemi NLP 👁️

- **Word embeddings biased** 📝
  - "uomo : programmatore :: donna : casalinga"
- **Generazione di testo discriminatoria** 💬
  - Descrizioni stereotipate di certi gruppi
- **Classificazione iniqua** 🏷️
  - Performance degradata per lingue minoritarie

<!--
I bias possono manifestarsi in vari modi nei sistemi NLP:

1. Word embeddings biased: le rappresentazioni vettoriali di parole possono catturare e codificare bias sociali. Un esempio famoso è l'analogia "uomo sta a programmatore come donna sta a casalinga", che riflette stereotipi di genere presenti nei dati.

2. Generazione di testo discriminatoria: i modelli generativi possono produrre contenuti che perpetuano stereotipi o linguaggio offensivo, come descrizioni stereotipate di certi gruppi o completamenti di testo che assumono caratteristiche basate su identità.

3. Classificazione iniqua: i classificatori di testo possono performare in modo diseguale tra gruppi diversi, con tassi di errore più alti per certi gruppi o performance degradata per lingue o dialetti minoritari.

Questi esempi mostrano come i bias non siano solo un problema teorico, ma abbiano manifestazioni concrete che possono causare danni reali.

Spunto di riflessione: "Avete mai notato bias in sistemi di traduzione automatica quando traducono professioni da lingue gender-neutral a lingue con distinzioni di genere?"
-->

---

![bg fit](images/cursor-fail.png)

---

## Misurazione e mitigazione dei bias 📏

### Metriche di fairness:
- Demographic parity, Equal opportunity, Equal accuracy

### Tecniche di mitigazione:
- **Interventi sui dati** 🧹
  - Bilanciamento, data augmentation
- **Interventi algoritmici** 🧮
  - Debiasing di embeddings, adversarial learning
- **Interventi post-processing** 🔄
  - Calibrazione, re-ranking, filtering

<!--
Per affrontare i bias, è essenziale poterli misurare e valutare in modo sistematico. Diverse metriche catturano aspetti diversi di equità:
- Demographic parity: output simili attraverso gruppi demografici
- Equal opportunity: tassi di falsi negativi simili tra gruppi
- Equal accuracy: accuratezza simile tra gruppi

Una volta identificati i bias, diverse tecniche possono essere utilizzate per mitigarli:

1. Interventi sui dati: bilanciamento del dataset, data augmentation per gruppi sottorappresentati
2. Interventi algoritmici: tecniche come debiasing di embeddings o adversarial learning
3. Interventi post-processing: aggiustamento delle soglie di decisione, modifica dell'ordine dei risultati

È importante notare che non esiste una soluzione universale: la scelta delle metriche e delle tecniche di mitigazione dipende dal contesto specifico e dai valori che si vogliono promuovere.

Domanda per la platea: "Quale approccio alla mitigazione dei bias pensate sia più promettente nel lungo termine: intervenire sui dati, sugli algoritmi o sul post-processing? Perché?"
-->

---


![bg fit ](images/bias-genere.png)

<!--
Questa slide mostra un esempio concreto di bias di genere nei sistemi di traduzione automatica. L'immagine illustra come, traducendo frasi da lingue gender-neutral (come il turco o l'ungherese) a lingue con distinzioni di genere (come l'inglese), il sistema tende ad assegnare generi basati su stereotipi.

Per esempio, "O bir doktor" in turco (che significa letteralmente "Quella persona è un dottore" senza specificare il genere) viene tradotto come "He is a doctor" (Lui è un dottore), mentre "O bir hemşire" (Quella persona è un infermiere/a) viene tradotto come "She is a nurse" (Lei è un'infermiera).

Questo esempio mostra chiaramente come i bias nei dati di addestramento si riflettano nelle traduzioni, perpetuando stereotipi di genere legati alle professioni.

Alcuni sistemi di traduzione hanno iniziato ad affrontare questo problema offrendo traduzioni di genere esplicite, mostrando entrambe le versioni quando il genere è ambiguo nella lingua di origine.

Domanda per stimolare la discussione: "Quali altre professioni pensate siano soggette a bias di genere nelle traduzioni automatiche? Come potrebbe questo influenzare la percezione delle professioni nelle diverse culture?"
-->
---

## Privacy e sicurezza dei dati 🔒

### Sfide uniche nell'NLP:

- **Informazioni personali nel testo** 📝
  - PII, informazioni sensibili, comportamentali
- **Memorizzazione nei modelli** 🧠
  - Riproduzione verbatim, inferenza da memorizzazione
- **Inferenze non autorizzate** 🔍
  - Profilazione demografica e psicografica

<!--
La privacy e la sicurezza dei dati rappresentano considerazioni etiche fondamentali nell'NLP, particolarmente rilevanti in un'epoca di crescente raccolta e analisi di dati linguistici personali e sensibili.

I sistemi NLP presentano sfide uniche alla privacy a causa della natura dei dati testuali:

1. I testi spesso contengono informazioni personali identificabili (PII) e sensibili, come nomi, indirizzi, informazioni sulla salute, opinioni politiche, ecc.

2. I modelli NLP, specialmente quelli di grandi dimensioni, possono memorizzare informazioni dai dati di addestramento e potenzialmente riprodurle quando stimolati con prompt specifici.

3. I sistemi NLP possono inferire attributi sensibili non esplicitamente dichiarati, come età, genere, background culturale, tratti di personalità o condizioni mediche.

Queste sfide richiedono approcci specifici per proteggere la privacy degli individui i cui dati vengono utilizzati per addestrare o interagire con sistemi NLP.

Spunto di riflessione: "Pensate a un messaggio che avete inviato recentemente. Quali informazioni personali conteneva che potrebbero essere problematiche se memorizzate da un modello linguistico?"
-->

---

## Tecniche per la privacy-preserving NLP 🛡️

- **Anonimizzazione e de-identificazione** 🎭
  - NER per PII, redaction, pseudonimizzazione
- **Privacy differenziale** 📊
  - Aggiunta di rumore controllato
- **Federated learning** 📱
  - Addestramento locale sui dispositivi

<!--
Diverse tecniche sono state sviluppate per proteggere la privacy nei sistemi NLP:

1. Anonimizzazione e de-identificazione: tecniche per rimuovere o mascherare informazioni identificabili, come Named Entity Recognition per identificare PII, redaction (rimozione completa), pseudonimizzazione (sostituzione con pseudonimi) o generalizzazione.

2. Privacy differenziale: tecniche matematiche che aggiungono rumore controllato durante l'addestramento o l'analisi, offrendo garanzie matematiche sulla protezione delle informazioni individuali.

3. Federated learning: approccio che mantiene i dati sui dispositivi degli utenti, addestrando i modelli localmente e condividendo solo gli aggiornamenti dei parametri, non i dati stessi.

Queste tecniche rappresentano un campo di ricerca attivo, con continui sviluppi per bilanciare utilità e privacy.

Domanda per la platea: "Quali di queste tecniche pensate sia più promettente per applicazioni consumer come assistenti vocali o tastiere predittive? E per applicazioni in ambito sanitario?"
-->

---

# Quiz per la platea! 🎯

### Quale di queste NON è una tecnica di privacy-preserving NLP?

A) Federated learning
B) Differential privacy
C) Gradient boosting
D) Pseudonimizzazione

<!--
Questo è un momento interattivo per verificare l'attenzione e la comprensione della platea. La risposta corretta è C) Gradient boosting, che è una tecnica di machine learning per migliorare la performance dei modelli, non specificamente legata alla privacy.

Posso spiegare brevemente perché le altre sono tecniche di privacy-preserving:
- Federated learning: addestra i modelli sui dispositivi degli utenti senza condividere i dati grezzi
- Differential privacy: aggiunge rumore controllato per proteggere le informazioni individuali
- Pseudonimizzazione: sostituisce identificatori reali con pseudonimi per proteggere l'identità

Questo quiz aiuta a consolidare la comprensione delle tecniche di privacy-preserving NLP discusse nella slide precedente.

Dopo aver raccolto alcune risposte, posso rivelare la risposta corretta e fornire una breve spiegazione prima di passare alla slide successiva.
-->
---

## Sicurezza e attacchi ai sistemi NLP 🛑

### Vulnerabilità specifiche:

- **Adversarial attacks** 🎯
  - Text perturbations, prompt injection, jailbreaking
- **Data poisoning** ☠️
  - Inserimento di esempi malevoli nei dati
  - Backdoor attacks
- **Model stealing** 🕵️
  - Estrazione di modelli proprietari

<!--
Oltre alle preoccupazioni sulla privacy, i sistemi NLP sono vulnerabili a vari attacchi di sicurezza:

1. Adversarial attacks: attacchi che mirano a ingannare o manipolare i sistemi NLP attraverso modifiche sottili all'input che preservano il significato ma ingannano il modello, manipolazione dell'input per indurre comportamenti non intenzionali (prompt injection), o tecniche per aggirare guardrails di sicurezza (jailbreaking).

2. Data poisoning: attacchi che mirano a compromettere l'addestramento inserendo esempi malevoli nei dati o creando trigger nascosti che causano comportamenti specifici (backdoor attacks).

3. Model stealing: tentativi di estrarre modelli proprietari attraverso query sistematiche, ricostruendo funzionalità simili senza accesso diretto al modello originale.

Questi attacchi sono particolarmente preoccupanti quando i modelli vengono utilizzati in contesti critici o quando vengono addestrati su dati raccolti da fonti pubbliche.

Spunto di riflessione: "Come possiamo bilanciare l'apertura e la trasparenza nella ricerca NLP con la necessità di proteggere i sistemi da potenziali abusi?"
-->

---

## Framework normativi e compliance 📜

- **Regolamentazioni rilevanti** ⚖️
  - GDPR (EU), CCPA/CPRA (California), HIPAA (US)
- **Principi di privacy by design** 🏗️
  - Minimizzazione dei dati
  - Limitazione dello scopo
  - Privacy come impostazione predefinita
  - Trasparenza

<!--
La privacy e la sicurezza dei dati nell'NLP sono sempre più regolamentate attraverso vari framework normativi:

1. Regolamentazioni rilevanti:
   - GDPR nell'Unione Europea, che impone requisiti specifici per il trattamento di dati personali
   - CCPA/CPRA in California, che stabilisce diritti dei consumatori sui loro dati personali
   - HIPAA negli Stati Uniti, che protegge le informazioni sanitarie
   - Varie regolamentazioni settoriali specifiche per finanza, istruzione, ecc.

2. Principi di privacy by design, un approccio che incorpora la privacy fin dalle prime fasi di sviluppo:
   - Minimizzazione dei dati: raccogliere solo i dati strettamente necessari
   - Limitazione dello scopo: utilizzare i dati solo per scopi dichiarati
   - Privacy come impostazione predefinita: massima protezione senza intervento dell'utente
   - Trasparenza: comunicazione chiara su raccolta e utilizzo dei dati

Questi framework non sono solo obblighi legali, ma offrono anche linee guida utili per lo sviluppo responsabile.

Domanda per la platea: "Quali sfide specifiche pensate che i sistemi NLP affrontino nel conformarsi a regolamentazioni come il GDPR? Come potrebbero queste sfide essere affrontate?"
-->

---

## Trasparenza e spiegabilità 🔍

### L'importanza della trasparenza:

- **Trasparenza sui dati** 📊
  - Provenienza, caratteristiche, annotazione
- **Trasparenza algoritmica** 🧮
  - Architettura, iperparametri, metriche
- **Trasparenza operativa** ⚙️
  - Scopo, processo decisionale, supervisione

<!--
La trasparenza e la spiegabilità sono principi etici fondamentali per i sistemi NLP, particolarmente cruciali quando questi sistemi influenzano decisioni significative.

La trasparenza nei sistemi NLP comprende diversi aspetti interconnessi:

1. Trasparenza sui dati: chiarezza riguardo ai dati utilizzati, inclusa la loro provenienza, caratteristiche, processi di annotazione e rappresentatività.

2. Trasparenza algoritmica: apertura riguardo agli algoritmi e ai modelli utilizzati, inclusa l'architettura, gli iperparametri, le metriche di performance e le limitazioni note.

3. Trasparenza operativa: chiarezza su come il sistema viene utilizzato in pratica, incluso lo scopo previsto, il processo decisionale, il ruolo della supervisione umana e i processi di monitoraggio.

Questa trasparenza è essenziale per costruire fiducia e permettere una valutazione critica dei sistemi NLP.

Spunto di riflessione: "Pensate che la trasparenza completa sia sempre desiderabile? Ci sono casi in cui potrebbe essere controproducente o rischiosa?"
-->
---

## Sfide alla spiegabilità nell'NLP 🤯


- **Complessità e opacità** 🧩
  - Miliardi di parametri
  - Rappresentazioni distribuite
  - Comportamenti emergenti
- **Trade-off tra performance e spiegabilità** ⚖️
  - Modelli più potenti tendono ad essere più opachi

<!--
I moderni sistemi NLP, specialmente quelli basati su deep learning, presentano sfide uniche alla spiegabilità:

1. Complessità e opacità:
   - Dimensioni enormi con milioni o miliardi di parametri che rendono impossibile l'ispezione manuale
   - Rappresentazioni distribuite dove le informazioni sono codificate in modo non localizzato
   - Relazioni non-lineari complesse difficili da caratterizzare
   - Comportamenti emergenti che non sono stati esplicitamente programmati

2. Trade-off tra performance e spiegabilità:
   - Spesso esiste una tensione tra modelli più semplici e interpretabili (come la regressione logistica) e modelli più complessi con performance superiori (come i Transformer)
   - Questo trade-off è particolarmente acuto nell'NLP, dove i modelli più potenti tendono ad essere i più opachi

Queste sfide rendono particolarmente difficile fornire spiegazioni comprensibili per le decisioni dei sistemi NLP avanzati.

Domanda per stimolare la discussione: "Come possiamo bilanciare il desiderio di modelli sempre più potenti con la necessità di sistemi comprensibili e spiegabili?"
-->

---

## Tecniche per l'interpretabilità nell'NLP 🔬

### Approcci principali:

- **Interpretabilità intrinseca** 🧠
  - Attention visualization
  - Sparse models
- **Interpretabilità post-hoc** 🔍
  - LIME, SHAP, Feature attribution
- **Spiegazioni in linguaggio naturale** 💬
  - Rationale generation
  - Counterfactual explanations

<!--
Diverse tecniche sono state sviluppate per rendere i sistemi NLP più interpretabili:

1. Interpretabilità intrinseca: approcci che rendono i modelli intrinsecamente più interpretabili
   - Attention visualization: visualizzazione dei pesi di attenzione per comprendere su quali parti dell'input il modello si sta focalizzando
   - Sparse models: modelli che utilizzano solo un sottoinsieme di feature
   - Modelli a strati: architetture che separano chiaramente diverse funzionalità

2. Interpretabilità post-hoc: tecniche applicate dopo l'addestramento per spiegare le decisioni
   - LIME: approssima localmente il comportamento del modello con un modello interpretabile
   - SHAP: attribuisce importanza alle feature basandosi sulla teoria dei giochi
   - Feature attribution: identifica quali parole o frasi hanno maggiormente influenzato una predizione

3. Spiegazioni in linguaggio naturale: generazione di spiegazioni comprensibili agli umani
   - Rationale generation: produzione di giustificazioni testuali per le decisioni
   - Counterfactual explanations: spiegazioni che indicano come l'input dovrebbe cambiare per ottenere un risultato diverso

Queste tecniche offrono diverse prospettive sulla spiegabilità, ciascuna con i propri punti di forza e limitazioni.

Spunto di riflessione: "Quale tipo di spiegazione pensate sia più utile per utenti non tecnici? E per sviluppatori o ricercatori?"
-->

---

# Momento di riflessione 💭

### Se un sistema NLP prende una decisione importante che vi riguarda...

### Quali informazioni vorreste avere sul suo funzionamento?

<!--
Questo è un momento interattivo per far riflettere la platea sulle proprie aspettative riguardo alla trasparenza e alla spiegabilità dei sistemi NLP.

Posso guidare la discussione con alcuni esempi:
- Se un sistema NLP analizza il vostro CV e decide se siete idonei per un colloquio
- Se un assistente vocale interpreta le vostre richieste e prende decisioni su vostro conto
- Se un sistema di moderazione dei contenuti rimuove un vostro post sui social media

Le risposte potrebbero includere:
- Quali dati sono stati utilizzati per addestrare il sistema
- Come il sistema pesa diversi fattori nella decisione
- Quali alternative avrei potuto presentare per ottenere un risultato diverso
- Chi è responsabile in ultima istanza della decisione
- Come posso contestare la decisione se ritengo sia errata

Questa riflessione aiuta a collegare i concetti astratti di trasparenza e spiegabilità alle esperienze concrete delle persone, rendendo il tema più tangibile e rilevante.
-->

---

## Impatto sociale e responsabilità 🌍

### Impatto multidimensionale:

- **Accesso all'informazione e filter bubbles** 🔍
  - Algoritmi che mediano l'accesso all'informazione
- **Disinformazione e manipolazione** 📰
  - Generazione di fake news, deepfake testuali
- **Impatto sul lavoro e sull'economia** 💼
  - Automazione di compiti cognitivi
- **Impatto su lingue e culture** 🗣️
  - Disparità linguistiche, omogeneizzazione culturale

<!--
I sistemi NLP hanno un impatto sociale profondo e multidimensionale che richiede un'attenta considerazione delle responsabilità etiche di sviluppatori, organizzazioni e società.

Questo impatto si manifesta in vari modi:

1. Accesso all'informazione: i sistemi NLP mediano sempre più il nostro accesso all'informazione attraverso motori di ricerca, feed di social media e sistemi di raccomandazione, potenzialmente creando "filter bubbles" che limitano l'esposizione a prospettive diverse.

2. Disinformazione: le tecnologie NLP possono essere utilizzate per creare o amplificare disinformazione, generando fake news plausibili o imitando lo stile di specifiche persone.

3. Lavoro ed economia: l'automazione basata su NLP sta trasformando il panorama lavorativo, impattando professioni precedentemente considerate immuni all'automazione e potenzialmente ampliando disuguaglianze esistenti.

4. Lingue e culture: i sistemi NLP possono influenzare la diversità linguistica e culturale, con performance e disponibilità migliori per lingue dominanti e potenziale omogeneizzazione culturale.

Questi impatti sollevano questioni fondamentali di responsabilità e governance.

Domanda per la platea: "Quale di questi impatti vi preoccupa maggiormente? Quali opportunità positive vedete invece nell'adozione diffusa di sistemi NLP?"
-->

---

## 💼 Impatto sul lavoro

- Automazione di mansioni ripetitive
- Nuove opportunità (traduzione, assistenza)
- Rischio sostituzione lavoro umano 🤖

💬 **Domanda:**  
Quali lavori potrebbero sparire? Quali nuovi lavori nasceranno?

<!-- 
NOTE PER LO SPEAKER:
Parla dei rischi e delle opportunità. Fai esempi: traduttori automatici vs. revisori umani. Ricorda che l’NLP può anche creare nuove professioni.
Domanda: “Siete più ottimisti o preoccupati per il futuro del lavoro?”
-->

---

## Responsabilità degli sviluppatori e delle organizzazioni 👩‍💻👨‍💼

### Approcci proattivi:

- **Valutazione dell'impatto** 📊
  - Impact assessment, stakeholder engagement
- **Design responsabile** 🎨
  - Value-sensitive design, inclusive design
- **Governance e accountability** ⚖️
  - Chiara attribuzione di responsabilità
  - Meccanismi di oversight

<!--
Lo sviluppo e l'implementazione responsabile di sistemi NLP richiede un approccio proattivo da parte di sviluppatori e organizzazioni:

1. Valutazione dell'impatto:
   - Impact assessment: valutazione sistematica di potenziali impatti prima dell'implementazione
   - Stakeholder engagement: coinvolgimento di diversi stakeholder, specialmente comunità potenzialmente impattate
   - Monitoraggio continuo e feedback loops per adattarsi agli effetti reali

2. Design responsabile:
   - Value-sensitive design: incorporazione esplicita di valori etici nel processo di design
   - Inclusive design: considerazione di diversi utenti e contesti d'uso
   - Safeguards by design e graceful failure per minimizzare danni potenziali

3. Governance e accountability:
   - Chiara attribuzione di responsabilità per decisioni e impatti
   - Meccanismi di oversight interni ed esterni
   - Trasparenza operativa e rimedi accessibili

Questi approcci proattivi sono essenziali per garantire che i sistemi NLP siano sviluppati e utilizzati in modo responsabile.

Spunto di riflessione: "Come possiamo incentivare le organizzazioni ad adottare questi approcci proattivi? Quali meccanismi potrebbero essere più efficaci?"
-->

---

# Una domanda provocatoria 🤔

### Chi è responsabile quando un sistema NLP causa un danno?

- Lo sviluppatore del modello?
- L'organizzazione che lo implementa?
- L'utente che lo utilizza?
- Il regolatore che non ha imposto limiti adeguati?

<!--
Questa è una domanda provocatoria per stimolare una discussione sulla responsabilità condivisa nei sistemi NLP. Non esiste una risposta semplice, poiché la responsabilità è spesso distribuita tra diversi attori.

Posso guidare la discussione esplorando diversi scenari:
- Un sistema di traduzione automatica che produce traduzioni offensive
- Un assistente virtuale che fornisce consigli medici errati
- Un sistema di moderazione dei contenuti che censura ingiustamente certe prospettive

Per ciascuno di questi scenari, possiamo considerare il ruolo e la responsabilità di diversi attori:
- Gli sviluppatori che hanno creato il modello base
- Le organizzazioni che hanno implementato e personalizzato il sistema
- Gli utenti che utilizzano il sistema in contesti specifici
- I regolatori che stabiliscono (o non stabiliscono) framework normativi

Questa discussione aiuta a comprendere la complessità della responsabilità etica nell'NLP e l'importanza di un approccio di governance multi-stakeholder.

Dopo alcune risposte, posso sottolineare che in molti casi la responsabilità è condivisa e che è necessario un approccio collaborativo per affrontare le sfide etiche.
-->
---

## Casi di studio etici nell'NLP 📚

### Esempi emblematici:

- **Bias di genere nei sistemi di traduzione** 🌐
  - Traduzioni stereotipate da lingue gender-neutral
- **Moderazione dei contenuti e libertà di espressione** 🗣️
  - Impatto sproporzionato su comunità marginalizate
- **Privacy nei modelli linguistici di grandi dimensioni** 🔒
  - Memorizzazione e potenziale rivelazione di dati personali

<!--
L'analisi di casi di studio concreti offre insights preziosi sulle sfide etiche nell'NLP e sugli approcci per affrontarle:

1. Bias di genere nei sistemi di traduzione automatica:
   - Quando traducono da lingue gender-neutral (come il finlandese o il turco) a lingue con distinzioni di genere, i sistemi tendono a perpetuare stereotipi
   - Soluzioni implementate includono traduzioni di genere esplicite, considerazione del contesto più ampio, e debiasing dei dati

2. Moderazione dei contenuti e libertà di espressione:
   - I sistemi di moderazione automatica hanno mostrato tendenze a flaggare erroneamente contenuti in certi dialetti o varianti linguistiche
   - Approcci come moderazione a più livelli, diversificazione dei dati di addestramento, e meccanismi di appello sono stati sviluppati

3. Privacy nei modelli linguistici di grandi dimensioni:
   - I LLM possono memorizzare e potenzialmente rivelare informazioni personali presenti nei dati di addestramento
   - Tecniche come filtraggio pre-addestramento, privacy differenziale, e unlearning sono state proposte

Questi casi illustrano come le sfide etiche nell'NLP siano complesse e richiedano approcci nuanced che bilanciano diversi valori e considerazioni.

Domanda per la platea: "Conoscete altri esempi di dilemmi etici nell'NLP che avete incontrato personalmente o di cui avete sentito parlare?"
-->

---

## Framework etici e linee guida 📏

### Principi etici fondamentali:

- **Beneficenza e non maleficenza** ❤️
  - Massimizzare benefici, minimizzare danni
- **Autonomia e consenso informato** 🤝
  - Rispetto delle scelte individuali
- **Giustizia ed equità** ⚖️
  - Distribuzione equa di benefici e rischi
- **Trasparenza e accountability** 🔍
  - Apertura su funzionamento e responsabilità

<!--
Per affrontare in modo sistematico le sfide etiche nell'NLP, sono stati sviluppati vari framework etici e linee guida che forniscono principi, processi e strumenti pratici.

Diversi principi etici fondamentali sono emersi come particolarmente rilevanti:

1. Beneficenza e non maleficenza: sviluppare sistemi NLP che portino benefici tangibili e evitino danni prevedibili, bilanciando attentamente rischi e benefici.

2. Autonomia e consenso informato: preservare la capacità degli individui di fare scelte informate, assicurando che comprendano come i loro dati vengono utilizzati.

3. Giustizia ed equità: garantire una distribuzione equa di benefici e rischi, processi decisionali non discriminatori, e rappresentazione equa di diverse prospettive.

4. Trasparenza e accountability: apertura su funzionamento, capacità e limitazioni dei sistemi, capacità di fornire spiegazioni comprensibili, e chiara attribuzione di responsabilità.

Questi principi forniscono una base etica per lo sviluppo e l'implementazione di sistemi NLP.

Spunto di riflessione: "Come possiamo bilanciare questi principi quando entrano in conflitto? Ad esempio, quando la massima trasparenza potrebbe compromettere la sicurezza?"
-->
---

## Framework etici esistenti 📚

- **Framework istituzionali** 🏛️
  - [Principi AI dell'OECD](https://legalinstruments.oecd.org/api/download/?uri=/public/0a206222-278b-4958-9698-6aaa23a9591e.pdf)
  - [Ethics Guidelines for Trustworthy AI (EU)](https://digital-strategy.ec.europa.eu/en/library/ethics-guidelines-trustworthy-ai)
- **Framework industriali** 🏢
  - [Microsoft Responsible AI Principles](https://www.microsoft.com/en-us/ai/principles-and-approach)
  - G[oogle AI Principles](https://ai.google/responsibility/principles/)

<!--
Numerose organizzazioni hanno sviluppato framework etici specifici per l'AI e l'NLP:

1. Framework istituzionali:
   - Principi AI dell'OECD: framework adottato da 42 paesi che enfatizza crescita inclusiva, valori centrati sull'umano, trasparenza, robustezza e accountability
   - Ethics Guidelines for Trustworthy AI (EU): framework europeo che identifica sette requisiti chiave per AI affidabile
   - UNESCO Recommendation on the Ethics of AI: primo strumento normativo globale sull'etica dell'AI

2. Framework industriali:
   - Microsoft Responsible AI Principles: framework che enfatizza fairness, reliability & safety, privacy & security, inclusiveness, transparency, accountability
   - Google AI Principles: linee guida che delineano applicazioni che Google non perseguirà e principi per lo sviluppo responsabile
   - Partnership on AI: principi sviluppati da un consorzio di aziende tecnologiche, organizzazioni di ricerca e società civile

3. Framework accademici e di ricerca:
   - Montreal Declaration for Responsible AI: framework sviluppato attraverso un processo deliberativo che include principi di benessere, autonomia, privacy, solidarietà, democrazia, equità e sostenibilità
   - IEEE Global Initiative on Ethics of Autonomous and Intelligent Systems: standard tecnici che incorporano considerazioni etiche

Questi framework offrono linee guida preziose, ma la loro implementazione pratica rimane una sfida.

Domanda per la platea: "Quali di questi framework conoscevate già? Pensate che siano sufficienti per garantire uno sviluppo etico dell'NLP?"
-->

---

## Implementazione pratica dei framework etici 🛠️

### Strumenti e processi:

- **Strumenti di valutazione etica** 📋
  - Ethical impact assessment, Ethics checklists
- **Processi di governance** 🏗️
  - Ethics review boards, Ethics by design
- **Cultura organizzativa** 👥
  - Leadership commitment, Incentivi allineati
  - Diversità e inclusione

<!--
La traduzione di principi astratti in pratiche concrete richiede strumenti e processi specifici:

1. Strumenti di valutazione etica:
   - Ethical impact assessment: metodologia strutturata per valutare implicazioni etiche
   - Algorithmic impact assessment: framework per valutare impatti potenziali di sistemi algoritmici
   - Ethics checklists: liste di controllo pratiche per diverse fasi di sviluppo
   - Scenario planning: esplorazione di potenziali conseguenze attraverso scenari

2. Processi di governance:
   - Ethics review boards: comitati che valutano progetti dal punto di vista etico
   - Ethics by design: incorporazione di considerazioni etiche fin dalle prime fasi di sviluppo
   - Continuous monitoring: valutazione continua di impatti etici dopo il deployment
   - Stakeholder engagement: coinvolgimento di diversi stakeholder nel processo decisionale

3. Cultura organizzativa:
   - Leadership commitment: impegno visibile della leadership verso l'etica
   - Incentivi allineati: strutture di incentivi che premiano considerazioni etiche
   - Diversità e inclusione: team diversificati che portano prospettive diverse
   - Psychological safety: ambiente che incoraggia la segnalazione di preoccupazioni etiche

Questi strumenti e processi sono essenziali per tradurre principi etici in pratiche concrete.

Spunto di riflessione: "Quali ostacoli vedete nell'implementazione di questi strumenti e processi nelle organizzazioni? Come potrebbero essere superati?"
-->

---

## Sviluppo responsabile di sistemi NLP 🌱

### Ethics by design:

- **Fase di concezione e pianificazione** 🧩
  - Valutazione della necessità, definizione di scopo e limiti
- **Fase di raccolta e preparazione dei dati** 📊
  - Sourcing etico, valutazione di rappresentatività
- **Fase di sviluppo del modello** 🛠️
  - Scelte di architetture, monitoraggio durante l'addestramento
- **Fase di testing e valutazione** 🔍
  - Test multidimensionali, adversarial testing
- **Fase di deployment e monitoraggio** 🚀
  - Deployment graduale, monitoraggio continuo

<!--
Lo sviluppo responsabile di sistemi NLP richiede l'integrazione di considerazioni etiche in ogni fase del ciclo di vita:

1. Fase di concezione e pianificazione:
   - Valutazione della necessità: determinare se un sistema NLP è la soluzione appropriata
   - Definizione di scopo e limiti: chiarire esplicitamente cosa il sistema dovrebbe e non dovrebbe fare
   - Stakeholder mapping e ethical impact assessment iniziale

2. Fase di raccolta e preparazione dei dati:
   - Sourcing etico: ottenere dati in modo rispettoso di diritti e consenso
   - Valutazione di rappresentatività: analizzare chi è rappresentato e chi è escluso
   - Documentazione dettagliata e preprocessing consapevole

3. Fase di sviluppo del modello:
   - Scelta di architetture appropriate che bilanciano performance e interpretabilità
   - Monitoraggio durante l'addestramento di metriche etiche
   - Valutazione disaggregata e documentazione del modello

4. Fase di testing e valutazione:
   - Test multidimensionali che valutano non solo accuratezza ma anche fairness, robustezza, privacy
   - Adversarial testing, red teaming e user testing diversificato

5. Fase di deployment e monitoraggio:
   - Deployment graduale con monitoraggio attento
   - Feedback loops e aggiornamento responsabile

Questo approccio "ethics by design" incorpora considerazioni etiche fin dalle prime fasi di sviluppo.

Spunto di riflessione: "In quale fase del ciclo di vita pensate sia più critico incorporare considerazioni etiche? Perché?"
-->
---

## Strumenti pratici per lo sviluppo responsabile 🧰

- **Documentazione standardizzata** 📝
  - Datasheets for Datasets
  - Model Cards
- **Toolkit e librerie** 💻
  - Fairness Indicators
  - What-If Tool
  - AI Fairness 360
- **Processi e framework** 🔄
  - Responsible AI Maturity Model
  - Ethics Canvas

<!--
Diversi strumenti concreti possono supportare lo sviluppo responsabile:

1. Documentazione standardizzata:
   - Datasheets for Datasets: template per documentare caratteristiche, limitazioni e considerazioni etiche dei dataset
   - Model Cards: documentazione standardizzata di capacità, casi d'uso previsti e limitazioni dei modelli
   - Transparency Notes: documentazione accessibile per utenti finali su funzionamento e limitazioni

2. Toolkit e librerie:
   - Fairness Indicators: strumenti per valutare e visualizzare metriche di fairness
   - What-If Tool: strumento interattivo per esplorare comportamento dei modelli
   - AI Fairness 360: libreria open-source per rilevare e mitigare bias
   - Privacy-preserving NLP libraries: strumenti per implementare tecniche come federated learning o differential privacy

3. Processi e framework:
   - Responsible AI Maturity Model: framework per valutare e migliorare pratiche organizzative
   - Ethics Canvas: strumento collaborativo per mappare implicazioni etiche
   - Consequence Scanning: workshop strutturato per identificare conseguenze intenzionali e non
   - Diverse Voices: metodologia per incorporare prospettive diverse nel design

Questi strumenti rendono più concreto e praticabile lo sviluppo responsabile di sistemi NLP.

Domanda per la platea: "Quali di questi strumenti avete già utilizzato o vorreste esplorare? Quali altri strumenti conoscete che potrebbero essere utili?"
-->

---

## Collaborazione multidisciplinare 👥

![bg right:40%](https://source.unsplash.com/random/800x600/?collaboration,team)

- **Team multidisciplinari** 🧠
  - Esperti di etica, scienze sociali, legge, domain experts
- **Coinvolgimento degli stakeholder** 🤝
  - Participatory design
  - Community engagement
- **Formazione e sensibilizzazione** 📚
  - Curriculum integration
  - Leadership awareness

<!--
Lo sviluppo responsabile richiede collaborazione tra diverse discipline:

1. Team multidisciplinari:
   - Composizione diversificata: inclusione di esperti di etica, scienze sociali, legge, domain experts
   - Collaborazione strutturata: processi che facilitano dialogo tra discipline diverse
   - Linguaggio comune e rispetto reciproco per diverse forme di expertise

2. Coinvolgimento degli stakeholder:
   - Participatory design: coinvolgimento attivo degli utenti finali nel processo di design
   - Community engagement: dialogo con comunità potenzialmente impattate
   - Expert consultation e feedback iterativo

3. Formazione e sensibilizzazione:
   - Curriculum integration: incorporazione di considerazioni etiche nella formazione tecnica
   - Hands-on training e continuing education
   - Leadership awareness e cross-functional training
   - Ethics champions interni

Questa collaborazione multidisciplinare è essenziale per identificare e affrontare le complesse sfide etiche nell'NLP.

Spunto di riflessione: "Come possiamo superare le barriere linguistiche e concettuali tra diverse discipline per facilitare una collaborazione efficace?"
-->

---

# Riflessione finale 💭

### L'etica nell'NLP non è un ostacolo all'innovazione, ma una componente essenziale di un'innovazione veramente benefica e sostenibile.

<!--
Questa slide offre una riflessione conclusiva sul tema dell'etica nell'NLP. È importante sottolineare che l'etica non è un freno all'innovazione, ma piuttosto un elemento che la rende più robusta, sostenibile e benefica per la società.

Posso elaborare su questo punto evidenziando che:
- L'innovazione tecnologica ha senso solo se migliora effettivamente la vita delle persone
- Considerare le implicazioni etiche fin dall'inizio può prevenire costosi fallimenti e perdita di fiducia
- Un approccio etico può stimolare innovazioni più creative che rispondono a esigenze reali
- La fiducia del pubblico è essenziale per l'adozione diffusa delle tecnologie NLP

Questo messaggio finale invita la platea a vedere l'etica non come una checklist di conformità o un insieme di restrizioni, ma come una bussola che guida verso un'innovazione più significativa e di impatto positivo.

Posso concludere chiedendo alla platea di riflettere su come possono incorporare considerazioni etiche nel loro lavoro quotidiano, sia che siano sviluppatori, ricercatori, manager o utenti di sistemi NLP.
-->
---

## Conclusione 🎯

- I sistemi NLP incorporano valori e priorità con profonde conseguenze sociali 🌍
- I bias possono perpetuare o amplificare disuguaglianze esistenti ⚖️
- La privacy è particolarmente critica per dati linguistici personali 🔒
- La trasparenza e la spiegabilità sono essenziali per costruire fiducia 🔍
- Lo sviluppo responsabile richiede un approccio multidisciplinare 👥
- L'etica nell'NLP è un processo continuo, non una destinazione finale 🔄

<!--
In questa conclusione, riassumiamo i punti chiave discussi durante la presentazione:

I sistemi NLP non sono strumenti neutri, ma incorporano valori e priorità che possono avere profonde conseguenze sociali. Questo richiede un'attenta considerazione delle implicazioni etiche in ogni fase del loro sviluppo e utilizzo.

I bias nei sistemi NLP possono perpetuare o amplificare disuguaglianze esistenti, richiedendo tecniche specifiche di misurazione e mitigazione.

La privacy è particolarmente critica quando si tratta di dati linguistici, che spesso contengono informazioni personali e sensibili.

La trasparenza e la spiegabilità sono essenziali per costruire fiducia e permettere oversight, nonostante le sfide poste dalla complessità dei moderni sistemi NLP.

Lo sviluppo responsabile richiede un approccio multidisciplinare che integri diverse prospettive e competenze.

Infine, è importante sottolineare che l'etica nell'NLP non è una checklist da completare una volta per tutte, ma un processo continuo di riflessione, valutazione e miglioramento.

Guardando al futuro, possiamo anticipare che le questioni etiche nell'NLP diventeranno sempre più complesse con l'evoluzione della tecnologia, rendendo ancora più importante un approccio proattivo e riflessivo.

Domanda finale per la platea: "Quale aspetto dell'etica nell'NLP pensate sarà più importante affrontare nei prossimi anni? Come possiamo prepararci per queste sfide future?"
-->>


---

## 🚀 Sfide future

- Maggiore explainability
- Modelli più inclusivi
- Regolamentazione internazionale 🌐

💬 **Domanda:**  
Chi dovrebbe decidere le regole? Governi, aziende, utenti?

<!-- 
NOTE PER LO SPEAKER:
Spiega che le sfide etiche richiederanno collaborazione tra governi, aziende, ricercatori e società civile.
Domanda: “Chi dovrebbe avere l’ultima parola sulle regole etiche?”
-->

---

## ✨ Come ci ricorda Spider-Man

> “Da grandi poteri derivano grandi responsabilità” 🕷️

💬 **Domanda finale:**  
Come usereste il vostro ‘potere NLP’ per fare del bene?

<!-- 
NOTE PER LO SPEAKER:
Chiudi con leggerezza ma lascia un messaggio potente. Cita Spider-Man per essere simpatico, ma ricollegati al concetto di responsabilità.
Domanda: “Se poteste costruire un modello NLP per aiutare il mondo, cosa fareste?”
-->

---

## Riferimenti e Approfondimenti 📚

- Bender, E. M., et al. (2021). **On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?**
- Blodgett, S. L., et al. (2020). **Language (Technology) is Power: A Critical Survey of "Bias" in NLP**
- Floridi, L., & Cowls, J. (2019). **A Unified Framework of Five Principles for AI in Society**
- Weidinger, L., et al. (2021). **Ethical and social risks of harm from Language Models**

<!--
Questa slide fornisce riferimenti bibliografici chiave per chi volesse approfondire i temi trattati nella presentazione.

I riferimenti selezionati rappresentano lavori fondamentali nel campo dell'etica nell'NLP:

- Il paper di Bender et al. (noto come "Stochastic Parrots") è un lavoro influente che discute i rischi dei modelli linguistici di grandi dimensioni
- Il survey di Blodgett et al. offre una panoramica critica della ricerca sui bias nell'NLP
- Il framework di Floridi e Cowls propone cinque principi etici fondamentali per l'AI in società
- Il paper di Weidinger et al. esplora i rischi etici e sociali dei modelli linguistici

Posso menzionare che questi sono solo alcuni dei lavori più rilevanti, e che il campo è in rapida evoluzione con nuove pubblicazioni che emergono continuamente.

Potrei anche suggerire risorse online come il AI Ethics Guidelines Global Inventory, il Partnership on AI, o il Montreal AI Ethics Institute per chi volesse rimanere aggiornato sugli sviluppi in questo campo.
-->

---

## 📣 Grazie per l’attenzione!

🙏 Domande? Commenti?  
Pronti a cambiare il mondo… in modo etico!

<!-- 
NOTE PER LO SPEAKER:
Ringrazia la classe, invita a fare domande e a continuare la discussione.
-->