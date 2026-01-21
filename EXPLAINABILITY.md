
# 1. Metodi Basati sui Gradienti (Gradient-Based)
Questi metodi sfruttano la struttura interna della rete ("White Box") calcolando il gradiente dell'output rispetto ai pixel di input.

## Saliency (Vanilla Gradients)
* **Metodo nel codice:** `get_saliency`
* **Concetto:** Calcola il gradiente assoluto della classe target rispetto ai pixel dell'immagine.
  $$Attribution = \left| \frac{\partial Output}{\partial Input} \right|$$
* **Pro:** Computazionalmente molto leggero (richiede solo una backpropagation).
* **Contro:** Le mappe sono spesso rumorose e soffrono del problema della "Saturazione del Gradiente" (se il modello è molto sicuro, il gradiente può essere zero anche per feature importanti).

#### Key points
- Funzione: Output Score della classe target $y_c$ (di solito i logits, prima della Softmax).
- Rispetto a cosa: All'Input ($x$, ovvero i pixel).
- Calcolo: $\frac{\partial y_c}{\partial x}$

#### Simple explanation
$L = f(\text{pixel})$,
Significa che quel numero "L" esiste solo perché il pixel aveva un certo valore (es. "0.5").Fare la derivata $\frac{\partial L}{\partial \text{pixel}}$ significa porsi una domanda ipotetica ("What if?"):"Se il valore del pixel cambiasse di una quantità piccolissima ($\epsilon$), di quanto cambierebbe il valore del Logit?"

## Input × Gradient
* **Metodo nel codice:** `get_input_gradients`
* **Concetto:** Pondera il gradiente moltiplicandolo per l'intensità del pixel di input.
  $$Attribution = Input \odot \frac{\partial Output}{\partial Input}$$
* **Pro:** Riduce il rumore rispetto alla Saliency semplice e tende a evidenziare meglio le strutture dell'oggetto.
* **Contro:** Se un pixel è nero (valore 0), l'attribuzione sarà 0 anche se quell'area è rilevante per il modello.

#### Simple explanation
Se il pixel è Nero (0):$$0 \cdot \text{Gradiente Alto} = \mathbf{0}$$(Ignora il pixel. Anche se il modello è sensibile lì, il segnale non c'è, quindi non ha contribuito).

## Integrated Gradients (IG) TODO: da cocnludere IG
* **Metodo nel codice:** `get_integrated_gradients`
* **Concetto:** Risolve il problema della saturazione del gradiente (Vanishing the gradient, specifically using sigmoid function). Calcola l'integrale dei gradienti lungo un percorso lineare che va da una **Baseline** (solitamente un'immagine nera) all'immagine di input.
* **Matematica:**
  $$IG(x) = (x - x') \times \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha \times (x - x'))}{\partial x} d\alpha$$
* **Pro:** Soddisfa l'assioma di completezza (la somma delle attribuzioni eguaglia la differenza di output tra input e baseline). È considerato lo standard attuale per le reti neurali.
* **Contro:** Più lento della Saliency (richiede `n_steps` passaggi forward/backward).
---

# 2. Metodi Basati su Perturbazione (Perturbation-Based)
Questi metodi modificano l'input e osservano come cambia l'output ("Black Box").

## Occlusion
* **Metodo nel codice:** `get_occlusion`
* **Concetto:** Fa scorrere una finestra grigia (patch) sopra l'immagine. Se coprendo una zona la probabilità della classe corretta crolla, quella zona è considerata importante.
* **Pro:** Molto intuitivo e facile da capire ("Cosa succede se nascondo questo pezzo?").
* **Contro:** Computazionalmente costoso; non cattura le relazioni tra feature distanti.

### Simple explanation
Immagina di avere una toppa grigia (un quadratino di pixel, ad esempio $15 \times 15$).
L'algoritmo esegue questi passaggi meccanici:
- Posizionamento: Mette la toppa nell'angolo in alto a sinistra dell'immagine ($x, y$).
- Inferenza: Passa questa immagine "censurata" al modello.Misurazione: Registra lo Score della classe target (es. "Cane").
Esempio: Senza toppa lo score era 0.99. Con la toppa lì, scende a 0.95.
- Differenza: 0.04 (Poco importante).
- Spostamento (Stride): Sposta la toppa di un tot di pixel a destra (lo stride, ad esempio 8 pixel).
- Ripetizione: Ripete il calcolo.Esempio: Ora la toppa copre il muso del cane. Lo score crolla a 0.10.Differenza: 0.89 (Importantissimo!).
- Mappa Finale: Alla fine, assegna a ogni pixel il valore del calo di probabilità registrato quando quel pixel era coperto.

## LIME (Local Interpretable Model-agnostic Explanations)
* **Metodo nel codice:** `get_lime`
* **Concetto:** Approssima il comportamento della rete neurale localmente utilizzando un modello lineare interpretabile.
* **Funzionamento:**
  1. Divide l'immagine in "Superpixels" (segmenti) `n_segments` usando l'algoritmo **SLIC**.
  2. Genera variazioni dell'immagine `n_samples` accendendo/spegnendo casualmente i segmenti.
  3. Addestra un modello lineare semplice per predire l'output della rete su queste variazioni.
* **Pro:** Agnostico rispetto al modello.
* **Contro:** Molto lento (richiede centinaia di inferenze). I risultati possono variare leggermente a causa del campionamento casuale.

### Simple explanation
- Genera centinaia di nuove immagini "finte". Accendendo e spegnendo a caso questi superpixel.
- Passa tutte queste 1000 immagini "bucate" alla rete neurale originale e registra la probabilità di "Cane" per ognuna.
 $$0.90 = 1 \cdot \text{seg.1} + 0 \cdot \text{seg.1} + 1 \cdot \text{seg.3}$$
 $$0.90 = 0 \cdot \text{seg.1} + 1 \cdot \text{seg.1} + 1 \cdot \text{seg.3}$$
 $$0.90 = 1 \cdot \text{seg.1} + 1 \cdot \text{seg.1} + 0 \cdot \text{seg.3}$$
- LIME addestra un Modello Lineare Semplice (Lsso Regression nel nostro caso).
 $$Score = 0.8 \cdot (\text{seg.1}) + 0.01 \cdot (\text{seg.2}) - 0.3 \cdot (\text{seg.3})$$

---

# 3. Metodi Basati su SHAP (Game Theory)
Metodi basati sui Valori di Shapley (Teoria dei Giochi) per distribuire equamente il "contributo" di ogni feature.

## Kernel SHAP
* **Metodo nel codice:** `get_kernel_shap`
* **Concetto:** Simile a LIME (usa segmentazione SLIC e perturbazioni), ma pesa i campioni utilizzando lo "Shapley Kernel".
* **Pro:** Fondamenta teoriche solide che garantiscono proprietà di equità nell'attribuzione.
* **Contro:** Estremamente lento, computazionalmente oneroso.

### Key points
LIME: Pesa i campioni in base alla distanza visiva (più l'immagine assomiglia all'originale, più pesa). È un approccio euristico.

Kernel SHAP: Pesa i campioni in base alla combinatoria. Dà un peso altissimo ai casi "estremi":
- Quando c'è attivo un solo superpixel (per isolare il suo contributo puro).
- Quando sono attivi tutti tranne uno (per vedere quanto si perde togliendolo). 
- I casi intermedi (metà accesi/metà spenti) hanno un peso molto basso perché sono "rumorosi".

## Gradient SHAP
* **Metodo nel codice:** `get_gradient_shap`
* **Concetto:** Un'approssimazione di SHAP che combina l'approccio teorico con i gradienti della rete per maggiore velocità. Invece di spegnere segmenti, aggiunge rumore gaussiano all'input e calcola la media dei gradienti.
* **Pro:** Più veloce di Kernel SHAP per reti profonde. Spesso produce mappe più nitide.

---

# 4. Tecniche di Robustezza (Noise Tunnel)
Tecniche per migliorare la qualità visiva delle mappe di attribuzione.

## SmoothGrad (Noise Tunnel)
* **Metodi nel codice:** `get_integrated_gradients_with_noise`, `get_saliency_with_noise`
* **Concetto:** Aggiunge rumore gaussiano all'immagine di input $N$ volte (`nt_samples`), calcola l'attribuzione per ogni copia rumorosa e ne fa la media.
* **Beneficio:** Rimuove il "rumore" ad alta frequenza nelle mappe di gradiente, rendendo l'attribuzione visivamente più pulita e focalizzata sull'oggetto.

### Simple explanation

I metodi basati sui gradienti (specialmente la Saliency semplice) soffrono di un problema chiamato fluttuazione locale. La rete neurale è una funzione matematica molto frastagliata. Un pixel potrebbe avere un gradiente positivo altissimo, e il pixel subito accanto (identico all'occhio umano) potrebbe avere un gradiente negativo o zero solo perché la funzione ha un piccolo picco matematico insignificante in quel punto.

Risultato visivo: Le mappe di attribuzione appaiono "sgranate", piene di puntini sparsi e rumore di fondo che distraggono dalle feature vere.

$$M_{smooth}(x) = \frac{1}{n} \sum_{i=1}^{n} M(x + \text{noise}_i)$$
---

## Tabella Comparativa Rapida

| Metodo | Tipo | Velocità | Qualità Visiva | Utilizzo Consigliato |
| :--- | :--- | :--- | :--- | :--- |
| **[MAYBE with noise] Saliency** | Gradiente | Molto Alta | Bassa (Rumorosa) | Debug rapido, controllo base |
| **[THIS] Input × Gradient** | Gradiente | Molto Alta | Media | Baseline migliore di Saliency |
| **[THIS] Integrated Gradients** | Gradiente | Media | Alta | Standard industriale (White-box) |
| **Gradient SHAP** | Ibrido | Media | Alta | Ottima alternativa a IG |
| **[MAYBE] Occlusion** | Perturbazione | Bassa | Bassa (a blocchi) | Verifica intuitiva "What-if" |
| **[THIS] LIME** | Perturbazione | Molto Bassa | Media (a segmenti) | Spiegazioni Black-box generiche |
| **[THIS] Kernel SHAP** | Perturbazione | Molto Bassa | Media/Alta | Quando serve rigore teorico (Black-box) |
| **[MAYBE] Noise Tunnel** | Wrapper | Bassa | Molto Alta | Per report e visualizzazioni pulite |
