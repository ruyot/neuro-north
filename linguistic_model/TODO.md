## Linguistic Model

### Inputs

* Probabilities corresponding to \(N\) ranges of letters

  * e.g. `[A-F]: 0.87, [G-L]: 0.13, [M-R]: 0.0, [S-Z]: 0.0`
* Previous range-selection probabilities for the current word
* Confirmed previous words / sentence context
* Optional explicit word-boundary signal (`space`)

### Outputs

* Ranked candidate words with probabilities

  * e.g. `{"apple": 0.61, "after": 0.21, "again": 0.08, ...}`
* Top \(K\) word completions for display
* Confidence / ambiguity score

  * determines whether to suggest a word or request another range selection
* On word confirmation:

  * most likely reconstructed word
  * updated sentence context

### Model

Combine EEG evidence with a language prior:

$$
P(w\mid R,C)\propto P(R\mid w)\,P(w\mid C)
$$

where:

* \(R\) = range-selection probability history
* \(C\) = confirmed linguistic context
* \(P(R\mid w)\) = compatibility of the word's letters with the observed EEG range probabilities
* \(P(w\mid C)\) = contextual probability from the language model

For example, if the observed selections are:

```text
1: A-F  0.87
2: M-R  0.74
3: M-R  0.81
```

then for candidate word \(w\):

$$
P(R\mid w)
=
\prod_i P_i(\operatorname{range}(w_i))
$$

The critical point is that **the model preserves uncertainty from every SSVEP selection** rather than turning each one into a hard letter-range choice. That lets the linguistic model correct marginal EEG mistakes when the sentence context strongly supports a particular word.

### MVP implementation

The standalone decoder lives in `linguistic_model`. It currently provides:

* configurable, disjoint letter ranges and a prefix trie
* calibrated range-posterior input with classifier-prior correction
* page-aware observations (only displayed ranges can match)
* interpolated unigram/bigram context scoring
* ranked completions, normalized probabilities, entropy, and top-two margin
* explicit word-boundary confirmation, completion acceptance, rejection, and undo

The EEG/SSVEP layer should call `LinguisticDecoder.observe` only for a committed
selection event. It must pass calibrated probabilities for every range displayed
on that page, rather than a CCA score, argmax, or text-formatted diagnostic.
