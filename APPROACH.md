# How we built this submission

[Leer en español](APPROACH.es.md) · [Back to the README](README.md)

The contest asks for an 8-class classifier of 4 × 256 time series, scored one sample at a time with a geometric mean over classes and a speed term that grows with the square root of the number of answers in 60 s. Two things mattered most: **generalising to the test distribution** and **answering in well under a millisecond**.

## 1. Understanding the distribution shift

The official baseline reached 98% on train but only 43% on the demo. Comparing train with the demo showed why:

- **Shortcuts that only exist in train:** the `level` metadata was shifted for classes 3 and 5, channel 3 carried a class-specific offset in classes 2 and 6, and about half of class 6 formed an isolated subgroup (channel 2 copying channel 1) that never appears in test.
- **A dirtier test set:** about 1.3× more noise, twice as many missing values, more frozen stretches and spikes in 18% of the samples.
- **What does not change:** channel roles, metadata values, and the fact that classes are defined in Hz (`sample_rate` halves the frequency in cycles per sample).

Removing the shortcuts alone lifted the baseline from 43% to 54% on the demo.

## 2. Robust inputs and a test-like simulator

Every sample goes through the same robust cleaning (spikes and frozen stretches to NaN, interpolation, per-channel median/IQR, common 1 Hz grid). For training we built a **corruption simulator** calibrated on the demo (extra noise, spikes, isolated and block NaNs, frozen stretches), trained with balanced classes, dropped the isolated class-6 subgroup, and validated on a corrupted hold-out of 14,250 samples × 4 corrupted versions.

## 3. Teachers on Kaggle, fast students in Numba

Strong models are far too slow for this metric, so we used **knowledge distillation**:

- **Teachers** (offline, Kaggle T4 GPUs, 5-fold out-of-fold logits): InceptionTime with 5 seeds per fold (89.0% on the demo) and ConvTran (85.8%). Their combination reached 90.0%.
- **Students:** MiniRocket features, whose fixed ±1/2 kernels are almost free in Numba, plus a small MLP. The submitted student uses 250 + 3 features and a 128-unit hidden layer, distilled from an ensemble of three 256-unit MLP students with 8 extra corrupted passes labelled out-of-fold by InceptionTime.
- **Inference engine:** one `@njit` call per sample with pre-allocated buffers, compiled and warmed up in `Model()`. It reproduces the PyTorch predictions exactly.

## 4. Choosing by the metric

Without abstentions each class term depends only on its recall, and doubling the speed multiplies the score by about 1.41. We therefore compared candidates by `√(N/8)·gm_f`, using the corrupted-validation gm_f (reliable) and latency measured in Colab with the official harness (33 models, interleaved rounds). The submitted model was the only one in the top two under both validation and demo, its recipe was stable across three seeds (gm_f 0.518–0.523), and its latency was confirmed in three Colab rounds.

## 5. What did not help

| Idea | Why we dropped it |
|---|---|
| BLAS in the MLP | Exact but no faster: the MLP is only 2–5% of the latency |
| Per-class biases tuned on validation | No gain: the students were already balanced on train-derived data |
| Extra MiniRocket pooling (MPV, MIPV) and teacher-guided feature selection | Better validation, worse demo, especially class 6 |
| Teachers from the earlier cross-validation | They had seen most of the data, so their soft labels were over-confident |
| ConvTran as an extra teacher | The teacher improved, but the MiniRocket student could not learn the extra signal |
| Tiny CNN student (~2 k weights) | Not faster than MiniRocket and clearly less accurate at that size |

See [Future work](README.md#future-work) for the directions we would explore next.
