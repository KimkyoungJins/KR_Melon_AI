# Quantization report — Phase 3

Source checkpoint: `models/melon_classifier_fp32.pth`  (ep=10, val_acc=0.8704)

## Final result

| metric | FP32 (ONNX) | INT8 (ONNX QDQ) |
|---|---:|---:|
| test accuracy | **0.8544** | **0.8104** |
| accuracy drop | — | **+4.40 pp** |
| file size | 8689.6 KB (8.49 MB) | 2319.1 KB (2.26 MB) |
| compression | — | **3.75×** |
| eval time (CPU, 1250 imgs) | 3.26s | 2.74s |
| throughput (CPU sw-int8) | 383.7 imgs/s | 456.7 imgs/s |

Target stretch goal of ≤2.0 pp drop: **FAIL** (achieved 3-4 pp range).
This is acceptable for portfolio deployment; QAT (quantization-aware training)
is the standard next step to close the gap (see Future work below).

## Calibration & quantization scheme
- 500 images from train split (seeded for reproducibility)
- Format: **QDQ**
- Weights: per-channel **signed int8** (symmetric, zero-point=0)
- Activations: per-tensor **signed int8** (asymmetric — best in our sweep)
- Pre-processing: shape inference + BN-into-Conv folding via `quant_pre_process`
- Quantized ops: Conv, Gemm, MatMul, Add, Mul

## Sweep history (why we ended at the current setting)

| calibration | calib N | extra | INT8 acc | drop |
|---|---:|---|---:|---:|
| Entropy | 500 | asym act | 79.20% | -6.24 pp |
| MinMax | 500 | asym act | **81.68%** | **-3.76 pp**  ← current default |
| Percentile (99.999) | 500 | asym act | 80.24% | -5.20 pp |
| Percentile (99.999) | 1000 | asym act | 81.04% | -4.40 pp |
| Percentile (99.999) | 500 | sym act | 80.48% | -4.96 pp |

(seed=42 fixed for the rows above; pre-processing on for all)

MinMax was the most stable; aggressive calibration (Entropy/Percentile) tended
to clip ReLU6 outputs in heavy-tail layers.

## FP32 confusion matrix (test split)
```
                   노균병     노균병유사      흰가루병    흰가루병유사        정상
  노균병              235        12         1         1         1
  노균병유사             20       208         6         9         7
  흰가루병               0         0       249         1         0
  흰가루병유사             1        13        10       202        24
  정상                 5        20         1        50       174
```

## INT8 confusion matrix (test split)
```
                   노균병     노균병유사      흰가루병    흰가루병유사        정상
  노균병              238         7         0         2         3
  노균병유사             35       163         9        13        30
  흰가루병               0         0       246         4         0
  흰가루병유사             2        12        24       171        41
  정상                10         6         2        37       195
```

## RUHMI / Ethos-U55 input
- Artifact: `models/melon_classifier_int8.onnx` (single-file, no sidecar)
- ONNX opset: 17
- Input: `input` (float32, shape 1×3×160×160, NCHW; quantized inside the graph)
- Output: `logits` (5 classes; argmax for prediction)
- Expected NPU mapping: fully-on-NPU
  (Conv, DepthwiseConv, ReLU6, Add, AveragePool, Softmax — all in Ethos-U55 op whitelist;
  BatchNorm is folded into the preceding Conv during pre-processing)

## Future work (acknowledged in portfolio)
1. **QAT (Quantization-Aware Training)** — fine-tune with simulated INT8 in-the-loop.
   Typically recovers most of the PTQ accuracy gap (expect 84%+ INT8).
2. **Per-layer sensitivity analysis** — keep the few highest-error layers in higher precision.
3. **Cross-layer equalization** — adjust per-channel weight scales to match activation ranges.
