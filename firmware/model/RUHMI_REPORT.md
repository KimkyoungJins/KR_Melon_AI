# RUHMI / Ethos-U55 Compile Report

Compiled from `models/melon_classifier_fp32.onnx` with MERA 2.6.0 inside an
x86_64 Docker container (Ubuntu 22.04, MERA's official wheel) running on Apple
Silicon via Rosetta 2.

## Tooling
- RUHMI Framework MCU: https://github.com/renesas/ruhmi-framework-mcu
- MERA backend: 2.6.0+pkg.4513 (manylinux_2_27_x86_64 wheel)
- `mcu_compile.py --npu --memory-mode Shared_Sram --optimization Performance --quantize --calib-data <100 npy> --calib-num 100`

## Quantization (MERA-internal PTQ)
- Calibration: 100 train images (1×3×160×160 float32 npy), seeded
- Result quality (MERA's own metric): **PSNR=16.45, Score=97.73**
- Note: We use MERA's quantizer (not our PyTorch INT8 ONNX) because MERA's ONNX
  frontend rejected both QDQ format (DequantizeLinear unsupported) and ORT's
  QOperator format ("Only the default ONNX operator set is supported").
  Our `melon_classifier_int8.onnx` is kept as a portable reference.

## Subgraph layout (3 subgraphs total)
```
sub_0000  CPU         (preprocessing — Quantize float→int8, layout transform)
sub_0001  Ethos-U55   ★ MobileNetV2 body (Conv/DWConv/Add/Pool/Softmax)
sub_0002  CPU         (postprocessing — Dequantize int8→float, output reshape)
```
The MobileNetV2 trunk runs **fully on NPU**; CPU subgraphs are tiny IO wrappers.

## Ethos-U55 metrics (Vela)
| metric | value |
|---|---|
| Accelerator | Ethos-U55 with 256 MACs/cycle |
| System config | RA8P1 |
| Memory mode | Shared_Sram |
| Core clock | 500 MHz |
| **Inference time** | **7.467 ms** |
| **Inferences / second** | **133.9 fps** |
| NPU cycles | 2,174,617 |
| SRAM access cycles | 3,484,116 |
| Off-chip flash access cycles | 1,820 |
| Total cycles | 3,733,713 |
| Total MACs / inference | 152,919,680 |
| TOPS @ 500 MHz | 0.041 |

## Memory footprint
| region | usage |
|---|---|
| **External SIP flash (weights)** | **2,323 KiB** |
| **SRAM (activations + arena)** | **754 KiB** |
| Total original weights | 2,196,160 B |
| NPU-encoded weights | 2,024,960 B |
| Arena cache size | 4,194,304 B (4 MB available) |

The 754 KiB SRAM footprint fits comfortably in the RA8P1's 2 MB ECC SRAM.
External SIP flash 4–8 MB hosts the 2.3 MB weight blob with plenty of headroom
for code, test images, and FSP/driver overhead.

## Generated C runtime API (`firmware/model/src/model.h`)

```c
#include <stdbool.h>
#include "sub_0001_tensors.h"

extern uint8_t sub_0001_arena[kArenaSize_sub_0001];

// Buffers used internally / for IO
extern float   buf_input[76800];             // 1×3×160×160 input (NCHW float)
extern int8_t  buf_logits_70207_10553[5];    // quantized intermediate
extern float   buf_logits_70207[5];          // 5-class float logits (output)

void   RunModel(bool clean_outputs);
float* GetModelInputPtr_input();             // points into buf_input
float* GetModelOutputPtr_logits_70207();     // points into buf_logits_70207
```

Inference sequence:
```c
float *in  = GetModelInputPtr_input();   // 76800 floats, NCHW normalized
// ... fill `in` with letterboxed, ImageNet-normalized image ...
RunModel(false);
float *out = GetModelOutputPtr_logits_70207();
// argmax over out[0..4] → predicted class
```

## Files (`firmware/model/src/`)
| file | role | size |
|---|---|---|
| `model.h` / `model.c` | top-level orchestration (`RunModel`) | 3 KB / 4 KB |
| `compute_sub_0000.*` | CPU preprocessing subgraph | 4 KB |
| `sub_0001_invoke.*` | NPU subgraph driver | 2 KB |
| `sub_0001_command_stream.*` | Ethos-U55 command stream | 76 KB |
| `sub_0001_model_data.*` | NPU-encoded INT8 weights | **14 MB source / 2 MB binary** |
| `sub_0001_tensors.*` | NPU tensor / arena metadata | 1 KB |
| `compute_sub_0002.*` | CPU postprocessing subgraph | 4 KB |
| `kernel_library_int.*` | INT8 kernel library (CPU side) | 96 KB |
| `kernel_library_utils.*` | shared utils | 30 KB |
| `ethosu_common.h` | driver glue | 1 KB |
