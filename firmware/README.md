# Firmware integration — EK-RA8P1 (Cortex-M85 + Ethos-U55)

This folder contains everything the embedded side needs:

```
firmware/
├── model/
│   ├── RUHMI_REPORT.md            ← Vela metrics, API summary
│   └── src/                       ← RUHMI/MERA generated C/H (19 files)
│       ├── model.{c,h}
│       ├── sub_0001_*.{c,h}       ← Ethos-U55 NPU subgraph
│       ├── compute_sub_{0000,0002}.{c,h}  ← CPU pre/post subgraphs
│       ├── kernel_library_*.{c,h}
│       └── ...
└── src/
    ├── hal_entry.c                ← main inference loop, RTT output
    ├── preprocess.{c,h}           ← uint8 HWC → float NCHW + ImageNet norm
    ├── test_imgs.{c,h}            ← 5 test images (one per class)
    └── README.md                  ← this file
```

## What you need in your e² studio FSP project

1. **Base project**
   - Board: **EK-RA8P1**
   - Toolchain: **GCC ARM**
   - FSP version: **6.2 or newer** (older versions lack the Ethos-U module)

2. **Stacks (FSP Configurator → Stacks tab → New Stack →)**
   - `Google TFLM Core Lib`
   - `Google TFLM CMSIS-NN Kernel`

   Adding these auto-pulls in:
   - ARM Ethos-U Core Driver  → exposes `g_ethosu0`
   - ARM CMSIS-NN Library
   - ARM CMSIS-DSP Library

3. **NPU module instance**
   - In the same Stacks tab → add module **`Ethos-U Module (rm_ethosu)`**
   - Default instance name **`g_rm_ethosu0`** (so handles become `g_rm_ethosu0_ctrl`, `g_rm_ethosu0_cfg`)
   - Watchdog timeout: 2000 ms (default) is fine for our model

4. **SEGGER RTT**
   - In Smart Configurator → add **`SEGGER RTT`** (or the `r_segger_rtt` stack if exposed)
   - No UART configuration needed; output appears in J-Link RTT Viewer

5. **OSPI flash** (because RUHMI placed weights in `.sdram_from_ospi0_cs1`)
   - Either include the **`OSPI_B_EP`** sample sources in your project, OR
   - Add OSPI driver stack and provide `ospi_b_init()` yourself
   - Define `RUN_MODEL_FROM_OSPI=1` in **Project Properties → C/C++ Build → Settings → GCC Compiler → Preprocessor → Defined symbols**

6. **Compiler defines**
   - `RUN_MODEL_FROM_OSPI=1`
   - (Optional) any debug flags you want; the code itself has no debug macros

7. **Linker script**
   - The default RA8P1 EK linker script already defines the `.sdram_from_ospi0_cs1` section.
   - Stack/heap: keep defaults; the model uses its own `sub_0001_arena[754 KiB]` placed in `.sdram`.

## Files to add to the project

Copy / link from this repo into your project's `src/` (or whatever source folder
FSP scans):

```
firmware/model/src/*.c   firmware/model/src/*.h    (19 files — RUHMI-generated)
firmware/src/hal_entry.c
firmware/src/preprocess.{c,h}
firmware/src/test_imgs.{c,h}
```

> Remove FSP's auto-generated stub `hal_entry.c` if present — ours replaces it.

## Build & flash

1. **Project → Build All** (Ctrl+B)
2. **Run → Debug As → Renesas GDB Hardware Debugging**
3. Make sure J-Link is selected and the board is connected (USB to J33).
4. **Open J-Link RTT Viewer**, connect to the running target (R7KA8P1KFLCBG → SWD → 4 MHz).
5. Reset; you should see:

```
=== Melon Disease Edge Classifier (RA8P1 + Ethos-U55) ===

[img 0] 노균병/S037-FM03-037-2022-09-20-000058  ground-truth=노균병
  pred = 노균병  conf = 99%  [OK]
  cycles: total=NNNNNN  npu_active=NNNNNN  npu_idle=NNNN  axi0=NNNN  axi1=NNNN
  cpu_other = NNNN (= total - npu_active)
...
=== summary  acc=5/5  avg_cpu=NNNNNN  avg_npu_active=NNNNNN cycles ===
```

## Cycles → milliseconds

- M85 core at 1 GHz → 1 cycle = 1 ns; multiply cycles by 1e-6 for ms.
- Ethos-U55 PMU runs at half-rate, **already pre-multiplied by 2** in the print code.
- Vela predicted: ~3.73 M total cycles ≈ **7.47 ms / inference**.

## Troubleshooting

| symptom | fix |
|---|---|
| Build error: `ethosu_driver.h: No such file` | TFLM CMSIS-NN stack not added (see step 2) |
| Build error: `g_rm_ethosu0_cfg undeclared` | `rm_ethosu` module not added with that exact instance name (step 3) |
| Build error: `ospi_b_init undeclared` | Need to add OSPI_B sources or remove the `RUN_MODEL_FROM_OSPI` define |
| Boots but RTT shows nothing | RTT not opened; in J-Link RTT Viewer, configure Search Range to match the linker (usually auto OK) |
| `RM_ETHOSU_Open failed` | NPU clock / power not enabled by FSP — re-run "Generate Project Content" |
| All predictions wrong (random) | Image preprocessing mismatch; verify mean/std in `preprocess.c` matches training |
| `pred OK` but cycles=0 | PMU events not enabled; check that `ETHOSU_PMU_Enable` was called before `RunModel` |

## Linking notes for the 14 MB `sub_0001_model_data.c`

This file contains the NPU-encoded weight blob (`.sdram_from_ospi0_cs1` section, 2.3 MB binary).

- If GCC complains about "section .sdram_from_ospi0_cs1 not found in linker script", you need either:
  - Use the **default EK-RA8P1 linker script** (it already defines this section), OR
  - Manually add to your custom script:
    ```
    .sdram_from_ospi0_cs1 :
    {
        . = ALIGN(16);
        KEEP(*(.sdram_from_ospi0_cs1*))
        . = ALIGN(16);
    } > OSPI0_CS1
    ```
