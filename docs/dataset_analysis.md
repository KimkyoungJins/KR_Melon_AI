# Dataset analysis — AIHub 지능형 스마트팜(참외)

Source: `data/labels`

**Total labeled images: 211,284**

## 1. Class distribution

| class | train | val | total |
|---|---:|---:|---:|
| 노균병 | 39,151 | 4,943 | **44,094** |
| 노균병유사 | 32,335 | 4,063 | **36,398** |
| 정상 | 68,206 | 8,553 | **76,759** |
| 흰가루병 | 32,247 | 3,900 | **36,147** |
| 흰가루병유사 | 15,873 | 2,013 | **17,886** |
| **total** | 187,812 | 23,472 | **211,284** |

Imbalance ratio (max/min) = 4.29× — smallest class: 흰가루병유사 (17,886).

## 2. Image dimensions

- width  : min=489, median=1248, mean=1289, max=4204
- height : min=405, median=1185, mean=1218, max=4498

Top unique (W×H) combinations:

  - 1018×1011  →  252  (0.1%)
  - 1012×1005  →  68  (0.0%)
  - 1048×1041  →  64  (0.0%)
  - 1010×1002  →  49  (0.0%)
  - 1000×1031  →  38  (0.0%)
  - 1005×1015  →  35  (0.0%)
  - 1051×1043  →  32  (0.0%)
  - 1008×1031  →  29  (0.0%)

## 3. Day vs night (image.day_section)

| class | 주간 | 야간 | 기타 |
|---|---:|---:|---:|
| 노균병 | 44,094 | 0 | 0 |
| 노균병유사 | 36,398 | 0 | 0 |
| 정상 | 30,483 | 46,276 | 0 |
| 흰가루병 | 27,941 | 8,206 | 0 |
| 흰가루병유사 | 1,790 | 16,096 | 0 |

## 4. Farm-level split leakage check

- Distinct farms (frm_id): **12**
- Farms appearing in BOTH train and val: **12**

  ⚠ Same farms exist in both splits — for proper generalization eval,
  consider grouping by `frm_id` when re-splitting.

    - FM01 → ['train', 'val']
    - FM03 → ['train', 'val']
    - FM02 → ['train', 'val']
    - FM09 → ['train', 'val']
    - FM04 → ['train', 'val']
    - FM10 → ['train', 'val']
    - FM05 → ['train', 'val']
    - FM08 → ['train', 'val']
    - FM06 → ['train', 'val']
    - FM12 → ['train', 'val']

## 5. Site (indvd_code) diversity per class (top 5)

- **노균병** — distinct sites: 112;  top: S215(1,373), S231(1,368), S206(1,135), S214(1,107), S209(1,100)
- **노균병유사** — distinct sites: 149;  top: S002(1,350), S137(1,265), S004(1,263), S149(1,221), S040(1,145)
- **정상** — distinct sites: 187;  top: S057(1,969), S058(1,850), S045(1,689), S041(1,680), S056(1,601)
- **흰가루병** — distinct sites: 89;  top: S306(2,632), S055(1,951), S349(1,531), S348(1,485), S307(1,336)
- **흰가루병유사** — distinct sites: 143;  top: S194(1,026), S153(973), S119(888), S044(821), S137(639)

## 6. Environment sensors (sanity)

- inner temperature : min=3.7, median=27.5, max=57.4, σ=9.6
- inner humidity    : min=12.3, median=77.7, max=100.0, σ=24.4
