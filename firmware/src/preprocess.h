#ifndef MELON_PREPROCESS_H
#define MELON_PREPROCESS_H

#include <stdint.h>

/* uint8 HWC (160*160*3) → float32 NCHW (3*160*160), ImageNet-normalized.
 *
 * Matches PyTorch's transforms.Compose([ToTensor(), Normalize(mean, std)]):
 *   v_float = (uint8 / 255.0 - mean[c]) / std[c]
 * with mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225).
 *
 * Layouts:
 *   src: H,W,C order (matches MELON_TEST_IMGS[] from test_imgs.h)
 *   dst: C,H,W order (matches RUHMI's buf_input[76800])
 */
void melon_preprocess_u8_hwc_to_f32_nchw(const uint8_t *src, float *dst);

#endif /* MELON_PREPROCESS_H */
