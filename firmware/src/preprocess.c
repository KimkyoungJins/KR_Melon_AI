#include "preprocess.h"

#define H 160
#define W 160
#define C 3

/* ImageNet stats — same as training (PyTorch torchvision.transforms.Normalize). */
static const float MEAN[C] = { 0.485f, 0.456f, 0.406f };
static const float STD[C]  = { 0.229f, 0.224f, 0.225f };

void melon_preprocess_u8_hwc_to_f32_nchw(const uint8_t *src, float *dst)
{
    /* Precompute 1/(255*std) so the inner loop is just (px*scale - bias). */
    float scale[C], bias[C];
    for (int c = 0; c < C; c++) {
        scale[c] = 1.0f / (255.0f * STD[c]);
        bias[c]  = MEAN[c] / STD[c];
    }

    for (int c = 0; c < C; c++) {
        float s = scale[c];
        float b = bias[c];
        float *dst_c = dst + c * (H * W);
        for (int y = 0; y < H; y++) {
            const uint8_t *row = src + (y * W) * C;
            float *out_row = dst_c + y * W;
            for (int x = 0; x < W; x++) {
                out_row[x] = (float)row[x * C + c] * s - b;
            }
        }
    }
}
