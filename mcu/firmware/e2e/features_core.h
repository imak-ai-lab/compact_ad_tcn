#ifndef FEATURES_CORE_H
#define FEATURES_CORE_H
#include "e2e_data.h"

#define N_CH 3U
#define N_FEAT 9U
#define N_FEAT_ALL 27U
#define ROWS_IN 120U
#define ROWS_TGT 240U
#ifndef BENCH_FORECAST
#define BENCH_FORECAST 0
#endif
#define RAW_MAX (BENCH_FORECAST ? BLOCK_TGT_N : BLOCK_IN_N)

extern float32_t g_block[N_CH][RAW_MAX + 2U];
extern float32_t g_rows_in[ROWS_IN][N_FEAT_ALL];
#if BENCH_FORECAST
extern float32_t g_rows_tgt[ROWS_TGT][N_FEAT_ALL];
#endif

void features_init(void);
void features_channel_denoise(uint32_t ch, uint32_t n);
void features_channel_rows(uint32_t ch, uint32_t rows, float32_t (*out)[N_FEAT_ALL]);
float32_t score_window(const float32_t *pred_cf, const float32_t (*target)[N_FEAT_ALL], uint32_t t_len);

#endif
