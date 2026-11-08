/* Phase 5 / stage 4e: signal processing of the end-to-end firmware, compiled for speed (-O3), as stage 5.
 *
 * Denoising and features: the same algorithm as firmware/features/main.c (threshold V1), generalised to the block length, because a
 * forecasting model also needs the features of its target: 240 future rows, a block of 5168 samples.
 * Scoring: features.py and scoring.py: mean absolute error per channel over time, then e' inv_cov e.
 *
 * Memory: the raw block and the feature rows live in AXI SRAM next to the activations (.ai_pool; DTCM cannot hold them next to the network
 * and an ADC DMA could not reach DTCM anyway); the FIR work buffers stay in DTCM.
 */
#include <math.h>
#include <string.h>
#include "arm_math.h"
#include "arm_const_structs.h"
#include "features_core.h"

#define WIN 64U
#define HOP 16U
#define TRIM 640U
#define FS_HZ 40000.0f
#define MAX_LEVEL 6U

__attribute__((section(".ai_pool"), aligned(32))) float32_t g_block[N_CH][RAW_MAX + 2U];
__attribute__((section(".ai_pool"), aligned(32))) float32_t g_rows_in[ROWS_IN][N_FEAT_ALL];
#if BENCH_FORECAST
__attribute__((section(".ai_pool"), aligned(32))) float32_t g_rows_tgt[ROWS_TGT][N_FEAT_ALL];
#endif
__attribute__((aligned(32))) float32_t g_coef[RAW_MAX];
__attribute__((aligned(32))) float32_t g_ext[RAW_MAX + 64U];
float32_t g_fft_in[WIN], g_fft_out[WIN], g_wext[WIN + 32U], g_a1[WIN];
static arm_rfft_fast_instance_f32 g_fft;

typedef struct {
  uint32_t F, level;
  const float32_t *dlr, *dhr;
  const float32_t *rle, *rlo, *rhe, *rho;
} wav_t;
static const wav_t W_SYM8 = {16U, 6U, sym8_dlr, sym8_dhr, sym8_rle, sym8_rlo, sym8_rhe, sym8_rho};
static const wav_t W_DB4 = {8U, 4U, db4_dlr, db4_dhr, db4_rle, db4_rlo, db4_rhe, db4_rho};
static const wav_t *const CH_WAV[N_CH] = {&W_SYM8, &W_SYM8, &W_DB4};

void features_init(void)
{
  g_fft.Sint = arm_cfft_sR_f32_len32;
  g_fft.fftLenRFFT = (uint16_t)WIN;
  g_fft.pTwiddleRFFT = twiddleCoef_rfft_64;
}

static uint32_t dwt_per(float32_t *x, uint32_t n, const wav_t *w, float32_t *cd)
{
  const uint32_t F = w->F, n2 = n + (n & 1U), pre = F - 1U, post = F / 2U;
  if (n & 1U) {
    x[n] = x[n - 1U];
  }
  memcpy(g_ext, &x[n2 - pre], pre * sizeof(float32_t));
  memcpy(&g_ext[pre], x, n2 * sizeof(float32_t));
  memcpy(&g_ext[pre + n2], x, post * sizeof(float32_t));
  for (uint32_t o = 0; o < n2 / 2U; o++) {
    const float32_t *p = &g_ext[F / 2U + 2U * o];
    arm_dot_prod_f32(w->dlr, p, F, &x[o]);
    arm_dot_prod_f32(w->dhr, p, F, &cd[o]);
  }
  return n2 / 2U;
}

static void idwt_per(float32_t *a, const float32_t *d, uint32_t M, const wav_t *w)
{
  const uint32_t hF = w->F / 2U, pre = hF - 1U, post = w->F / 4U, len = pre + M + post;
  float32_t *ea = g_ext, *ed = &g_ext[len];
  memcpy(ea, &a[M - pre], pre * sizeof(float32_t));
  memcpy(&ea[pre], a, M * sizeof(float32_t));
  memcpy(&ea[pre + M], a, post * sizeof(float32_t));
  memcpy(ed, &d[M - pre], pre * sizeof(float32_t));
  memcpy(&ed[pre], d, M * sizeof(float32_t));
  memcpy(&ed[pre + M], d, post * sizeof(float32_t));
  for (uint32_t t = 0; t < M; t++) {
    const uint32_t i = w->F / 4U - 1U + t;
    float32_t s0, s1, s2, s3;
    arm_dot_prod_f32(w->rle, &ea[i], hF, &s0);
    arm_dot_prod_f32(w->rhe, &ed[i], hF, &s1);
    arm_dot_prod_f32(w->rlo, &ea[i], hF, &s2);
    arm_dot_prod_f32(w->rho, &ed[i], hF, &s3);
    a[t == 0U ? 2U * M - 1U : 2U * t - 1U] = s0 + s1;
    a[2U * t] = s2 + s3;
  }
}

static float32_t kth_smallest(float32_t *a, uint32_t n, uint32_t k)
{
  uint32_t lo = 0U, hi = n - 1U;
  while (lo < hi) {
    const uint32_t mid = lo + (hi - lo) / 2U;
    float32_t p = a[mid];
    if (a[lo] > a[hi]) { float32_t t = a[lo]; a[lo] = a[hi]; a[hi] = t; }
    if (p < a[lo]) { p = a[lo]; } else if (p > a[hi]) { p = a[hi]; }
    uint32_t i = lo, j = hi;
    while (i <= j) {
      while (a[i] < p) { i++; }
      while (a[j] > p) { j--; }
      if (i <= j) {
        float32_t t = a[i]; a[i] = a[j]; a[j] = t;
        i++;
        if (j == 0U) { break; }
        j--;
      }
    }
    if (k <= j) { hi = j; } else if (k >= i) { lo = i; } else { break; }
  }
  return a[k];
}

static float32_t median_f32(float32_t *a, uint32_t n)
{
  const float32_t upper = kth_smallest(a, n, n / 2U);
  if (n & 1U) {
    return upper;
  }
  float32_t lower = a[0];
  for (uint32_t i = 1U; i < n / 2U; i++) {
    lower = a[i] > lower ? a[i] : lower;
  }
  return 0.5f * (lower + upper);
}

static void denoise(float32_t *x, uint32_t n_in, const wav_t *w)
{
  uint32_t len[MAX_LEVEL] = {0U}, off[MAX_LEVEL] = {0U}, n = n_in, used = 0U;
  for (uint32_t l = 0; l < w->level; l++) {
    off[l] = used;
    n = dwt_per(x, n, w, &g_coef[used]);
    len[l] = n;
    used += n;
  }
  const uint32_t n1 = len[0];
  memcpy(g_ext, g_coef, n1 * sizeof(float32_t));
  const float32_t med = median_f32(g_ext, n1);
  for (uint32_t i = 0; i < n1; i++) {
    g_ext[i] = fabsf(g_coef[i] - med);
  }
  const float32_t thr = median_f32(g_ext, n1) / 0.6745f * THR_K_V1;
  for (uint32_t i = 0; i < used; i++) {
    const float32_t v = g_coef[i], m = fabsf(v) - thr;
    g_coef[i] = m > 0.0f ? copysignf(m, v) : 0.0f;
  }
  for (uint32_t l = w->level; l-- > 0U;) {
    idwt_per(x, &g_coef[off[l]], len[l], w);
  }
}

static uint32_t dwt_sym_energy(const float32_t *x, uint32_t n, float32_t *a, float32_t *energy)
{
  const uint32_t F = 8U, pre = F - 1U;
  for (uint32_t k = 0; k < pre; k++) {
    g_wext[k] = x[pre - 1U - k];
    g_wext[pre + n + k] = x[n - 1U - k];
  }
  memcpy(&g_wext[pre], x, n * sizeof(float32_t));
  const uint32_t n_out = (n + F - 1U) / 2U;
  float32_t e = 0.0f;
  for (uint32_t o = 0; o < n_out; o++) {
    const float32_t *p = &g_wext[1U + 2U * o];
    float32_t d;
    arm_dot_prod_f32(db4_dlr, p, F, &a[o]);
    arm_dot_prod_f32(db4_dhr, p, F, &d);
    e += d * d;
  }
  *energy = e;
  return n_out;
}

static void features(const float32_t *x, uint32_t ch, uint32_t rows, float32_t (*out)[N_FEAT_ALL])
{
  const float32_t df = FS_HZ / (float32_t)WIN;
  for (uint32_t r = 0; r < rows; r++) {
    const float32_t *w = &x[r * HOP];
    float32_t sum = 0.0f, mn = w[0], mx = w[0], line = 0.0f;
    for (uint32_t i = 0; i < WIN; i++) {
      sum += w[i];
      mn = w[i] < mn ? w[i] : mn;
      mx = w[i] > mx ? w[i] : mx;
      if (i) {
        line += fabsf(w[i] - w[i - 1U]);
      }
    }
    const float32_t mean = sum / (float32_t)WIN;
    float32_t m2 = 0.0f, m3 = 0.0f;
    for (uint32_t i = 0; i < WIN; i++) {
      const float32_t dv = w[i] - mean, d2 = dv * dv;
      m2 += d2;
      m3 += d2 * dv;
    }
    m2 /= (float32_t)WIN;
    m3 /= (float32_t)WIN;
    float32_t sq;
    (void)arm_sqrt_f32(m2, &sq);
    const float32_t skew = m3 / (m2 * sq);

    memcpy(g_fft_in, w, WIN * sizeof(float32_t));
    arm_rfft_fast_f32(&g_fft, g_fft_in, g_fft_out, 0U);
    const float32_t p0 = g_fft_out[0] * g_fft_out[0], pn = g_fft_out[1] * g_fft_out[1];
    float32_t energy = p0 + pn, first = pn * (float32_t)(WIN / 2U) * df;
    for (uint32_t k = 1U; k < WIN / 2U; k++) {
      const float32_t pk = g_fft_out[2U * k] * g_fft_out[2U * k] + g_fft_out[2U * k + 1U] * g_fft_out[2U * k + 1U];
      energy += pk;
      first += pk * (float32_t)k * df;
    }
    const float32_t centroid = first / (energy + 1e-12f);
    float32_t second = p0 * centroid * centroid;
    {
      const float32_t fn = (float32_t)(WIN / 2U) * df - centroid;
      second += pn * fn * fn;
    }
    for (uint32_t k = 1U; k < WIN / 2U; k++) {
      const float32_t pk = g_fft_out[2U * k] * g_fft_out[2U * k] + g_fft_out[2U * k + 1U] * g_fft_out[2U * k + 1U];
      const float32_t fk = (float32_t)k * df - centroid;
      second += pk * fk * fk;
    }
    float32_t spread;
    (void)arm_sqrt_f32(second / (energy + 1e-12f), &spread);

    float32_t d1, d2;
    const uint32_t na1 = dwt_sym_energy(w, WIN, g_a1, &d1);
    (void)dwt_sym_energy(g_a1, na1, g_fft_out, &d2);

    const float32_t v[N_FEAT] = {mean, m2, mx - mn, line, skew, energy, spread, d2, d1};
    float32_t *row = &out[r][ch * N_FEAT];
    for (uint32_t f = 0; f < N_FEAT; f++) {
      row[f] = (v[f] - scaler_mean[ch * N_FEAT + f]) / scaler_std[ch * N_FEAT + f];
    }
  }
}

void features_channel_denoise(uint32_t ch, uint32_t n)
{
  denoise(g_block[ch], n, CH_WAV[ch]);
}

void features_channel_rows(uint32_t ch, uint32_t rows, float32_t (*out)[N_FEAT_ALL])
{
  features(&g_block[ch][TRIM], ch, rows, out);
}

/* scoring.py: e = mean over time of |pred - target| per channel; score = sum_ij e_i inv_cov_ij e_j. pred is [27][T], target [T][27]. */
float32_t score_window(const float32_t *pred_cf, const float32_t (*target)[N_FEAT_ALL], uint32_t t_len)
{
  float32_t e[N_FEAT_ALL];
  for (uint32_t c = 0; c < N_FEAT_ALL; c++) {
    float32_t s = 0.0f;
    const float32_t *p = &pred_cf[c * t_len];
    for (uint32_t t = 0; t < t_len; t++) {
      s += fabsf(p[t] - target[t][c]);
    }
    e[c] = s / (float32_t)t_len;
  }
  float32_t out = 0.0f;
  for (uint32_t i = 0; i < N_FEAT_ALL; i++) {
    float32_t acc;
    arm_dot_prod_f32(&inv_cov[i * N_FEAT_ALL], e, N_FEAT_ALL, &acc);
    out += e[i] * acc;
  }
  return out;
}
