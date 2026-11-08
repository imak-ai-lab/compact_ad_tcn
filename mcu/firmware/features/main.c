/* Phase 5 / stage 5: feature extraction on the Cortex-M7 (NUCLEO-H755ZI-Q), CMSIS-DSP f32.
 *
 * Same board, clock, MPU, cache settings and UART protocol as firmware/bench/main.c (stage 4); Cortex-M4 never
 * released. Port of industrial_ad/datasets/PU/features.py, applied to ONE BLOCK instead of a whole file:
 *
 *   input   3 channels x BLOCK_N = 3248 samples (float32): 1968 under the T = 120 rows + 640 per trimmed edge
 *   denoise wavedec 'per' (sym8 L6 current, db4 L4 vibration) as FIR + decimation, soft threshold on all details,
 *           waverec 'per' as upsampling + FIR. sigma = MAD(finest details) / 0.6745, thr = sigma * THR_K
 *           BENCH_VARIANT 0: THR_K = sqrt(2 ln BLOCK_N)  (the repo function applied to the block)
 *           BENCH_VARIANT 1: THR_K = sqrt(2 ln 256000)   (whole-file factor, closest to the training features,
 *                                                         stage5_threshold_variants.py)
 *   features trim 640 per edge; window 64, hop 16, 120 rows: mean, variance, peak-to-peak, line integral, skewness,
 *           spectral energy, spectral spread (rFFT, fs = 40000 as in the code and the checkpoint configs),
 *           db4 'symmetric' detail energies D2, D1; preset scaler -> float32 [120][27]
 *
 * FIR and FFT are CMSIS-DSP: arm_dot_prod_f32 over contiguous extended buffers, arm_rfft_fast_f32. The index formulas
 * were checked against pywt by check_dwt_formulas.py.
 *
 * Protocol: start-up as firmware/bench (host 'G' or 30 s). Check pass: each of BENCH_K stored blocks once, the output
 * bytes + CRC32 to UART for the host comparison. 2 s WFI. Timed loop: BENCH_N blocks (block i % K), the block is
 * copied from flash to RAM first, outside the DWT measurement (a device would have it in RAM from DMA); no UART inside
 * the loop. Then per-block cycles, per-stage cycle sums and stack use to UART; green LED; WFI.
 */
#include <math.h>
#include <string.h>
#include "main.h"
#include "arm_math.h"
#include "arm_const_structs.h"
#include "features_data.h"

#ifndef BENCH_N
#define BENCH_N 4000U
#endif
#ifndef BENCH_VARIANT
#define BENCH_VARIANT 1
#endif
#ifndef BENCH_TAG
#define BENCH_TAG unknown
#endif
#ifndef BENCH_WAIT_S
#define BENCH_WAIT_S 30U
#endif
#define STR_(x) #x
#define STR(x) STR_(x)

#define PULSE_PORT GPIOD          /* PD0 = Zio D67, CN9 pin 25, as firmware/bench */
#define PULSE_PIN  GPIO_PIN_0

#define THR_K ((BENCH_VARIANT) ? THR_K_V1 : THR_K_V0)
#define N_CH 3U
#define N_FEAT 9U
#define WIN 64U
#define HOP 16U
#define ROWS 120U
#define TRIM 640U
#define FS_HZ 40000.0f
#define MAX_LEVEL 6U
_Static_assert(BLOCK_N == WIN + (ROWS - 1U) * HOP + 2U * TRIM, "block length");

UART_HandleTypeDef UartHandle;

/* ------------------------------------------------------------- algorithm RAM (see build_features.py) ---- */
__attribute__((aligned(32))) float32_t g_block[N_CH][BLOCK_N + 2U];      /* raw block, then approximations, then denoised */
__attribute__((aligned(32))) float32_t g_coef[BLOCK_N];                 /* detail coefficients of all levels */
__attribute__((aligned(32))) float32_t g_ext[BLOCK_N + 64U];            /* extended FIR input; median scratch */
__attribute__((aligned(32))) float32_t g_rows[ROWS][N_CH * N_FEAT];     /* output: scaled features */
float32_t g_fft_in[WIN], g_fft_out[WIN], g_wext[WIN + 32U], g_a1[WIN];
static arm_rfft_fast_instance_f32 g_fft;

/* benchmark only, not part of the algorithm */
static uint32_t cyc[BENCH_N] __attribute__((section(".ai_pool")));
static uint64_t stage_cyc[2U * N_CH];

typedef struct {
  uint32_t F, level;
  const float32_t *dlr, *dhr;             /* analysis filters, time-reversed */
  const float32_t *rle, *rlo, *rhe, *rho; /* synthesis filters, even / odd taps, time-reversed */
} wav_t;
static const wav_t W_SYM8 = {16U, 6U, sym8_dlr, sym8_dhr, sym8_rle, sym8_rlo, sym8_rhe, sym8_rho};
static const wav_t W_DB4 = {8U, 4U, db4_dlr, db4_dhr, db4_rle, db4_rlo, db4_rhe, db4_rho};
static const wav_t *const CH_WAV[N_CH] = {&W_SYM8, &W_SYM8, &W_DB4};

/* ---------------------------------------------------------------------------------- wavelet ---- */
/* One level of wavedec, mode 'per': a[o] = sum_j h[j] xt[(F/2 + 2o - j) mod n2]. Approximation written over x. */
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

/* One level of waverec, mode 'per', M = len(d); a may hold one extra element, which is ignored as in pywt. y over a.
 * y[(2t-1) mod 2M] = sum_{j<F/2} f[2j] c[(i-j) mod M], y[2t] = sum_{j<F/2} f[2j+1] c[(i-j) mod M], i = F/4-1+t */
static uint32_t idwt_per(float32_t *a, const float32_t *d, uint32_t M, const wav_t *w)
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
  return 2U * M;
}

/* k-th smallest, Hoare quickselect with median of three; a is reordered around k. */
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

/* np.median: for even n the mean of the two central order statistics. a is reordered. */
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

/* features.py: denoise_signal_dwt() on one block, in place. */
static void denoise(float32_t *x, const wav_t *w)
{
  uint32_t len[MAX_LEVEL] = {0U}, off[MAX_LEVEL] = {0U}, n = BLOCK_N, used = 0U;
  for (uint32_t l = 0; l < w->level; l++) {
    off[l] = used;
    n = dwt_per(x, n, w, &g_coef[used]);
    len[l] = n;
    used += n;
  }
  /* MAD of the finest details (first level) */
  const uint32_t n1 = len[0];
  memcpy(g_ext, g_coef, n1 * sizeof(float32_t));
  const float32_t med = median_f32(g_ext, n1);
  for (uint32_t i = 0; i < n1; i++) {
    g_ext[i] = fabsf(g_coef[i] - med);
  }
  const float32_t thr = median_f32(g_ext, n1) / 0.6745f * THR_K;
  for (uint32_t i = 0; i < used; i++) {
    const float32_t v = g_coef[i], m = fabsf(v) - thr;
    g_coef[i] = m > 0.0f ? copysignf(m, v) : 0.0f;
  }
  for (uint32_t l = w->level; l-- > 0U;) {
    (void)idwt_per(x, &g_coef[off[l]], len[l], w);
  }
}

/* One level of wavedec, mode 'symmetric', F = 8 (db4): returns the output length; approximation to a, detail energy. */
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

/* features.py: extract_features_vectorized() on 120 windows of one channel, then the preset scaler. */
static void features(const float32_t *x, uint32_t ch)
{
  const float32_t df = FS_HZ / (float32_t)WIN;
  for (uint32_t r = 0; r < ROWS; r++) {
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
    /* packed: [X0, X(N/2), Re1, Im1, ...]. Energy and moments over bins 0..N/2, as np.fft.rfft. */
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
    (void)dwt_sym_energy(g_a1, na1, g_fft_out, &d2);     /* scratch: approximation of level 2 is not used */

    const float32_t v[N_FEAT] = {mean, m2, mx - mn, line, skew, energy, spread, d2, d1};
    float32_t *row = &g_rows[r][ch * N_FEAT];
    for (uint32_t f = 0; f < N_FEAT; f++) {
      row[f] = (v[f] - scaler_mean[ch * N_FEAT + f]) / scaler_std[ch * N_FEAT + f];
    }
  }
}

static void process_block(void)
{
  for (uint32_t ch = 0; ch < N_CH; ch++) {
    const uint32_t t0 = DWT->CYCCNT;
    denoise(g_block[ch], CH_WAV[ch]);
    const uint32_t t1 = DWT->CYCCNT;
    features(&g_block[ch][TRIM], ch);
    const uint32_t t2 = DWT->CYCCNT;
    stage_cyc[2U * ch] += t1 - t0;
    stage_cyc[2U * ch + 1U] += t2 - t1;
  }
}

static void load_block(uint32_t k)
{
  for (uint32_t ch = 0; ch < N_CH; ch++) {
    memcpy(g_block[ch], &bench_blocks_u32[(k * N_CH + ch) * BLOCK_N], BLOCK_N * sizeof(float32_t));
  }
}

/* ---------------------------------------------------------------- UART text, as firmware/bench ---- */
static void MPU_Config(void);
static void SystemClock_Config(void);
static void GPIO_Init(void);
static void MX_USART3_UART_Init(void);

static void tx(const void *p, uint32_t n)
{
  const uint8_t *b = (const uint8_t *)p;
  while (n) {
    uint16_t c = n > 0x8000U ? 0x8000U : (uint16_t)n;
    if (HAL_UART_Transmit(&UartHandle, (uint8_t *)b, c, HAL_MAX_DELAY) != HAL_OK) {
      Error_Handler();
    }
    b += c;
    n -= c;
  }
}
static void tx_s(const char *s) { tx(s, (uint32_t)strlen(s)); }
static void tx_u64(uint64_t v)
{
  char buf[21];
  int i = 20;
  buf[i] = 0;
  do {
    buf[--i] = (char)('0' + (v % 10U));
    v /= 10U;
  } while (v);
  tx_s(&buf[i]);
}
static void tx_kv(const char *k, uint64_t v)
{
  tx_s(" ");
  tx_s(k);
  tx_s("=");
  tx_u64(v);
}
static void tx_kx(const char *k, uint32_t v)
{
  static const char hex[] = "0123456789abcdef";
  char buf[9];
  for (int i = 7; i >= 0; i--) {
    buf[i] = hex[v & 0xFU];
    v >>= 4;
  }
  buf[8] = 0;
  tx_s(" ");
  tx_s(k);
  tx_s("=");
  tx_s(buf);
}
static void tx_flush(void)
{
  while (__HAL_UART_GET_FLAG(&UartHandle, UART_FLAG_TC) == RESET) {}
}
static uint32_t crc32(const void *p, uint32_t n)
{
  const uint8_t *b = (const uint8_t *)p;
  uint32_t c = 0xFFFFFFFFU;
  while (n--) {
    c ^= *b++;
    for (int k = 0; k < 8; k++) {
      c = (c >> 1) ^ (0xEDB88320U & (0U - (c & 1U)));
    }
  }
  return c ^ 0xFFFFFFFFU;
}
static void wfi_ms(uint32_t ms)
{
  uint32_t t0 = HAL_GetTick();
  while ((HAL_GetTick() - t0) < ms) {
    __WFI();
  }
}

/* Stack high-water mark: paint below the current SP once at start, scan after the run. */
extern uint32_t end;
extern uint32_t _estack;
static void stack_paint(void)
{
  uint32_t *p = &end;
  uint32_t *sp = (uint32_t *)__get_MSP();
  while (p < sp - 16) {
    *p++ = 0xA5A5A5A5U;
  }
}
static uint32_t stack_used(void)
{
  uint32_t *p = &end;
  while (p < &_estack && *p == 0xA5A5A5A5U) {
    p++;
  }
  return (uint32_t)((uint8_t *)&_estack - (uint8_t *)p);
}

/* ---------------------------------------------------------------------------------- main ---- */
int main(void)
{
  stack_paint();
  MPU_Config();
  SCB_EnableICache();
  SCB_EnableDCache();

  HAL_Init();
  SystemClock_Config();
  GPIO_Init();
  MX_USART3_UART_Init();

  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->LAR = 0xC5ACCE55;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  /* only the tables of this FFT length are linked (arm_rfft_fast_init_f32 pulls in all lengths) */
  g_fft.Sint = arm_cfft_sR_f32_len32;
  g_fft.fftLenRFFT = (uint16_t)WIN;
  g_fft.pTwiddleRFFT = twiddleCoef_rfft_64;

  const uint32_t wcrc = crc32(bench_blocks_u32, sizeof(bench_blocks_u32));
  uint32_t waited = 0;
  for (;;) {
    tx_s("\r\n#BENCH v1 tag=" STR(BENCH_TAG) " mode=run clk=HSE_BYPASS");
    tx_kv("n", BENCH_N);
    tx_kv("k", BENCH_K);
    tx_kv("sysclk", HAL_RCC_GetSysClockFreq());
    tx_kv("hclk", HAL_RCC_GetHCLKFreq());
    tx_kv("in_bytes", N_CH * BLOCK_N * sizeof(float32_t));
    tx_kv("out_bytes", sizeof(g_rows));
    tx_kv("act_bytes", 0U);
    tx_kx("windows_crc", wcrc);
    tx_s(" built=" __DATE__ "_" __TIME__ "\r\n");
    tx_s("#INIT");
    tx_kv("err_type", 0U);
    tx_kv("err_code", 0U);
    tx_kv("waited_s", waited);
    tx_s("\r\n");
    tx_flush();
    if (waited >= BENCH_WAIT_S) {
      break;
    }
    uint8_t c = 0;
    __HAL_UART_CLEAR_FLAG(&UartHandle, UART_CLEAR_OREF | UART_CLEAR_NEF | UART_CLEAR_PEF | UART_CLEAR_FEF);
    if (HAL_UART_Receive(&UartHandle, &c, 1, 1000) == HAL_OK && c == 'G') {
      break;
    }
    waited++;
  }
  tx_s("#GO");
  tx_kv("host", waited < BENCH_WAIT_S);
  tx_kv("waited_s", waited);
  tx_s("\r\n");

  /* 1. check pass */
  for (uint32_t k = 0; k < BENCH_K; k++) {
    load_block(k);
    const uint32_t t0 = DWT->CYCCNT;
    process_block();
    const uint32_t dt = DWT->CYCCNT - t0;
    tx_s("#OUT");
    tx_kv("k", k);
    tx_kv("cyc", dt);
    tx_kv("r", 1U);
    tx_kx("crc_in", crc32(&bench_blocks_u32[k * N_CH * BLOCK_N], N_CH * BLOCK_N * sizeof(float32_t)));
    tx_kx("crc_out", crc32(g_rows, sizeof(g_rows)));
    tx_kv("n", sizeof(g_rows));
    tx_s("\n");
    tx(g_rows, sizeof(g_rows));
    tx_s("\n");
  }
  memset(stage_cyc, 0, sizeof(stage_cyc));

  /* 2. idle step before the timed loop */
  tx_s("#PRE_IDLE ms=2000\r\n");
  tx_flush();
  wfi_ms(2000);

  /* 3. timed loop, no UART inside */
  tx_s("#RUN_START\r\n");
  tx_flush();
  const uint32_t ms0 = HAL_GetTick();
  for (uint32_t i = 0; i < BENCH_N; i++) {
    load_block(i % BENCH_K);
    PULSE_PORT->BSRR = PULSE_PIN;
    const uint32_t t0 = DWT->CYCCNT;
    process_block();
    const uint32_t t1 = DWT->CYCCNT;
    PULSE_PORT->BSRR = (uint32_t)PULSE_PIN << 16U;
    cyc[i] = t1 - t0;
  }
  const uint32_t loop_ms = HAL_GetTick() - ms0;

  /* 4. results */
  uint64_t sum = 0;
  uint32_t mn = 0xFFFFFFFFU, mx = 0;
  for (uint32_t i = 0; i < BENCH_N; i++) {
    sum += cyc[i];
    mn = cyc[i] < mn ? cyc[i] : mn;
    mx = cyc[i] > mx ? cyc[i] : mx;
  }
  tx_s("#RUN_END");
  tx_kv("loop_ms", loop_ms);
  tx_kv("errs", 0U);
  tx_s("\r\n");
  for (uint32_t i = 0; i < BENCH_N; i++) {
    tx_s("#CYC");
    tx_kv("i", i);
    tx_kv("k", i % BENCH_K);
    tx_kv("cyc", cyc[i]);
    tx_s("\r\n");
  }
  tx_s("#DONE");
  tx_kv("sum", sum);
  tx_kv("min", mn);
  tx_kv("max", mx);
  tx_kv("errs", 0U);
  tx_kv("denoise_ch0", stage_cyc[0]);
  tx_kv("features_ch0", stage_cyc[1]);
  tx_kv("denoise_ch1", stage_cyc[2]);
  tx_kv("features_ch1", stage_cyc[3]);
  tx_kv("denoise_ch2", stage_cyc[4]);
  tx_kv("features_ch2", stage_cyc[5]);
  tx_kv("stack_used", stack_used());
  tx_s("\r\n");
  tx_flush();
  HAL_GPIO_WritePin(LED_GREEN_PORT, LED_GREEN_PIN, GPIO_PIN_SET);
  for (;;) {
    __WFI();
  }
}

/* ---------------------------------------------- board set-up, same as firmware/bench/main.c ---- */
static void SystemClock_Config(void)
{
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};

  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  while (!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.HSIState = RCC_HSI_OFF;
  RCC_OscInitStruct.CSIState = RCC_CSI_OFF;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLN = 400;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_1;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
    Error_Handler();
  }

  RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_D1PCLK1 |
                                 RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2 | RCC_CLOCKTYPE_D3PCLK1);
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK) {
    Error_Handler();
  }
}

static void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct = {0};

  HAL_MPU_Disable();
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.BaseAddress = 0x00;
  MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
  MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER0;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.SubRegionDisable = 0x87;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  HAL_MPU_ConfigRegion(&MPU_InitStruct);
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}

static void GPIO_Init(void)
{
  GPIO_InitTypeDef gpio = {0};
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  gpio.Mode = GPIO_MODE_OUTPUT_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_LOW;
  gpio.Pin = LED_GREEN_PIN | LED_RED_PIN;
  HAL_GPIO_Init(GPIOB, &gpio);
  HAL_GPIO_WritePin(GPIOB, LED_GREEN_PIN | LED_RED_PIN, GPIO_PIN_RESET);
  gpio.Pin = PULSE_PIN;
  HAL_GPIO_Init(PULSE_PORT, &gpio);
  HAL_GPIO_WritePin(PULSE_PORT, PULSE_PIN, GPIO_PIN_RESET);
}

static void MX_USART3_UART_Init(void)
{
  UartHandle.Instance = USART3;
  UartHandle.Init.BaudRate = VALIDATION_BAUDRATE;
  UartHandle.Init.WordLength = UART_WORDLENGTH_8B;
  UartHandle.Init.StopBits = UART_STOPBITS_1;
  UartHandle.Init.Parity = UART_PARITY_NONE;
  UartHandle.Init.Mode = UART_MODE_TX_RX;
  UartHandle.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  UartHandle.Init.OverSampling = UART_OVERSAMPLING_16;
  UartHandle.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  UartHandle.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  UartHandle.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&UartHandle) != HAL_OK) {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&UartHandle, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK ||
      HAL_UARTEx_SetRxFifoThreshold(&UartHandle, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK ||
      HAL_UARTEx_EnableFifoMode(&UartHandle) != HAL_OK) {
    Error_Handler();
  }
}

void Error_Handler(void)
{
  __disable_irq();
  HAL_GPIO_WritePin(LED_RED_PORT, LED_RED_PIN, GPIO_PIN_SET);
  while (1) {}
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
  (void)file;
  (void)line;
  Error_Handler();
}
#endif
