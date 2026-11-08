/* Phase 5 / stage 4e: end-to-end firmware, one decision from raw samples to the anomaly score, NUCLEO-H755ZI-Q, Cortex-M7.
 *
 * Board, clock, MPU, caches, UART protocol and the generated network as firmware/bench (stage 4), signal processing as firmware/features
 * (stage 5, threshold V1). Cortex-M4 never released (BCM4 = 0).
 *
 * One decision, all inside the DWT measurement:
 *   features of the input block (3 x 3248 samples -> [120][27]) -> transpose to the network input [27][120] -> ai_network_run()
 *   -> for a forecasting model: features of the target block (3 x 5168 samples -> [240][27]), since the forecast is compared with
 *      the 240 rows that follow -> mean absolute error per channel and Mahalanobis score.
 * Outside the measurement: the copy of the stored raw samples into the raw block buffer (on a device they arrive there by DMA).
 *
 * Protocol: as firmware/bench. Check pass: each of the K stored segments once; UART gets the input rows, target rows if any, the network
 * output and the score. 2 s WFI. Timed loop of BENCH_N decisions with no UART. Then per-decision cycles, per-stage cycle sums, stack and
 * heap use; green LED; WFI.
 */
#include <string.h>
#include "main.h"
#include "network.h"
#include "network_data.h"
#include "features_core.h"

#ifndef BENCH_N
#define BENCH_N 500U
#endif
#ifndef BENCH_TAG
#define BENCH_TAG unknown
#endif
#ifndef BENCH_WAIT_S
#define BENCH_WAIT_S 30U
#endif
#define STR_(x) #x
#define STR(x) STR_(x)

#define PULSE_PORT GPIOD
#define PULSE_PIN  GPIO_PIN_0

UART_HandleTypeDef UartHandle;

#if AI_NETWORK_DATA_ACTIVATIONS_COUNT != 1
#error "one activation segment expected"
#endif
_Static_assert(AI_NETWORK_IN_1_SIZE_BYTES == ROWS_IN * N_FEAT_ALL * sizeof(float32_t), "network input size");
_Static_assert(AI_NETWORK_OUT_1_SIZE_BYTES == (BENCH_FORECAST ? ROWS_TGT : ROWS_IN) * N_FEAT_ALL * sizeof(float32_t), "network output size");
AI_ALIGNED(32) static uint8_t pool0[AI_NETWORK_DATA_ACTIVATION_1_SIZE] __attribute__((section(".ai_pool")));
static ai_handle net = AI_HANDLE_NULL;
static uint32_t cyc[BENCH_N] __attribute__((section(".ai_pool")));     /* benchmark only */

enum { ST_DEN_IN, ST_FEA_IN, ST_TRANS, ST_INFER, ST_DEN_TGT, ST_FEA_TGT, ST_SCORE, ST_N };
static uint64_t stage_cyc[ST_N];
static const char *const ST_NAME[ST_N] = {"den_in", "fea_in", "trans", "infer", "den_tgt", "fea_tgt", "score"};

static ai_buffer *g_in, *g_out;
static float32_t *g_in_p;
static const float32_t *g_out_p;
static float32_t g_score;

static void MPU_Config(void);
static void SystemClock_Config(void);
static void GPIO_Init(void);
static void MX_USART3_UART_Init(void);

static void load_raw(uint32_t k, uint32_t start, uint32_t n)
{
  for (uint32_t ch = 0; ch < N_CH; ch++) {
    memcpy(g_block[ch], &bench_blocks_u32[(k * N_CH + ch) * SEG_N + start], n * sizeof(float32_t));
  }
}

static inline void lap(uint32_t *t, uint32_t stage, uint32_t *total)
{
  const uint32_t now = DWT->CYCCNT, d = now - *t;
  stage_cyc[stage] += d;
  *total += d;
  *t = now;
}

/* One decision on stored segment k; returns the cycles inside the measurement. */
static uint32_t decision(uint32_t k)
{
  uint32_t total = 0U, t;
  load_raw(k, 0U, BLOCK_IN_N);
  PULSE_PORT->BSRR = PULSE_PIN;
  t = DWT->CYCCNT;
  for (uint32_t ch = 0; ch < N_CH; ch++) {
    features_channel_denoise(ch, BLOCK_IN_N);
    lap(&t, ST_DEN_IN, &total);
    features_channel_rows(ch, ROWS_IN, g_rows_in);
    lap(&t, ST_FEA_IN, &total);
  }
  for (uint32_t c = 0; c < N_FEAT_ALL; c++) {
    for (uint32_t r = 0; r < ROWS_IN; r++) {
      g_in_p[c * ROWS_IN + r] = g_rows_in[r][c];
    }
  }
  lap(&t, ST_TRANS, &total);
  if (ai_network_run(net, &g_in[0], &g_out[0]) != 1) {
    Error_Handler();
  }
  lap(&t, ST_INFER, &total);
#if BENCH_FORECAST
  PULSE_PORT->BSRR = (uint32_t)PULSE_PIN << 16U;
  load_raw(k, TGT_OFFSET, BLOCK_TGT_N);
  PULSE_PORT->BSRR = PULSE_PIN;
  t = DWT->CYCCNT;
  for (uint32_t ch = 0; ch < N_CH; ch++) {
    features_channel_denoise(ch, BLOCK_TGT_N);
    lap(&t, ST_DEN_TGT, &total);
    features_channel_rows(ch, ROWS_TGT, g_rows_tgt);
    lap(&t, ST_FEA_TGT, &total);
  }
  g_score = score_window(g_out_p, (const float32_t (*)[N_FEAT_ALL])g_rows_tgt, ROWS_TGT);
#else
  g_score = score_window(g_out_p, (const float32_t (*)[N_FEAT_ALL])g_rows_in, ROWS_IN);
#endif
  lap(&t, ST_SCORE, &total);
  PULSE_PORT->BSRR = (uint32_t)PULSE_PIN << 16U;
  return total;
}

/* ---------------------------------------------------------------- UART text, as firmware/bench ---- */
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
static uint32_t crc32_add(uint32_t c, const void *p, uint32_t n)
{
  const uint8_t *b = (const uint8_t *)p;
  c ^= 0xFFFFFFFFU;
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

/* Stack and heap high-water marks: everything between the end of .bss and the stack pointer is painted once at start. */
extern uint32_t end;
extern uint32_t _estack;
extern uint32_t _Min_Heap_Size;
static void ram_paint(void)
{
  uint32_t *p = &end;
  uint32_t *sp = (uint32_t *)__get_MSP();
  while (p < sp - 16) {
    *p++ = 0xA5A5A5A5U;
  }
}
static uint32_t stack_used(void)
{
  uint32_t *p = (uint32_t *)((uint8_t *)&end + (uint32_t)&_Min_Heap_Size);
  while (p < &_estack && *p == 0xA5A5A5A5U) {
    p++;
  }
  return (uint32_t)((uint8_t *)&_estack - (uint8_t *)p);
}
static uint32_t heap_used(void)
{
  uint32_t *top = (uint32_t *)((uint8_t *)&end + (uint32_t)&_Min_Heap_Size);
  while (top > &end && top[-1] == 0xA5A5A5A5U) {
    top--;
  }
  return (uint32_t)((uint8_t *)top - (uint8_t *)&end);
}

/* ---------------------------------------------------------------------------------- main ---- */
int main(void)
{
  ram_paint();
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

  features_init();
  const uint32_t wcrc = crc32_add(0U, bench_blocks_u32, sizeof(bench_blocks_u32));
  const ai_handle acts[] = { pool0 };
  ai_error err = ai_network_create_and_init(&net, acts, NULL);

  uint32_t waited = 0;
  for (;;) {
    tx_s("\r\n#BENCH v1 tag=" STR(BENCH_TAG) " mode=run clk=HSE_BYPASS");
    tx_kv("n", BENCH_N);
    tx_kv("k", BENCH_K);
    tx_kv("sysclk", HAL_RCC_GetSysClockFreq());
    tx_kv("hclk", HAL_RCC_GetHCLKFreq());
    tx_kv("in_bytes", AI_NETWORK_IN_1_SIZE_BYTES);
    tx_kv("out_bytes", AI_NETWORK_OUT_1_SIZE_BYTES);
    tx_kv("act_bytes", AI_NETWORK_DATA_ACTIVATION_1_SIZE);
    tx_kx("windows_crc", wcrc);
    tx_s(" built=" __DATE__ "_" __TIME__ "\r\n");
    tx_s("#INIT");
    tx_kv("err_type", err.type);
    tx_kv("err_code", err.code);
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
  if (err.type != AI_ERROR_NONE) {
    tx_flush();
    Error_Handler();
  }
  g_in = ai_network_inputs_get(net, NULL);
  g_out = ai_network_outputs_get(net, NULL);
  if (g_in == NULL || g_out == NULL || g_in[0].data == NULL || g_out[0].data == NULL) {
    Error_Handler();
  }
  g_in_p = (float32_t *)g_in[0].data;
  g_out_p = (const float32_t *)g_out[0].data;

  /* 1. check pass */
  for (uint32_t k = 0; k < BENCH_K; k++) {
    const uint32_t dt = decision(k);
    uint32_t c = 0U, n = sizeof(g_rows_in) + AI_NETWORK_OUT_1_SIZE_BYTES + sizeof(g_score);
    c = crc32_add(c, g_rows_in, sizeof(g_rows_in));
#if BENCH_FORECAST
    c = crc32_add(c, g_rows_tgt, sizeof(g_rows_tgt));
    n += sizeof(g_rows_tgt);
#endif
    c = crc32_add(c, g_out_p, AI_NETWORK_OUT_1_SIZE_BYTES);
    c = crc32_add(c, &g_score, sizeof(g_score));
    tx_s("#OUT");
    tx_kv("k", k);
    tx_kv("cyc", dt);
    tx_kv("r", 1U);
    tx_kx("crc_in", crc32_add(0U, &bench_blocks_u32[k * N_CH * SEG_N], N_CH * SEG_N * sizeof(float32_t)));
    tx_kx("crc_out", c);
    tx_kv("n", n);
    tx_s("\n");
    tx(g_rows_in, sizeof(g_rows_in));
#if BENCH_FORECAST
    tx(g_rows_tgt, sizeof(g_rows_tgt));
#endif
    tx(g_out_p, AI_NETWORK_OUT_1_SIZE_BYTES);
    tx(&g_score, sizeof(g_score));
    tx_s("\n");
  }
  memset(stage_cyc, 0, sizeof(stage_cyc));

  /* 2. idle step */
  tx_s("#PRE_IDLE ms=2000\r\n");
  tx_flush();
  wfi_ms(2000);

  /* 3. timed loop, no UART inside */
  tx_s("#RUN_START\r\n");
  tx_flush();
  const uint32_t ms0 = HAL_GetTick();
  for (uint32_t i = 0; i < BENCH_N; i++) {
    cyc[i] = decision(i % BENCH_K);
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
  for (uint32_t s = 0; s < ST_N; s++) {
    tx_kv(ST_NAME[s], stage_cyc[s]);
  }
  tx_kv("stack_used", stack_used());
  tx_kv("heap_used", heap_used());
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
