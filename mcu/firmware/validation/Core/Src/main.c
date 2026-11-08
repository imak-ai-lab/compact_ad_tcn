/* Phase 5 / stage 3: Cube.AI aiValidation firmware for NUCLEO-H755ZI-Q, Cortex-M7 only.
 *
 * Board and clock settings follow ST's own NUCLEO-H745ZI-Q templates (STM32CubeH7,
 * Projects/NUCLEO-H745ZI-Q/Templates/BootCM7_CM4Gated; the readme there says all
 * H745ZI-Q projects are compatible with H755ZI-Q):
 *   - power: direct SMPS supply (USE_PWR_DIRECT_SMPS_SUPPLY, applied by ExitRun0Mode()
 *     in the startup code) = factory configuration of the "-Q" board;
 *   - VOS1, SYSCLK 400 MHz from PLL1 (HSE 8 MHz bypass = ST-LINK MCO), HCLK 200 MHz,
 *     APBx 100 MHz, FLASH_LATENCY_4 as in ST's examples for this board.
 *     480 MHz needs VOS0, which the HAL allows only when Vcore comes from the LDO
 *     (stm32h7xx_hal_pwr_ex.c, HAL_PWREx_ControlVoltageScaling), i.e. not in the
 *     default direct-SMPS configuration;
 *   - MPU background region + I/D caches as in ST's template.
 * The Cortex-M4 is never released: there is no HAL_RCCEx_EnableBootCore(RCC_BOOT_C2)
 * here, and the option byte BCM4 must be 0.
 */
#include "main.h"
#include "bsp_ai.h"

UART_HandleTypeDef UartHandle;

static void MPU_Config(void);
static void SystemClock_Config(void);
static void LED_Init(void);
static void MX_USART3_UART_Init(void);

int main(void)
{
  MPU_Config();
  SCB_EnableICache();
  SCB_EnableDCache();

  HAL_Init();
  SystemClock_Config();

  LED_Init();
  MX_USART3_UART_Init();

  /* DWT cycle counter is used by aiValidation for the inference time. On the
   * Cortex-M7 the DWT has a software lock; unlock it so CYCCNT runs without a
   * debugger attached (port_dwt_init_imp() only does this for STM32F7). */
  DWT->LAR = 0xC5ACCE55;

  HAL_GPIO_WritePin(LED_GREEN_PORT, LED_GREEN_PIN, GPIO_PIN_SET);

  MX_X_CUBE_AI_Init();
  while (1)
  {
    MX_X_CUBE_AI_Process();
  }
}

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
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 400;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_1;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
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
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

static void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct = {0};

  HAL_MPU_Disable();
  /* Background region: strongly ordered / no access for undefined regions
   * (speculative-access protection), sub-regions 0,1,2,7 disabled -> default map. */
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

static void LED_Init(void)
{
  GPIO_InitTypeDef gpio = {0};
  __HAL_RCC_GPIOB_CLK_ENABLE();
  gpio.Pin = LED_GREEN_PIN | LED_RED_PIN;
  gpio.Mode = GPIO_MODE_OUTPUT_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &gpio);
  HAL_GPIO_WritePin(GPIOB, LED_GREEN_PIN | LED_RED_PIN, GPIO_PIN_RESET);
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
  if (HAL_UART_Init(&UartHandle) != HAL_OK)
  {
    Error_Handler();
  }
  /* FIFO on: with blocking HAL_UART_Receive and no flow control, the 16-byte RX FIFO keeps the
   * link from overrunning at high baud rates (first trial without FIFO stopped after 102 samples
   * with a host read timeout, cubeai/validate_target/validate_target_tr047_fp32.log). */
  if (HAL_UARTEx_SetTxFifoThreshold(&UartHandle, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK ||
      HAL_UARTEx_SetRxFifoThreshold(&UartHandle, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK ||
      HAL_UARTEx_EnableFifoMode(&UartHandle) != HAL_OK)
  {
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
