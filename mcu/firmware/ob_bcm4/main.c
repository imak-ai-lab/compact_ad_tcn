/* Phase 5: one-shot Cortex-M7 program that sets option byte BCM4 = 0 (Cortex-M4 boot disabled)
 * with ST's HAL (v1.11.6): HAL_FLASH_OB_Unlock -> HAL_FLASHEx_OBProgram(OPTIONBYTE_USER, OB_USER_BCM4,
 * OB_BCM4_DISABLE) -> HAL_FLASH_OB_Launch -> lock. Only the BCM4 bit is in the mask; RDP and every
 * other option byte are left as they are.
 *
 * Why: STM32CubeProgrammer v2.23 cannot do it for this chip (`-ob BCM4=0`: "does not exist";
 * `-w32` into the option-byte registers crashes the CLI). See cubeai/ob_bcm4_write_*.log.
 *
 * Idempotent: if BCM4 is already 0 it changes nothing. Result on the LEDs and in RAM:
 *   green LD1 (PB0) = BCM4 is 0 now;  red LD3 (PB14) = HAL error or BCM4 still 1.
 *   g_ob_result (in .bss) = 0xB0C40000 | status, readable with STM32_Programmer_CLI -r32.
 */
#include "stm32h7xx_hal.h"

volatile uint32_t g_ob_result;          /* 0xB0C40001 ok-changed, 0xB0C40002 ok-already-0, 0xB0C4EEEx error */
volatile uint32_t g_optsr_before, g_optsr_after;

void SysTick_Handler(void) { HAL_IncTick(); }
void HardFault_Handler(void) { while (1) {} }

static void leds(int ok)
{
  GPIO_InitTypeDef g = {0};
  __HAL_RCC_GPIOB_CLK_ENABLE();
  g.Pin = GPIO_PIN_0 | GPIO_PIN_14;
  g.Mode = GPIO_MODE_OUTPUT_PP;
  g.Pull = GPIO_NOPULL;
  g.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &g);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, ok ? GPIO_PIN_SET : GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, ok ? GPIO_PIN_RESET : GPIO_PIN_SET);
}

int main(void)
{
  HAL_Init();                                   /* HSI 64 MHz, no clock change needed */
  g_optsr_before = FLASH->OPTSR_CUR;

  if ((g_optsr_before & FLASH_OPTSR_BCM4) == 0U)
  {
    g_ob_result = 0xB0C40002U;
  }
  else
  {
    FLASH_OBProgramInitTypeDef ob = {0};
    ob.OptionType = OPTIONBYTE_USER;
    ob.USERType = OB_USER_BCM4;
    ob.USERConfig = OB_BCM4_DISABLE;

    if (HAL_FLASH_Unlock() != HAL_OK)              g_ob_result = 0xB0C4EEE1U;
    else if (HAL_FLASH_OB_Unlock() != HAL_OK)      g_ob_result = 0xB0C4EEE2U;
    else if (HAL_FLASHEx_OBProgram(&ob) != HAL_OK) g_ob_result = 0xB0C4EEE3U;
    else if (HAL_FLASH_OB_Launch() != HAL_OK)      g_ob_result = 0xB0C4EEE4U;
    else                                           g_ob_result = 0xB0C40001U;
    HAL_FLASH_OB_Lock();
    HAL_FLASH_Lock();
  }

  g_optsr_after = FLASH->OPTSR_CUR;
  leds(((g_ob_result & 0xFFF0U) != 0xEEE0U) && ((g_optsr_after & FLASH_OPTSR_BCM4) == 0U));
  while (1) {}
}
