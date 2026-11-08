/* Phase 5: Cortex-M7 exception handlers. Faults light the red LED and stop. */
#include "main.h"

static void fault(void)
{
  __disable_irq();
  HAL_GPIO_WritePin(LED_RED_PORT, LED_RED_PIN, GPIO_PIN_SET);
  while (1) {}
}

void NMI_Handler(void) { fault(); }
void HardFault_Handler(void) { fault(); }
void MemManage_Handler(void) { fault(); }
void BusFault_Handler(void) { fault(); }
void UsageFault_Handler(void) { fault(); }
void SVC_Handler(void) {}
void DebugMon_Handler(void) {}
void PendSV_Handler(void) {}

void SysTick_Handler(void)
{
  HAL_IncTick();
}
