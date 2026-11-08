/* Phase 5, NUCLEO-H755ZI-Q (Cortex-M7): platform header expected by
 * ai_device_adaptor.h / aiValidation (ST Edge AI 10.2). Imports the HAL and the
 * UART used for the validation protocol: USART3 on PD8/PD9 = ST-LINK V3 virtual
 * COM port (COM1 of the ST NUCLEO-H7 BSP, stm32h7xx_nucleo.h). */
#ifndef BSP_AI_H
#define BSP_AI_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32h7xx_hal.h"
#include "app_x-cube-ai.h"

extern UART_HandleTypeDef UartHandle;

#ifdef __cplusplus
}
#endif

#endif /* BSP_AI_H */
