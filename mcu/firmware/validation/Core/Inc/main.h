#ifndef MAIN_H
#define MAIN_H

#include "stm32h7xx_hal.h"

/* NUCLEO-H755ZI-Q user LEDs (ST BSP stm32h7xx_nucleo.h) */
#define LED_GREEN_PORT  GPIOB
#define LED_GREEN_PIN   GPIO_PIN_0
#define LED_RED_PORT    GPIOB
#define LED_RED_PIN     GPIO_PIN_14

/* Validation link. ST-LINK V3 VCP; the host side passes the same rate in
 * `stedgeai validate --mode target -d serial:COMx:<baud>`. */
#ifndef VALIDATION_BAUDRATE
#define VALIDATION_BAUDRATE 460800U
#endif

void Error_Handler(void);

#endif /* MAIN_H */
