# Carpeta 8 — Análisis de selección de la muestra (15–45 días a vencimiento)

Se analiza la muestra restringiéndola a opciones con entre 15 y 45 días naturales a vencimiento.

---

## 1. Selección de contratos por vencimiento

Se filtran los contratos con días a vencimiento en el rango **[15, 45] días naturales**.

---

## 2. Asignación de grupos de moneyness

Se asignan buckets de moneyness desde **0.1 hasta 2.0**, en incrementos de 0.1. Se construyen dos métricas de moneyness en paralelo:

- **Moneyness sobre spot (S)**
- **Moneyness sobre forward (F)**

Se aplica el mismo proceso de limpieza que en la carpeta **6**.

---

## 3. Estadísticos por grupo de moneyness

Para cada bucket de moneyness se calculan:

|  | Métrica | Descripción | Medida
|---|---|---|---|
| A | Número de contratos negociados | Calculo el número de contratos diariamente | Porcentaje de días sin contratos |
| B | Open interest medio | Suma del OI diariamente | Valor mínimo y máximo |
| C | Volumen en dólares | Suma del Volumen diariamente | Valor mínimo y máximo |
| D | Análisis bid | Media de Variable Dummy: 1 si bid >0  | Valor mínimo y máximo |

---

Con estas medidas, genero una tabla resumen para tener la información agregada de toda la serie.
Con esto se obtiene como un mejor/peor.

Adicionalmente, mantengo la descirpción de la métrica para cada uno de los puntos para la serie temporal.

## 4. Serie temporal de precios por bucket

Se examina la cobertura temporal de cada bucket respondiendo a:

- ¿Hay suficientes contratos y suficiente OI en cada día y bucket?
    - 1. **Nivel Contrato**: Para cada bucket, verifico si hay días que rompan la serie, es decir (no hay ningún contrato).
    - 2. **Combinado con BID>0**: si ademas Bid>0
    - 4. **Serie temporal** del rango de moneyness cubierto [m_min, m_max] por día (con y sin filtros del punto 2).



- ¿Existe continuidad uniforme a lo largo del tiempo en cada bucket?

        Respuesta: NO. Solo hay continuidad suficiente para el ATM de ciertos rango de vencimientos.

- Continuidad en cada bucket a lo largo del tiempo.

En cada bucket, obtengo para cada día el número días seguidos tenemos hasta la siguiente rotura, lo que he llamado rachas. Adicionalmente muestro en número minimo y macximo de rachas y el número de rachas, cortes que se realizan, cuanto menos cortes más continuidad.

---

## 5. Robustez: moneyness sobre forward

Se repiten los pasos **2, 3 y 4** usando moneyness calculado sobre el forward *F* en lugar del spot *S*, para evaluar si los resultados son sensibles a esta elección.

---

## 6. Cálculo de sensibilidades

Las sensibilidades (delta y gamma) se estiman por dos vías:

- **A — Variación temporal:** diferencia en el tiempo del precio MidPoint sobre la variación del Spot Price.

Para hacer esto, tengo tres opciones: **op1** a nivel contrato individual (lo que ya mostré en el anterior documento de FD_description) dandome la variación para cada moneyness eacto; como extensión, **op2** podría luego agrupar las sensibilidades para cada bucket de moneyness; y una tercera, **op3** vía es agrupando por bucket de moneyness, en el que calcularía una media ponderada del Mid_price (esto me daría mayor posibilidad de tener suficientes puntos aun viendo las debilidades en buckets iniciales).

- **B — Local Poylnomial Regression:** PENDIENTE

### Diagnóstico.

Para evaluar/ las sensibilidades compruebo:
 - Descriptivos desagregando por call y put para varios percentiles.
 - De los valores con el signo opuesto al teórico y outliers, en qué zonas de OI se enuentran, contratos - racha hay, en que zonas de moneyness falla más. Distinguiendo por CallPut


---

## 7. Serie temporal de sensibilidades por bucket

Se verifica la calidad de los estimadores:

- ¿Se cubren todos los buckets de moneyness?
- ¿Hay un nivel elevado de violaciones o ruido en los estimadores?