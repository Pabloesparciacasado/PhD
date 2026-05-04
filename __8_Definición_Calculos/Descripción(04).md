# Comparativa de las griegas empíricas con BS y con variables macroeconómicas:

En los pasos anteriores hemos generado las griegas empíricas.

## Paso 1: Selección de los puntos sin NaNs 

Tanto para Delta como para Gamma.
En este primer análisis cojo la opción 1 ya que no recae sobre ninguna agregación de datos. Pero ha de notarse la reducción en puntos omitidos cuando se agrega (siendo la opción 2, agregación sobre deltas individuales la menos problemática).

## Paso 2: Selección del bucket 15-45 días. 

Para analizar seleccionamos los contratos que tienen vencimiento de 15 a 45 días, ya que nos interesa en primer lugar una expectativa sobre el movimiento del subyacente a un mes vista.

Llamando a este conjunto de datos: *opt_df_greek_filt*

## Análisis 1: Sensibilidades (como variación temporal) vs las griegas de BS.

Genero la **tabla** en la que se ve principales estadísticos descritptivos de las sensibilidades empíricas conjuntamente con las teóricas de BS.

    Lo que se observa principalmente es una cantidad desmesurada en outliers en los cálculos empíricos, muchos casos con signo contrario al teórico. 

Para facilitar el análisis consiguiente, elimino para aquellos puntos fuera del percentil 95% y el percentil complementario, manteniendo una magnitud similar a los teóricos.

    Genero una nueva tabla descriptiva y es llamativa la similitud de la media y desciación tipica de las deltas y adicionalmente la distribución en calls.
    En gammas se destaca el signo contrario al teórico

## Análisis 2: Graficos de las serie temporal.

Se generan 2X2 gráficos para mostrar la serie temporal de las deltas y gammas para la media aritmética del mes y media ponderada mensual por OI.

    Se ve como en los gráficos de las DELTAS, está completamente alineado con la DELTA teórica de BS, mientras que la gamma es consistentemente negativa (en media) es mucho más acelerado y dinámico.

En un análisis adicional **Análisis 3** muestro una tabla de correlaciones que verifican este efecto.


## Análisis 4: Construcción sensibilidad mensual:

En el análisis 2, calculé la media mensual ponderada por OI. (HECHO)

- A: Podríase calcular la griega agregada con frecuecia mensual como diferencia de la media del día 1 y final de mes. (HECHO)

- B: Asimismo también podría coger únicamente el último día del mes. (PENDIENTE)

- C: Otra forma podría ser el coger para cada fecha los días (naturales) más cercanos. En este caso, par el día 15 del primer mes, haría la media de los primeros 15 días y los siguiente 15, pero en el día 30 del mes, cogería los últimos de 15 días y los siguientes 15 del mes siguiente.
    (PENDIENTE, me ubico en el 31 y cojo mitad del mismo y mitad del siguiente mes)

- D. Quizá sería conveniente en calcular las griegas directamente sobre la frecuencia mensual. Es decir, cogiendo día 1 y 31 y hacer las diferencias sobre los precios en estas fechas. Ya que la ocpción A y se basaría en el calculo de la variación con el día anterior. (PENDIENTE)

Para cada una de las opciones, vemos correlaciones con las griegas de BS.

Y saco las autocorrelaciones, y test de ADF

## Análisis 5: Cobertura

Ya tenemos el análisis de cobertura por moneyness y días no cubiertos para los datos diarios.
Aplico el mismo análisis para la cobertura mensual.