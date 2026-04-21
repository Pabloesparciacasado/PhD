Parametros iniciales:
Inicializo un minimo de 3 strikes para las derivadadas transversales. Minimo necesario para poder calcular derivadas.

Inicializo también un minmo de 1 en el cambio de subyacente para evitar que la derivada explote demasiado (podría revisar esto)

Para la comparativa del método 1 (bucket) genero los grids y sus edges. Adicionalmente también de moneyness labels.


EN la sección de utilidades creo las funciones:
1-> safe_divide : en caso de que el denominador sea 0 o infinito, se devuelve un NaN.

2-> assign_Bucket: los edges y bins anteriores se asignan a los datos, empleando la función de pandas pd.cut.
3-> assign_moneyness_bin: de manera similar aplciamos pd.cut() para asiganr el bucket de moneyness.

4-> filter_groups_min_strikes: agrupando por las columnas de interés según el método el calculo, cogemos aquellos grupos que tienen el mínimo de puntos necesario.

5-> compute_cross_sectional_greeks: Calcula Delta y Gamma a partir de diferencias finitas centradas sobre la dimensión strike dentro de cada grupo:
    En primer lugar agrupamos según el método que empleemos, y ordenamos de menor a mayor el strike.
    Movemos las series con el método shift() para obtener el valor previo y siguiente necesario para calcular las diferencias, entrando en juego el método safe_divide() generado antes, para luego recuperar la delta por la propiedad de homogeneidad.
    De forma similar, pero con 3 puntos: el previo, el actual y el siguiente, controlando la distancia media de izquierda a derecha, calculo la segunda derivada y recuperamos gamma.
Como quality check invalido aquellos calculos en los que tengamos una violación de monotonicidad:  ecesitamos que los strikes vecinos no sean NaNs, que la variación de strikes no sea menor o igual a cero (recordar que hemos ordenado de menor a mayor), asimismo no pouede haber un precio infinito o negativo.

6-> compute_temporal_greeks: Delta empírica temporal y una gamma empírica (creo que es la idea de Belén):
    En este caso, se analiza la variación a nivel contrato, incialmente no restrinjo vencimiento y su agrupamiento no tendría sentido.
    Por lo que agrupando por contrato en fecha (dd-mm-aaaa), calculamos la variación del precio y variación del spot con la función diff(), y obtengo adicionalmente el lag Spot. 
Con todo lo anterior, calculo la delta teniendo en cuenta un mínimo movimiento del subyacente (aún no muy claro)
Y de forma similar, pero teniendo en cuenta el lag de la diferencia calculamos gamma como la variación de la variación del Mid en el numerador.

7-> diagnostico_metodo: Calculamos métricas de diagnóstico del metodo de interés.


En el caso de que para la misma agrupación haya diversos strikes, mantendré aquel que tenga mayor liquidez, primero por OI y en segundo lugar Volume.

Como medida de liquidez general, sumo OI y volumen. 