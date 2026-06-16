# IA-Workshop
Workshop de IA para aprender a tratar datos. Estos métodos incluyen importación, limpieza, imputación y representación de datasets, a través de liberías de Python como ```pandas```, ```matplotlib``` y ```numpy```.

## California Housing
En las primeras semanas, se trabajará con el conocido dataset de 'California Housing' con el objetivo de aprender estas técnicas de análisis de datos y predicción. Después de tratar correctamente los datos, pasaremos a crear modelos simples de Inteligencia Artificial como regresores lineales y 'random forests' para predecir los valores seleccionados.

## Digit Classifier
Más adelante, clasificaremos imágenes de digitos escritos a mano ('MNIST handwritten digits dataset'). En nuestro caso, hemos probado 5 métodos distintos (4 de ellos con algoritmos "no inteligentes"), para clasificar correctamente los dígitos 0-9. Estos métodos se basan en la creación de matrices que representan como es el representante "ideal" o medio de una clase; esto es, haciendo la media de todos los pixeles del set de entrenamiento. Después, según el método, se calcula la diferencia entre la matriz a predecir y cada matriz de números ideales, eligiendo la de menor diferencia como la matriz buscada.

<img alt="Imágen de los números ideales" src="Digit Recognizer/numeros_ideales.png" width="400">

También hemos diseñado un pequeño script que nos sirve para probar el funcionamiento de estos modelos (en concreto, el método 3), importando las matrices de números ideales. Adjuntamos un GIF probando diferentes números dibujados a mano

<img alt="GIF mostrando como el modelo predice correctamente los números" src="Digit Recognizer/Digit Classifier.gif" width="300">
