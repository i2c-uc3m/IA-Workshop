import pyxel
import pandas as pd
import numpy as np

####################   TO DO   ####################
# Mostrar y calcular las matrices correctamente (x <--> y)
# Refactor del código (clases, funciones, variables globales)
# Cambiar nombre (y funcionalidad?) de AREA_DIBUJO
# Cambiar nombre clase Numero()
# Elegir otros colores


scl = 4
WIDTH, HEIGHT = 500 / scl, 510 / scl

AREA_DIBUJO = {"scl": 3, "w": 28, "h": 28}

# Posición en la que se dibuja el cuadrado
offset_x = (WIDTH - AREA_DIBUJO["w"] * AREA_DIBUJO["scl"]) / 2
offset_y = 30

vectores_ideales = None

# Números a predecir
numeros = []

numero = None
prediccion = -1

class Numero:
    def __init__(self, label, matriz):
        self.label = label
        self.matriz = matriz

    def draw(self, max=0.8):
        # Dibujar la matriz
        for i in range(len(self.matriz)):
            for j in range(len(self.matriz[i])):
                pixel = self.matriz[i][j]
                if pixel >= max:
                    color = pyxel.COLOR_WHITE
                elif pixel > 0.8:
                    color = pyxel.COLOR_GRAY
                elif pixel > 0.5:
                    color = pyxel.COLOR_LIGHT_BLUE
                elif pixel > 0.2:
                    color = pyxel.COLOR_DARK_BLUE
                else:
                    color = pyxel.COLOR_BLACK
                
                x = offset_x + j * AREA_DIBUJO["scl"]
                y = offset_y + i * AREA_DIBUJO["scl"]
                pyxel.rect(x, y, AREA_DIBUJO["scl"], AREA_DIBUJO["scl"], color)

    def pintar_pixeles(self, x, y, gris=0.6, max=0.8, pixeles_adjacentes=True):
        # Comprobamos que el ratón está dentro del área de dibujo
        # Primero, trasladamos las coordenadas al origen del cuadrado
        # siendo las coordendas (offset_x, offset_y) trasladadas a (0, 0)
        x = x - offset_x
        y = y - offset_y
        # y escalamos
        x = int(x / AREA_DIBUJO["scl"])
        y = int(y / AREA_DIBUJO["scl"])
        # Como dibujamos la matriz al revés, hacemos un cambio x <--> y
        temp = x
        x, y = y, temp

        if x > 0 and x < AREA_DIBUJO["w"] and y > 0 and y < AREA_DIBUJO["h"]:
            self.matriz[x][y] = max  # 1

            # Sumamos la cantidad 'gris' a los pixeles no-blancos
            if pixeles_adjacentes:
                if self.matriz[x - 1][y] != 1:
                    # Suma 'gris', excepto si el color del pixel ya está en el máximo (1)
                    self.matriz[x - 1][y] = min(self.matriz[x - 1][y] + gris, max)

                if self.matriz[x][y - 1] != 1:
                    self.matriz[x][y - 1] = min(self.matriz[x][y - 1] + gris, max)

                index = int(min(x + 1, AREA_DIBUJO["w"] - 1))
                if self.matriz[index][y] != 1:
                    self.matriz[index][y] = min(self.matriz[index][y] + gris, max)

                index = int(min(y + 1, AREA_DIBUJO["h"] - 1))
                if self.matriz[x][index] != 1:
                    self.matriz[x][index] = min(self.matriz[x][index] + gris, max)


def main():
    # Leemos el archivo con el modelo (en este caso, vectores ideales)
    global vectores_ideales
    global numero
    global numeros

    df = pd.read_csv("vectores_ideales.csv")
    vectores_ideales = []
    for vector in df.values:
        # 'vector' es una columna del dataset de tipo: "['Vector0' 0.0 0.0 ... 0.0 0.0 0.0]"
        # Queremos añadir los valores de esa columna, sin el label
        vectores_ideales.append(vector[1:])

    # nums = pd.read_csv("numeros.csv")
    # for num in nums.values:
    #     matriz = np.reshape(num[1:], (28, 28))
    #     # Dividimos cada valor entre 255
    #     matriz = matriz / 255

    #     numeros.append({"label": int(num[0]), "matriz": matriz})

    # numero = Numero(numeros[1]["label"], numeros[1]["matriz"])
    numero = Numero(-1, np.zeros((28, 28)))

    pyxel.init(int(WIDTH), int(HEIGHT), title="Dibuja tu número!", display_scale=scl)

    pyxel.mouse(True)
    pyxel.run(update, draw)


def update():
    global numeros
    global vectores_ideales
    global numero
    global prediccion

    # index = (pyxel.frame_count / 50) % len(numeros)
    # numero.matriz = numeros[int(index)]["matriz"]
    # prediccion = predict(numero.matriz, mostrar_distancias=False)
    # print("Predición:", prediccion, "Label:", numeros[int(index)]["label"])

    # index = (pyxel.frame_count / 20) % len(vectores_ideales)
    # numero.matriz = np.reshape(vectores_ideales[int(index)], (28, 28))

    # Condición de salida
    if pyxel.btnp(pyxel.KEY_Q):
        pyxel.quit()

    # Si se hace click, se añade un pixel a la matriz
    if pyxel.btn(pyxel.MOUSE_BUTTON_LEFT):
        # Pintamos de blanco el pixel que estamos pulsando
        # y con tonos grises los adjacentes (si no están ya pintados de blanco)
        x, y = pyxel.mouse_x, pyxel.mouse_y
        numero.pintar_pixeles(x, y, gris=0.7, max=1)

    # Predecir el número dibujado
    if pyxel.btnp(pyxel.KEY_RETURN) or pyxel.btnp(pyxel.KEY_SPACE):
        prediccion = predict(numero.matriz, mostrar_distancias=False)

    # Borrar el número
    if pyxel.btnp(pyxel.KEY_BACKSPACE):
        numero.matriz = np.zeros((28, 28))
        prediccion = -1


def predict(matriz, mostrar_distancias=False):
    global vectores_ideales
    # Vectorizamos la imagen
    vector_img = matriz.reshape((1, 784))

    distancias = []
    # Calculamos la distancia euclídea (Pitágoras)
    for vect in vectores_ideales:
        sumatorio = np.sum((vect - vector_img) ** 2)
        dist = np.sqrt(sumatorio)
        distancias.append(dist)

    # Mismo procedimiento que en el método 1:
    # Nos interesa el menor valor, asi que invertimos los números para
    # calcular las probabilidades
    valor_maximo = np.max(distancias)
    distancias_invertidas = np.subtract(distancias, valor_maximo)

    sum_distancias = np.sum(distancias_invertidas)
    vector_probabilidades = distancias_invertidas / sum_distancias
    vector_probabilidades = list(vector_probabilidades)

    max_probabilidad = max(vector_probabilidades)
    index = vector_probabilidades.index(max_probabilidad)

    if mostrar_distancias:
        print("Distancias:\n", distancias)
        print("Probabilidad:\n", vector_probabilidades, "-->", index)

    return index


def draw():
    global prediccion
    pyxel.cls(0)

    if prediccion >= 0:
        txt = f"HAS DIBUJADO EL NUMERO {prediccion}"
    else:
        txt = "DIBUJA UN NUMERO"
        
    longitud_string = len(txt) * 4
        
    pyxel.text((WIDTH - longitud_string) / 2, 15, txt, pyxel.COLOR_ORANGE)

    numero.draw()

    # Dibujamos el cuadrado para la zona de dibujo
    pyxel.rectb(
        offset_x,
        offset_y,
        AREA_DIBUJO["w"] * AREA_DIBUJO["scl"],
        AREA_DIBUJO["h"] * AREA_DIBUJO["scl"],
        pyxel.COLOR_WHITE,
    )


main()
