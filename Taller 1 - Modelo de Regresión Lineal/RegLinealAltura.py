import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import joblib
import os
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Definir la ruta del CSV
ruta_csv = os.path.join(os.path.dirname(os.path.abspath(_file_)), "..", "DATOS", "datos_altura.csv")

# Cargar los datos desde el CSV con codificación adecuada
datos = pd.read_csv(ruta_csv, encoding="latin1")

# Características (X) y etiqueta (y)
X = datos[["Edad"]]
y = datos.iloc[:, 1:].mean(axis=1)  # Promedio de las alturas como variable objetivo

# Procesar los datos, convertir de series a arreglos
X_procesada = X.values
y_procesada = y.values.reshape(-1, 1)

# Crear y entrenar el modelo de regresión polinómica con un ajuste mejorado
grado_polinomio = 3  # Se ajusta el grado para mejor precisión sin sobreajuste
modelo = make_pipeline(PolynomialFeatures(grado_polinomio), LinearRegression())
modelo.fit(X_procesada, y_procesada)

# Guardar el modelo entrenado
ruta_modelo = os.path.join(os.path.dirname(os.path.abspath(_file_)), "..", "DATOS", "modelo_altura.pkl")
joblib.dump(modelo, ruta_modelo)
print(f"Modelo de predicción de altura entrenado y guardado en: {ruta_modelo}")

# Pedir al usuario una edad para predecir la altura
edad_usuario = int(input("Ingrese una edad para estimar la altura (1-80 años): "))
while edad_usuarioimport pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import joblib
import os
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Definir la ruta del CSV
ruta_csv = os.path.join(os.path.dirname(os.path.abspath(_file_)), "..", "DATOS", "datos_altura.csv")

# Cargar los datos desde el CSV con codificación adecuada
datos = pd.read_csv(ruta_csv, encoding="latin1")

# Características (X) y etiqueta (y)
X = datos[["Edad"]]
y = datos.iloc[:, 1:].mean(axis=1)  # Promedio de las alturas como variable objetivo

# Procesar los datos, convertir de series a arreglos
X_procesada = X.values
y_procesada = y.values.reshape(-1, 1)

# Crear y entrenar el modelo de regresión polinómica con un ajuste mejorado
grado_polinomio = 3  # Se ajusta el grado para mejor precisión sin sobreajuste
modelo = make_pipeline(PolynomialFeatures(grado_polinomio), LinearRegression())
modelo.fit(X_procesada, y_procesada)

# Guardar el modelo entrenado
ruta_modelo = os.path.join(os.path.dirname(os.path.abspath(_file_)), "..", "DATOS", "modelo_altura.pkl")
joblib.dump(modelo, ruta_modelo)
print(f"Modelo de predicción de altura entrenado y guardado en: {ruta_modelo}")

# Pedir al usuario una edad para predecir la altura
edad_usuario = int(input("Ingrese una edad para estimar la altura (1-80 años): "))
while edad_usuario < 1 or edad_usuario > 80:
    edad_usuario = int(input("Edad fuera de rango. Ingrese una edad válida (1-80 años): "))

prediccion = modelo.predict([[edad_usuario]])
prediccion = np.clip(prediccion, 0.5, 2.1)  # Limitar valores de altura a un rango realista
print(f"Para edad {edad_usuario} años, la altura estimada es {prediccion[0][0]:.2f} m")

# Evaluar el modelo
score = modelo.score(X_procesada, y_procesada)
print(f"Precisión del modelo: {score:.2f}")

# Visualización con Seaborn
plt.figure(figsize=(8,6))
sb.scatterplot(x=datos["Edad"], y=y, label="Datos reales")
edades_pred = np.linspace(min(X.values), max(X.values), 100)
predicciones = modelo.predict(edades_pred.reshape(-1, 1))
predicciones = np.clip(predicciones, 0.5, 2.1)  # Aplicar límite visual
sb.lineplot(x=edades_pred.flatten(), y=predicciones.flatten(), color="red", label="Regresión polinómica")
plt.xlabel("Edad")
plt.ylabel("Altura promedio (m)")
plt.title("Relación entre Edad y Altura Promedio")
plt.legend()
plt.show() < 1 or edad_usuario > 80:
    edad_usuario = int(input("Edad fuera de rango. Ingrese una edad válida (1-80 años): "))

prediccion = modelo.predict([[edad_usuario]])
prediccion = np.clip(prediccion, 0.5, 2.1)  # Limitar valores de altura a un rango realista
print(f"Para edad {edad_usuario} años, la altura estimada es {prediccion[0][0]:.2f} m")

# Evaluar el modelo
score = modelo.score(X_procesada, y_procesada)
print(f"Precisión del modelo: {score:.2f}")

# Visualización con Seaborn
plt.figure(figsize=(8,6))
sb.scatterplot(x=datos["Edad"], y=y, label="Datos reales")
edades_pred = np.linspace(min(X.values), max(X.values), 100)
predicciones = modelo.predict(edades_pred.reshape(-1, 1))
predicciones = np.clip(predicciones, 0.5, 2.1)  # Aplicar límite visual
sb.lineplot(x=edades_pred.flatten(), y=predicciones.flatten(), color="red", label="Regresión polinómica")
plt.xlabel("Edad")
plt.ylabel("Altura promedio (m)")
plt.title("Relación entre Edad y Altura Promedio")
plt.legend()
plt.show()
