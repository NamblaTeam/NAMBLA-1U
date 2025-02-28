import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# 📌 Definir la ruta del archivo (ya está en el entorno de Colab)
ruta_csv = "/content/Salary Data.csv"  # Asegúrate de que el nombre coincide

# 📌 Cargar los datos
datos = pd.read_csv(ruta_csv, encoding="latin1")

# 📌 Mostrar las primeras filas para verificar la carga
print("Primeras filas del dataset:")
print(datos.head())

# 📌 Eliminar valores nulos y duplicados
datos.dropna(inplace=True)
datos.drop_duplicates(inplace=True)

# 📌 Convertir variables categóricas en numéricas
label_encoders = {}
for col in ["Gender", "Education Level", "Job Title"]:
    le = LabelEncoder()
    datos[col] = le.fit_transform(datos[col])
    label_encoders[col] = le  # Guardamos los codificadores para usarlos después

# 📌 Características (X) y variable objetivo (y)
X = datos[["Years of Experience"]]  # Solo usamos "Years of Experience"
y = datos["Salary"]

# 📌 Convertir a arrays
X_procesada = X.values
y_procesada = y.values.reshape(-1, 1)

# 📌 Dividir datos en entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X_procesada, y_procesada, test_size=0.2, random_state=42)

# 📌 Crear y entrenar el modelo de regresión polinomial
grado_polinomio = 3  # Ajustamos el grado del polinomio
modelo = make_pipeline(PolynomialFeatures(grado_polinomio), LinearRegression())
modelo.fit(X_train, y_train)

# 📌 Evaluar el modelo
score = modelo.score(X_test, y_test)
print(f"Precisión del modelo en datos de prueba: {score:.2f}")

# 📌 Pedir al usuario años de experiencia para predecir salario
exp_usuario = float(input("Ingrese sus años de experiencia para estimar su salario: "))
prediccion = modelo.predict([[exp_usuario]])
print(f"Para {exp_usuario} años de experiencia, el salario estimado es ${prediccion[0][0]:,.2f}")

# 📌 GRAFICAR LA REGRESIÓN POLINOMIAL
plt.figure(figsize=(8,6))

# 🔹 Graficar datos reales
sb.scatterplot(x=datos["Years of Experience"], y=datos["Salary"], label="Datos reales", color="blue")

# 🔹 Crear valores de experiencia para predecir la curva
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = modelo.predict(X_range)

# 🔹 Graficar la curva de regresión polinomial
plt.plot(X_range, y_pred, color="red", label="Regresión Polinomial", linewidth=2)

# 🔹 Etiquetas y título
plt.xlabel("Años de Experiencia")
plt.ylabel("Salario")
plt.title("Regresión Polinomial: Experiencia vs Salario")
plt.legend()
plt.grid()

# 📌 Mostrar la gráfica
plt.show()
