import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar el dataset
df = pd.read_csv("Salary_Data.csv")
df.dropna(inplace=True)

# 2. Codificación de variables categóricas
label_enc = LabelEncoder()
df["Gender"] = label_enc.fit_transform(df["Gender"])
df["Education Level"] = label_enc.fit_transform(df["Education Level"])
df["Job Title"] = label_enc.fit_transform(df["Job Title"])

# 3. Seleccionar variable independiente y dependiente
X = df[["Years of Experience"]]
y = df["Salary"]

# 4. Normalización de la variable independiente
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Transformación polinómica
degree = 2  # Se puede cambiar el grado del polinomio
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 7. Entrenar el modelo
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 8. Predicciones en datos de prueba
y_pred = model.predict(X_test_poly)

# 9. Evaluación general del modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n🔹 Evaluación General del Modelo:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")
print("=" * 50)

# 10. Predicciones según los años de experiencia ingresados por el usuario
experiencia_input = [1, 3, 5, 10, 15, 20, 25, 30, 35]  # Lista de años de experiencia para hacer predicciones

print("\n📌 Predicciones según los años de experiencia:")
for exp in experiencia_input:
    exp_scaled = scaler.transform(pd.DataFrame([[exp]], columns=["Years of Experience"]))  # ✅ Corrección aplicada
    exp_poly = poly.transform(exp_scaled)  # Transformar a polinómico
    salario_predicho = model.predict(exp_poly)[0]  # Predicción

    print(f"Años de experiencia: {exp} ➝ Salario predicho: ${salario_predicho:.2f}")
print("=" * 50)

# 11. Visualización de la curva de regresión
X_sorted = np.sort(X_scaled, axis=0)  # Ordenar para graficar correctamente
X_poly_sorted = poly.transform(X_sorted)  # Transformar a polinómico
y_sorted = model.predict(X_poly_sorted)  # Obtener predicciones

plt.scatter(X, y, color="#839192", s=9.0,label="Datos reales")  # Puntos reales
plt.plot(scaler.inverse_transform(X_sorted), y_sorted, color="#d4ac0d", linewidth=2, label="Regresión Polinómica")  # Línea de predicción
plt.xlabel("Años de Experiencia")
plt.ylabel("Salario")
plt.title("Regresión Polinómica del Salario según Años de Experiencia")
plt.legend()
plt.show()
