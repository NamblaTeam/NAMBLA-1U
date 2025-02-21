mport pandas as pd
import numpy as np
import os

# Definir la ruta para guardar el CSV
ruta_csv = os.path.join(os.path.dirname(os.path.abspath(_file_)), "..", "DATOS", "datos_altura.csv")

# Generar datos realistas
np.random.seed(42)
edades = np.random.randint(1, 81, size=100)
edades.sort()  # Ordenar edades para mayor coherencia en el crecimiento

# Fórmula más realista basada en el crecimiento humano
altura_base = 0.5  # Altura base en metros para un recién nacido

def altura_realista(edad):
    if edad <= 2:
        return 0.5 + edad * 0.35  # Crecimiento acelerado en los primeros años
    elif edad <= 12:
        return 0.9 + (edad - 2) * 0.07  # Crecimiento en la infancia
    elif edad <= 18:
        return 1.6 + (edad - 12) * 0.05  # Adolescencia
    else:
        return 1.75 + np.random.normal(0, 0.02)  # Adultos con variación leve

alturas1 = np.array([altura_realista(e) + np.random.normal(0, 0.03) for e in edades])
alturas2 = np.array([altura_realista(e) + np.random.normal(0, 0.04) for e in edades])
alturas3 = np.array([altura_realista(e) + np.random.normal(0, 0.035) for e in edades])
alturas4 = np.array([altura_realista(e) + np.random.normal(0, 0.038) for e in edades])

# Crear DataFrame
datos = pd.DataFrame({
    "Edad": edades,
    "Altura1 (m)": alturas1,
    "Altura2 (m)": alturas2,
    "Altura3 (m)": alturas3,
    "Altura4 (m)": alturas4
})

# Guardar en CSV
datos.to_csv(ruta_csv, index=False, encoding="latin1")
print(f"CSV generado y guardado en: {ruta_csv}")
