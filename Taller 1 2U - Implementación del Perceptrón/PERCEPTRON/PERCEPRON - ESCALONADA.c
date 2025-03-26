#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Definir la cantidad de entradas y épocas
#define NUM_ENTRADAS 2
#define EPOCAS 100
#define TASA_DE_APRENDIZAJE 0.01

// Función de activación: Escalonada
int activacion(double *pesos, int *entrada, double b) {
    double suma = 0.0;
    for (int i = 0; i < NUM_ENTRADAS; i++) {
        suma += pesos[i] * entrada[i]; // Multiplicamos los pesos por las entradas
    }
    suma += b; // Añadimos el sesgo
    return (suma > 0) ? 1 : 0; // Si la suma ponderada es mayor que 0, la salida es 1, de lo contrario 0
}

int main() {
    // Datos de las entradas binarias para AND
    int entradas[4][NUM_ENTRADAS] = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };

    // Clases: 1 -> aprobado, 0 -> denegado (según tabla AND)
    int clases[4] = {0, 0, 0, 1};

    // Inicialización de pesos y sesgo
    double pesos[NUM_ENTRADAS];
    double b;
    
    // Inicialización aleatoria de pesos y sesgo entre -1 y 1
    srand(time(NULL));  // Usar la semilla para números aleatorios
    for (int i = 0; i < NUM_ENTRADAS; i++) {
        pesos[i] = ((double)rand() / RAND_MAX) * 2 - 1; // Pesos entre -1 y 1
    }
    b = ((double)rand() / RAND_MAX) * 2 - 1; // Sesgo entre -1 y 1

    // Imprimir los pesos y el sesgo iniciales
    printf("Resultados del perceptrón para la tabla AND (usando activación escalonada):\n");

    // Entrenamiento
    for (int epoca = 0; epoca < EPOCAS; epoca++) {
        double error_total = 0;
        for (int i = 0; i < 4; i++) {
            // Predicción
            int prediccion = activacion(pesos, entradas[i], b);
            int error = clases[i] - prediccion;
            error_total += error * error; // Acumulamos el error cuadrado

            // Actualización de pesos y sesgo
            pesos[0] += TASA_DE_APRENDIZAJE * entradas[i][0] * error;
            pesos[1] += TASA_DE_APRENDIZAJE * entradas[i][1] * error;
            b += TASA_DE_APRENDIZAJE * error;
        }

        // Imprimir el error total por cada época (opcional, puede descomentarse si deseas ver el progreso)
        // printf("Época %d: Error Total: %.4f\n", epoca + 1, error_total);
    }

    // Mostrar los resultados de la tabla AND
    for (int i = 0; i < 4; i++) {
        int prediccion = activacion(pesos, entradas[i], b);
        printf("Entrada: (%d, %d) => Salida: %d\n", entradas[i][0], entradas[i][1], prediccion);
    }

    return 0;
}
