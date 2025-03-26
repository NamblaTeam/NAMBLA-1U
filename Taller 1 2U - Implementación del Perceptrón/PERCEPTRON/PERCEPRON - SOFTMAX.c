#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h> // Para la función exponencial

// Definir la cantidad de entradas y épocas
#define NUM_ENTRADAS 2
#define EPOCAS 100
#define TASA_DE_APRENDIZAJE 0.01

// Función de activación: Softmax con corrección para evitar desbordamientos
double softmax(double *pesos, int *entrada, double b, int num_clases) {
    double suma[2];  // Para las dos clases posibles (en nuestro caso: 0 o 1)
    double suma_exponencial = 0.0;

    // Calculamos las sumas ponderadas
    for (int i = 0; i < NUM_ENTRADAS; i++) {
        suma[0] += pesos[0] * entrada[i]; // Para la clase 0
        suma[1] += pesos[1] * entrada[i]; // Para la clase 1
    }
    
    // Añadimos el sesgo
    suma[0] += b;
    suma[1] += b;

    // Estabilizamos las exponenciales restando el máximo de las sumas
    double max_suma = (suma[0] > suma[1]) ? suma[0] : suma[1];
    
    // Calculamos las exponenciales de las sumas estabilizadas
    double exp_suma[2];
    exp_suma[0] = exp(suma[0] - max_suma); // Restamos el máximo para evitar grandes números
    exp_suma[1] = exp(suma[1] - max_suma);
    
    // Sumamos las exponenciales
    suma_exponencial = exp_suma[0] + exp_suma[1];
    
    // Retornamos la probabilidad de la clase 1 (ya que la suma de ambos será 1)
    return exp_suma[1] / suma_exponencial;
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
    printf("Resultados del perceptrón para la tabla AND (usando activación Softmax):\n");

    // Entrenamiento
    for (int epoca = 0; epoca < EPOCAS; epoca++) {
        double error_total = 0;
        for (int i = 0; i < 4; i++) {
            // Predicción con Softmax (solo usamos la clase 1)
            double prediccion = softmax(pesos, entradas[i], b, 2);
            double error = clases[i] - prediccion;
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
        double prediccion = softmax(pesos, entradas[i], b, 2);
        printf("Entrada: (%d, %d) => Salida: %.4f\n", entradas[i][0], entradas[i][1], prediccion);
    }

    return 0;
}
