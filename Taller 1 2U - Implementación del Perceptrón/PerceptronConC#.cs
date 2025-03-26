using System;
using System.Linq;

class Perceptron
{
    static Random rand = new Random();
    static double[] pesos = { rand.NextDouble() * 2 - 1, rand.NextDouble() * 2 - 1 };
    static double b = rand.NextDouble() * 2 - 1;
    static double tasaAprendizaje = 0.1;
    static int epocas = 100;

    // Funciones de activación
    static double Lineal(double x) => x;
    static int Escalon(double x) => x > 0 ? 1 : 0;
    static double Sigmoide(double x) => 1 / (1 + Math.Exp(-x));
    static double ReLU(double x) => Math.Max(0, x);
    static double Tanh(double x) => Math.Tanh(x);

    // Función Softmax (se aplica a un conjunto de valores)
    static double[] Softmax(double[] valores)
    {
        double sumaExp = valores.Sum(v => Math.Exp(v));
        return valores.Select(v => Math.Exp(v) / sumaExp).ToArray();
    }

    // Función de activación generalizada
    static double Activacion(double suma, string tipo, double[]? valores = null)
    {
        return tipo switch
        {
            "lineal" => Lineal(suma),
            "escalon" => Escalon(suma),
            "sigmoide" => Sigmoide(suma),
            "relu" => ReLU(suma),
            "tanh" => Tanh(suma),
            "softmax" => valores != null ? Softmax(valores)[0] : 0, // Usa Softmax si hay un conjunto de valores
            _ => Escalon(suma) // Por defecto usa escalón
        };
    }

    // Función de entrenamiento del perceptrón
    static void Entrenar(double[][] datos, int[] clases, string tipoActivacion)
    {
        for (int epoca = 0; epoca < epocas; epoca++)
        {
            int errorTotal = 0;
            for (int i = 0; i < datos.Length; i++)
            {
                double suma = pesos[0] * datos[i][0] + pesos[1] * datos[i][1] + b;

                // Para Softmax, pasamos todas las entradas y obtenemos la primera salida
                double salida = tipoActivacion == "softmax" 
                    ? Activacion(0, tipoActivacion, new double[] { suma }) 
                    : Activacion(suma, tipoActivacion);

                int prediccion = (int)Math.Round(salida); // Convertir salida continua a 0 o 1
                int error = clases[i] - prediccion;
                errorTotal += error * error;

                // Ajuste de pesos y sesgo
                pesos[0] += tasaAprendizaje * datos[i][0] * error;
                pesos[1] += tasaAprendizaje * datos[i][1] * error;
                b += tasaAprendizaje * error;
            }
            Console.Write(errorTotal + " ");
        }
        Console.WriteLine();
    }

    static void Main()
    {
        // Datos de la tabla AND
        double[][] datosAND =
        {
            new double[] {0, 0}, new double[] {0, 1},
            new double[] {1, 0}, new double[] {1, 1}
        };
        int[] clasesAND = { 0, 0, 0, 1 };

        // Seleccionar la función de activación
        string tipoActivacion = "tanh"; 

        // Entrenar el perceptrón
        Entrenar(datosAND, clasesAND, tipoActivacion);

        // Evaluar el perceptrón
        Console.WriteLine("\nResultados del perceptrón para la tabla AND:");
        foreach (var dato in datosAND)
        {
            double suma = pesos[0] * dato[0] + pesos[1] * dato[1] + b;

            // Para Softmax, pasamos todas las entradas y obtenemos la primera salida
            double resultado = tipoActivacion == "softmax"
                ? Activacion(0, tipoActivacion, new double[] { suma })
                : Activacion(suma, tipoActivacion);

            int salidaFinal = (int)Math.Round(resultado); // Convertir a 0 o 1
            Console.WriteLine($"Entrada: ({dato[0]}, {dato[1]}) => Salida: {salidaFinal}");
        }
    }
}
