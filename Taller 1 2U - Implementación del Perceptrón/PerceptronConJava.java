import java.util.Random;
import java.util.Scanner;

public class Perceptron {
    private double[] weights;
    private double bias;
    private double learningRate;
    private int epochs;

    public Perceptron(int inputSize, double learningRate, int epochs) {
        this.weights = new double[inputSize];
        this.bias = new Random().nextDouble() * 0.1;
        this.learningRate = learningRate;
        this.epochs = epochs;
        initializeWeights();
    }

    private void initializeWeights() {
        Random rand = new Random();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = rand.nextDouble() * 0.1;
        }
    }

    private double activationFunction(double x, String functionType) {
        switch (functionType.toLowerCase()) {
            case "lineal":
                return x;
            case "escalon":
                return x >= 0 ? 1 : 0;
            case "sigmoidal":
                return 1 / (1 + Math.exp(-x));
            case "relu":
                return Math.max(0, x);
            case "tanh":
                return Math.tanh(x);
            case "softmax":
                return activationSoftmax(x); // Ahora usa la función corregida
            default:
                throw new IllegalArgumentException("Función de activación no reconocida.");
        }
    }
    

    private int activationSoftmax(double x) {
        double[] logits = {Math.exp(-x), Math.exp(x)}; // Calculamos e^x para 0 y 1
        double sum = logits[0] + logits[1]; // Normalizamos sumando ambas probabilidades
        double probability1 = logits[1] / sum; // Probabilidad de la salida ser 1
    
        return (probability1 >= 0.5) ? 1 : 0; // Si la probabilidad de 1 es >= 0.5, activamos 1, sino 0
    }
    

    public void train(int[][] inputs, int[] outputs, String activationType) {
        System.out.println("\nEntrenamiento del Perceptrón:");
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("\nÉpoca " + (epoch + 1) + ":");
            for (int i = 0; i < inputs.length; i++) {
                double sum = bias;
                for (int j = 0; j < weights.length; j++) {
                    sum += inputs[i][j] * weights[j];
                }
                double prediction = activationFunction(sum, activationType);

                // Softmax genera valores entre 0 y 1, convertimos a binario
                int outputPrediction = (prediction >= 0.5) ? 1 : 0;
                double error = outputs[i] - outputPrediction;

                System.out.printf("Entrada: (%d, %d) => Esperado: %d, Obtenido: %d, Error: %.2f\n",
                        inputs[i][0], inputs[i][1], outputs[i], outputPrediction, error);

                for (int j = 0; j < weights.length; j++) {
                    weights[j] += learningRate * error * inputs[i][j];
                }
                bias += learningRate * error;
            }
        }
    }

    public int predict(int[] input, String activationType) {
        double sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += input[i] * weights[i];
        }
        double result = activationFunction(sum, activationType);
        
        return (result >= 0.5) ? 1 : 0;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.println("Seleccione la función de activación:");
        System.out.println("1. Lineal");
        System.out.println("2. Escalón");
        System.out.println("3. Sigmoidal");
        System.out.println("4. ReLU");
        System.out.println("5. Tanh");
        System.out.println("6. Softmax");
        System.out.print("Ingrese el número de la opción: ");
        
        int option = scanner.nextInt();
        String activationType = "escalon";

        switch (option) {
            case 1: activationType = "lineal"; break;
            case 2: activationType = "escalon"; break;
            case 3: activationType = "sigmoidal"; break;
            case 4: activationType = "relu"; break;
            case 5: activationType = "tanh"; break;
            case 6: activationType = "softmax"; break;
            default:
                System.out.println("Opción no válida. Se usará la función Escalón por defecto.");
        }

        int[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] outputs = {0, 0, 0, 1}; // AND lógico

        Perceptron perceptron = new Perceptron(2, 0.1, 10);
        perceptron.train(inputs, outputs, activationType);

        System.out.println("\nResultados finales del Perceptrón para AND lógico:");
        for (int[] input : inputs) {
            System.out.println("Entrada: (" + input[0] + ", " + input[1] + ") => Salida: " + perceptron.predict(input, activationType));
        }
        scanner.close();
    }
}
