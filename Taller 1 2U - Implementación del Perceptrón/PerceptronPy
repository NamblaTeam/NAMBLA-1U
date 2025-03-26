import numpy as np

def linear(x):
    return x

def step(x):
    return 1 if x >= 0 else 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def tanh(x):
    return np.tanh(x)

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10, activation='step'):
        self.weights = np.random.rand(input_size + 1)  # Pesos + sesgo
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation
    
    def activation_function(self, x):
        if self.activation == 'linear':
            return linear(x)
        elif self.activation == 'step':
            return step(x)
        elif self.activation == 'sigmoid':
            return sigmoid(x)
        elif self.activation == 'relu':
            return relu(x)
        elif self.activation == 'softmax':
            return softmax(np.array([x]))[0]  # Softmax espera un array
        elif self.activation == 'tanh':
            return tanh(x)
        else:
            raise ValueError("Función de activación no reconocida")
    
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # Producto punto + sesgo
        return self.activation_function(summation)
    
    def train(self, training_inputs, labels):
        print("\nInicio del entrenamiento:")
        for epoch in range(self.epochs):
            print(f"\nÉpoca {epoch + 1}:")
            total_error = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                total_error += abs(error)
                self.weights[1:] += self.learning_rate * error * np.array(inputs)
                self.weights[0] += self.learning_rate * error  # Ajuste del sesgo
                print(f"Entrada: {inputs}, Salida esperada: {label}, Predicción: {prediction}, Error: {error}")
            print(f"Error total en época {epoch + 1}: {total_error}")
            if total_error == 0:
                print("No hay más errores, finalizando entrenamiento.")
                break

# Datos de entrenamiento para la compuerta lógica AND
training_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
labels = np.array([0, 0, 0, 1])  # Salida esperada

# Seleccionar la función de activación
def main():
    print("Seleccione la función de activación:")
    print("1. Lineal")
    print("2. Escalón")
    print("3. Sigmoide")
    print("4. ReLU")
    print("5. Softmax")
    print("6. Tangente Hiperbólica")
    
    choice = input("Ingrese el número de la función de activación: ")
    activation_functions = {'1': 'linear', '2': 'step', '3': 'sigmoid', '4': 'relu', '5': 'softmax', '6': 'tanh'}
    activation = activation_functions.get(choice, 'step')
    
    print(f"\nEntrenando con función de activación: {activation}")
    perceptron = Perceptron(input_size=2, activation=activation)
    perceptron.train(training_inputs, labels)
    
    # Probar el perceptrón
    print("\nResultados finales:")
    for inputs in training_inputs:
        print(f"Entrada: {inputs}, Salida: {perceptron.predict(inputs)}")

if __name__ == "__main__":
    main()
