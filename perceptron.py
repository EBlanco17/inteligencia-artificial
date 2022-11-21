import random 

class Perceptron:

    def __init__(self, inputSize):
        self.tamano = inputSize
        self.weights = self.random_weights()
        self.umbral = random.uniform(-10.0, 10.0) 

    def random_weights(self):
        """
        Crea una lista de números aleatorios entre -1.0 y 1.0 que se asignan a los pesos
        """
        weigths = []
        for i in range(self.tamano):
            weigths.append(random.uniform(-1.0, 1.0))
        return weigths

    def normalizeValue(self,val):
        if val < 0:
            return -1
        else:
            return 1

    def processInput(self, nnInput):
        """
        Toma el producto de la entrada y los pesos, agrega el umbral y luego normaliza el valor
        """
        assert len(nnInput) == len(self.weights)
        unprocessedOutputVal = self.umbral 

        for i in range(len(nnInput)):
            unprocessedOutputVal += nnInput[i] * self.weights[i]

        return self.normalizeValue(unprocessedOutputVal)

    def trainOnInput(self, inputVals, expectedOutputVal, learningRate):
        """
        La función toma un valor de entrada, un valor de salida esperado y una tasa de aprendizaje.
        """
        nnVal = self.processInput(inputVals)
        error = expectedOutputVal - nnVal
        self.adjustForError(inputVals, error, learningRate)


    def adjustForError(self, inputVals, error, learningRate):
        """
        La función ajusta los pesos y el umbral de la neurona en función del error y la tasa de aprendizaje.
        """
        for i in range(len(self.weights)):
            self.weights[i] += error * inputVals[i] * learningRate
        self.umbral += error * learningRate

##################################################################

def generateTrainingSet(trainingSize):
    """
    Genera una lista de números aleatorios entre 0.0 y 1.0 y los asigna a la lista de entrenamiento.
    """
    lista = []
    for i in range(trainingSize):
        lista.append(float (random.randint(0, 1)))
    return lista

dataset = [
    [[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 
    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],0],
    [[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 
    1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],0],
    [[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 
    0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],0],
    [[1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 
    0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],1],
]

inputs = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 
0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
n = inputs.__len__()
perceptron = Perceptron(n)
trainingRate = 0.1
trainingSetSize = n
trainingSet = generateTrainingSet(trainingSetSize)
print("Pesos Aleactorios: ", perceptron.weights)
# Agrega 1000 datos aleatorios de 36 numeros al conjunto de datos de entrenamiento.
for i in range(1000):
    dataset.append([generateTrainingSet(trainingSetSize), 0])   
z = 0
tries = 0
while z < 1:
    
    for i in range(len(dataset)):
        if tries == 1000:
            print("No se pudo entrenar la red")
            tries = 0
            perceptron.weights = perceptron.random_weights()
            
        else:
            weigths = perceptron.trainOnInput(dataset[i][0], dataset[i][1], trainingRate)
            tries += 1
            #print ("umbral: ", perceptron.umbral)

    z = perceptron.processInput(inputs)
    print("Intentos: ", tries)
    print(perceptron.weights)
    print("Salida: ", z)