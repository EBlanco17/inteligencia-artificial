#usando kvecinos predecir star rating usando sentimentValue, wordcount 
import json

class KVecinos:
    def __init__(self, wordcount, sentimentValue, k, data):
        self.data = data
        self.sentimentValue = sentimentValue
        self.wordcount = wordcount
        self.k = k
        
    def distancia(self, p1, p2):
        # Cálculo de la distancia entre dos puntos.
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    def kvecinos(self):
        # Cálculo de la distancia entre el punto y el resto de puntos del conjunto de datos.
        distancias = []
        for i in range(len(self.data)):
            distancias.append((self.distancia((self.wordcount, self.sentimentValue), (self.data[i][0], self.data[i][1])), self.data[i][2]))
        distancias.sort()
        return distancias[:self.k]

    def predecir(self):
        # Cálculo de la media de los k vecinos más próximos.
        kvecinos = self.kvecinos()
        suma = 0
        for i in range(len(kvecinos)):
            suma += kvecinos[i][1]
        return suma / len(kvecinos)

    def clasificar(self):
        if self.predecir() >= 4:
            return "Bueno"
        elif self.predecir() >=2 and self.predecir() < 4:
            return "Regular"
        else:
            return "Malo"

    def __str__(self):
        return "Star Rating: " + str(int(round(self.predecir(),0))) + "\nClasificacion: " + self.clasificar()

def leerDatos():
    data = []
   
    f = open('sentimientos.json', 'r')
    content = f.read()
    jsondecode = json.loads(content)
    data = list()

    for entity in jsondecode:
        data.append([int(entity['wordcount']), float(entity['sentimentValue']), int(entity['Star Rating'])])
    return data

def main():
    data = leerDatos()
    continuar = 'Y'
    while continuar == 'Y' or continuar == 'y':
        wordcount = int(input("Ingrese wordcount: "))
        sentimentValue = float(input("Ingrese sentimentValue: "))
        k = int(input("Ingrese k: "))
        kvecinos = KVecinos(wordcount, sentimentValue, k, data)
        print(kvecinos)
        continuar = input("Desea continuar? (Y/N): ")
    
if __name__ == "__main__":
    main()