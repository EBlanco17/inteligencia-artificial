import networkx as nx
import matplotlib.pyplot as plt

class Dijsktra:
    # rutas = {
    #     'A':{'D':5, 'E':4, 'B':8},
    #     'B':{'A':8, 'E':12, 'F':4, 'C':3},
    #     'C':{'B':3, 'F':9, 'G':11},
    #     'D':{'A':5, 'H':6, 'E':9},
    #     'E':{'D':9, 'A':4, 'B':12, 'F':3, 'J':5, 'I':8},
    #     'F':{'B':4, 'C':9, 'E':3, 'G':1, 'K':8},
    #     'G':{'C':11, 'F':1, 'L':7, 'K':8},
    #     'H':{'D':6, 'I':2, 'M':7},
    #     'I':{'E':8, 'M':6, 'H':2, 'J':10},
    #     'J':{'I':10, 'E':5, 'K':6,'N':9},
    #     'K':{'F':8, 'G':8, 'J':6, 'L':5, 'P':7},
    #     'L':{'G':7, 'K':5, 'P':6},
    #     'M':{'H':7, 'I':6, 'N':2},
    #     'N':{'J':9, 'M':2, 'P':12},
    #     'P':{'K':7, 'L':6, 'N':12}

    # }
    rutas = {
        'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
        'Zerind': {'Oradea': 71, 'Oradea': 71},
        'Oradea': {'Sibiu': 151, 'Zerind': 71},
        'Timisoara': {'Lugoj': 111, 'Arad': 118},
        'Lugoj': {'Mehadia': 70, 'Timisoara': 111},
        'Mehadia': {'Lugoj': 70, 'Dobreta': 75},
        'Dobreta': {'Mehadia': 75, 'Craiova': 120},
        'Craiova': {'RimnicuVilcea': 146, 'Pitesti': 138, 'Dobreta': 120},
        'RimnicuVilcea': {'Pitesti': 97, 'Craiova': 146},
        'Sibiu': {'Arad': 140, 'Fagaras': 99, 'RimnicuVilcea': 80},
        'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
        'Pitesti': {'RimnicuVilcea': 97, 'Craiova': 138, 'Bucharest': 101},
        'Bucharest': {'Fagaras': 211, 'Giurgiu': 90, 'Pitesti': 101, 'Urziceni': 85},
        'Giurgiu': {'Bucharest': 90},
        'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
        'Hirsova': {'Urziceni': 98, 'Eforie': 86},
        'Eforie': {'Hirsova': 86},
        'Vaslui': {'Urziceni': 142, 'Iasi': 92},
        'Iasi': {'Vaslui': 92, 'Neamt': 87},
        'Neamt': {'Iasi': 87}
    }

    def __init__(self):
        self.camino_corto = None
        
    def setOrigen(self, origen):
        self.origen = origen
    
    def setDestino(self, destino):
        self.destino = destino
    
    def busqueda(self):
        self.camino_corto = {self.origen: (None, 0)}
        ubicacion = self.origen
        ya_Recorrido = set()
        # Comprobando si la ubicación actual es el destino.
        while ubicacion != self.destino:
            for vecino, peso in self.rutas[ubicacion].items():
                if vecino not in ya_Recorrido:
                    costo = self.camino_corto[ubicacion][1] + peso
                    # Si el costo del vecino es menor que el costo del nodo actual, entonces actualiza el costo del nodo actual
                    if vecino not in self.camino_corto or costo < self.camino_corto[vecino][1]:
                        # actualizando la ruta más corta.
                        self.camino_corto[vecino] = (ubicacion, costo)
            ya_Recorrido.add(ubicacion)
            menor_costo = float("inf")
            for nodo in self.camino_corto:
                if nodo not in ya_Recorrido:
                    # encontrar el nodo que no se ha visitado, con el costo más bajo del nodo actual.
                    if self.camino_corto[nodo][1] < menor_costo:
                        menor_costo = self.camino_corto[nodo][1]
                        ubicacion = nodo

        return self.camino_corto[self.destino][1], self.getCamino()
        
    def getCamino(self):
        camino = []
        while self.destino != None:
            camino.append(self.destino)
            self.destino = self.camino_corto[self.destino][0]
        camino.reverse()
        return camino

    #funcion que dibuja el grafo, coloreando los nodos según el camino más corto.
    def dibujar(self, camino):
        G = nx.Graph()
        G.add_nodes_from(self.rutas.keys())
        for nodo, vecinos in self.rutas.items():
            for vecino, peso in vecinos.items():
                G.add_edge(nodo, vecino, w=peso)
        pos = nx.spring_layout(G)
        nodos = list(G.nodes())
        
        color = ['red' if nodo in camino else 'blue' for nodo in nodos]
        color_edge = ['green' if nodo in camino else 'gray' for nodo in nodos]
        nx.draw(G, pos, node_color=color, edge_color=color_edge, node_size=1700, node_shape='H', with_labels=True)
               
        nx.draw_networkx_edge_labels(G, pos, font_size=7, font_family='Calibri', font_color='black')
        plt.axis('on')
        plt.show()

def main():
    
    continuar = "Y"
    while continuar == "Y" or continuar == "y":
        dijsktra = Dijsktra()
        print("\n") 
        print("Bienvenido a la aplicacion de Dijsktra")
        print("#"*50)
        for key in dijsktra.rutas: 
            print(key)
        print("#"*50)
        origen = input("Ingrese el origen: ")
        destino = input("Ingrese el destino: ")

        if origen not in dijsktra.rutas.keys() or destino not in dijsktra.rutas.keys():
            print("*"*40)
            print("El origen o destino que ingreso no existe!!!")
            print("*"*40)
        elif origen == destino:
            print("*"*40)
            print("El origen y destino son iguales!")
            print("*"*40)
        else:
            
            dijsktra.setOrigen(origen)
            print("El origen es: ", dijsktra.origen)
            dijsktra.setDestino(destino)
            print("El destino es: ", dijsktra.destino)
            costo, camino = dijsktra.busqueda()
            
            print("*"*40)
            print(f"El mejor camino de {camino[0]} a {camino[-1]} es: ")
            print("El costo del camino es: ", costo)
            print("El camino es: ", camino)
            print("*"*40)
            dijsktra.dibujar(camino)
        continuar = input("Desea continuar? (Y/N): ")

    print("*"*40)
    print("Gracias por usar la aplicacion!")
    print("*"*40)

    
if __name__ == '__main__':
    main()
    