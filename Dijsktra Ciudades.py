def dijsktra(ruta, inicio, destino):

    camino_corto = {inicio: (None, 0)}
    ubicacion = inicio
    ya_Recorrido = set()
    
    while ubicacion != destino:
        # Adición de la ubicación actual al conjunto de ubicaciones que ya se han visitado.
        ya_Recorrido.add(ubicacion)
        destinos = ruta.get(ubicacion, [])
        # Itera sobre todos los nodos que están conectados al nodo actual. comprueba si ya ha sido visitado
        # para saltarlo. Si no, calcula el nuevo peso de la ruta a ese nodo 
        for siguiente_nodo, weight in destinos:
            if siguiente_nodo in ya_Recorrido:
                continue
            nueva_distancia = camino_corto[ubicacion][1] + weight
            if siguiente_nodo not in camino_corto or nueva_distancia < camino_corto[inicio][1]:
                camino_corto[siguiente_nodo] = (ubicacion, nueva_distancia)

        # Creación de un diccionario con las ubicaciones que no están en el conjunto `ya_Recorrido`.
        siguiente_ubicacion = {nodo: camino_corto[nodo] for nodo in camino_corto if nodo not in ya_Recorrido}
        if not siguiente_ubicacion:
            return "No se puede hacer esta ruta! :("
        #Seleccionar el nodo con menor peso
        ubicacion = min(siguiente_ubicacion, key=lambda k: siguiente_ubicacion[k][1])

    # Guardar el camino mas corto
    mejor_camino = []
    while ubicacion is not None:
        mejor_camino.append(ubicacion)
        siguiente_nodo = camino_corto[ubicacion][0]
        ubicacion = siguiente_nodo
       
    # Invertir la lista.
    mejor_camino = mejor_camino[::-1]
    print("Distancia: ",str(camino_corto[destino][1]))
    return mejor_camino

def main():
    ruta = {}
    ruta["Arad"] = [("Zerind",75), ("Sibiu",140), ("Timisoara",118)]
    ruta["Zerind"] = [("Arad",75), ("Oradea",71)]
    ruta["Oradea"] = [("Zerind",71), ("Sibiu",151)]
    ruta["Timisoara"] = [("Arad",118), ("Lugoj",111)]
    ruta["Lugoj"] = [("Timisoara",111), ("Mehadia",70)]
    ruta["Mehadia"] = [("Lugoj",70), ("Dobreta",75)]
    ruta["Dobreta"] = [("Mehadia",75), ("Craicova",120)]
    ruta["Craicova"] = [("Dobreta",120), ("Rimnicu Vilcea",146), ("Pitesti",138)]
    ruta["Rimnicu Vilcea"] = [("Sibiu",80), ("Pitesti",97), ("Craicova",146)]
    ruta["Sibiu"] = [("Arad",140), ("Oradea",151), ("Fagaras",99), ("Rimnicu Vilcea",80)]
    ruta["Fagaras"] = [("Sibiu",99), ("Bucharest",211)]
    ruta["Pitesti"] = [("Rimnicu Vilcea",97), ("Bucharest",101), ("Craicova",138)]
    ruta["Bucharest"] = [("Pitesti",101), ("Fagaras",211), ("Urziceni",85), ("Giurgiu",90)]
    ruta["Giurgiu"] = [("Bucharest",90)]
    ruta["Urziceni"] = [("Bucharest",85), ("Hirsova",98), ("Vaslui",142)]
    ruta["Hirsova"] = [("Urziceni",98), ("Eforie",86)]
    ruta["Eforie"] = [("Hirsova",86)]
    ruta["Vaslui"] = [("Urziceni",142), ("Iasi",92)]
    ruta["Iasi"] = [("Vaslui",92), ("Neamt",87)]
    ruta["Neamt"] = [("Iasi",87)]
    continuar = "Y"
    while continuar == "Y" or continuar == "y":
        print("\n") 
        print("Bienvenido a la aplicacion de Dijsktra")
        print("#"*100)
        print(sorted(ruta.keys()))
        print("#"*100)
        
        inicio = input("Ingrese el nombre de la ciudad de origen: ")
        destino = input("Ingrese el nombre de la ciudad de destino: ")
        

        if inicio not in ruta or destino not in ruta:
            print("*"*40)
            print("La ciudad no existe!")
            print("*"*40)
            continue
        elif inicio == destino:
            print("-"*60)
            print("La ciudad de origen es igual a la ciudad de destino!")
            print("-"*60)
        else:
            camino = dijsktra(ruta, inicio, destino)
            print("="*80)
            print("El camino mas corto es: ", camino)
            print("\n")
            print("="*80)
                      
        continuar = input("Desea continuar? (Y/N) ")
        
       

if __name__ == "__main__":
    main()
  

    

   