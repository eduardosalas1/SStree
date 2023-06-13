import numpy as np
from scipy.spatial import distance
import pickle
import os
from queue import PriorityQueue
from scipy.spatial import distance
import sys

class SSnode:
    # Inicializa un nodo de SS-Tree
    def __init__(self, leaf=False, points=None, children=None, data=None, parent=None):
        self.leaf     = leaf
        self.points   = points if points is not None else []
        self.children = children if children is not None else []
        self.data     = data if data is not None else [] 
        self.centroid = np.mean(np.array([p for p in self.points]), axis=0) if self.points else None
        self.radius   = self.compute_radius() if points is not None else 0.0 
        self.parent   = parent

    # Calcula el radio del nodo como la máxima distancia entre el centroide y los puntos contenidos en el nodo
    def compute_radius(self):

        if self.leaf:

            self.radius = np.max(np.array([distance.euclidean(self.centroid,p) for p in self.points]))           

        else:

            self.radius = np.max(np.array([distance.euclidean(self.centroid,child.centroid)+ child.radius for child in self.children]))

        return 


    # Verifica si un punto dado está dentro del radio del nodo
    def intersects_point(self, point):
        if distance.euclidean(self.centroid,point) <= self.radius:
            return True
        
        return False

    # Actualiza el envolvente delimitador del nodo recalculando el centroide y el radio
    def update_bounding_envelope(self):
        
        puntos = np.array(self.get_entries_centroids())

        self.centroid = np.mean(puntos)

        self.compute_radius()



    # Encuentra y devuelve el hijo más cercano al punto objetivo
    # Se usa para entrar el nodo correcto para insertar un nuevo punto
    def find_closest_child(self, target):
        # Completar aqui!
        nodo = self.children[0]
        minimo = float("inf")
        for child in self.children:
            distMin = distance.euclidean(child.centroid,target)
            if distMin < minimo:
                nodo = child
                minimo = distMin

        return nodo



    # Divide el nodo en dos a lo largo del eje de máxima varianza
    def split(self, m):

        splitIdx = self.find_split_index(m)

        node1 = None
        node2 = None

        if self.leaf:
            
            node1 = SSnode(leaf = True, points = self.points[:splitIdx])
            node2 = SSnode(leaf = True, points = self.points[splitIdx:])

        else:

            node1 = SSnode(children = self.children[:splitIdx])
            node2 = SSnode(children = self.children[splitIdx:])

        return node1,node2




    # Encuentra el índice en el que dividir el nodo para minimizar la varianza total
    def find_split_index(self,m):

        coordinateidx = self.direction_of_max_variance()
        
        self.sort_by_coordinate(coordinateidx)
        
        puntos = np.array([point[coordinateidx] for point in self.get_entries_centroids()])

        return self.min_variance_split(puntos,m)

    def sort_by_coordinate(self,idx):

        if self.leaf:
            sorted_indices = np.argsort(np.array(self.points[:,idx]))
            self.points = np.array(self.points[sorted_indices])
            self.data = self.data[sorted_indices]

        else:

            self.children = np.array(self.children.sort(key=lambda node:node.centroid[idx]))


    # Encuentra la división que minimiza la varianza total
    def min_variance_split(self, values, m):
        
        minVar = float("inf")

        splitidx = m

        for i in range(m, len(values)-m):
            
            sumvar = np.var(values[:i]) + np.var(values[i:]) 
            
            if sumvar < minVar:
                minVar = sumvar
                splitidx = i

        return splitidx


    # Encuentra el eje a lo largo del cual los puntos tienen la máxima varianza
    def direction_of_max_variance(self):
        
        varx = 0.0
        vary = 0.0

        if self.leaf:
            varx = np.var(np.array([p[0] for p in self.points]))
            vary = np.var(np.array([p[1] for p in self.points]))
        else:
            varx = np.var(np.array([child.centroid[0] for child in self.children]))
            vary = np.var(np.array([child.centroid[1] for child in self.children]))


        if varx >= vary:
            return 0
        
        return 1

    # Obtiene los centroides de las entradas del nodo
    def get_entries_centroids(self):
        
        if self.leaf:
            return self.points
        
        return np.array([child.centroid for child in self.children])
        

class SSTree:
    # Inicializa un SS-Tree
    def __init__(self, M=None, m=None, filename=None):
        if filename is None:
            self.M = M
            self.m = m
            if M is not None and m is not None:
                self.root = SSnode(leaf=True)
            else:
                self.root = None
        else:
            if os.path.exists(filename):
                loaded_tree = self.load(filename)
                self.M = loaded_tree.M
                self.m = loaded_tree.m
                self.root = loaded_tree.root
            else:
                print(f"'{filename}' no existe.")
                self.M = None
                self.m = None
                self.root = None

    # Inserta un punto en el árbol
    def insert(self, point, data=None):

        newChild1, newChild2 = self.insertR(self.root, point, data)
        if newChild1 is not None:
            self.root = SSnode(leaf = False, children = np.array([newChild1,newChild2]))

        
    
    def insertR(self,node,point,data=None):

        if node.leaf:
            if point in node.points:
                return None,None
            node.points = np.append(np.array(node.points), point)
            node.data = np.append(np.array(node.data), data)
            node.update_bounding_envelope()

            if len(node.points) <= self.M:
                return None,None

        else:

            closestChild = node.find_closest_child(point)
            newChild1, newChild2 = self.insert1(closestChild, point,data)
            if newChild1 == None:
                node.update_bounding_envelope()
                return None,None
            else:
                node.children = np.array(node.children)
                index = np.where(node.children == closestChild)[0]
                node.children = np.delete(node.children,index)
                node.children = np.append(np.array(node.children), newChild1)
                node.children = np.append(np.array(node.children), newChild2)

                node.update_bounding_envelope()

                if len(node.children) <= self.M:
                    return None,None
                
        return node.split(self.m)


    # Busca un punto en el árbol y devuelve el nodo que lo contiene si existe
    def search(self, target):
        return self._search(self.root, target)

    # Función recursiva de ayuda para buscar un punto en el árbol
    def _search(self, node, target):
        if node.leaf:
            return node if target in node.points else None
        else:
            for child in node.children:
                if child.intersects_point(target):
                    result = self._search(child, target)
                    if result is not None:
                        return result
        return None
    


    # Depth-First K-Nearest Neighbor Algorithm
    def knn(self, q, k=3):
        # Completar aqui!


        pass

    def priorityQueue():

        pass

    # Guarda el árbol en un archivo
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    # Carga un árbol desde un archivo
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)