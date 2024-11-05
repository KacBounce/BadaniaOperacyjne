from Graph import Graph
from Point import Point
import numpy as np
import random


class PertGraph(Graph):

    def __init__(self, point_num, connection_num, point_weights, connections):
        self.point_num = int(point_num) 
        self.connection_num = int(connection_num) 
        self.point_weights = point_weights
        self.connections = connections  
        self.points = []  
        self.total_variance = 0

    def GetAllPoints(self):
         random.seed()
         for weight in self.point_weights:
            weight = weight.split(" ")
            weight = [int(weight[x]) for x in range(len(weight))]
            random_value = random.triangular(weight[0], weight[1], weight[2])
            rounded_value = round(random_value, 5)
            x = Point(rounded_value, weight)
            self.points.append(x)
    
    def GetVarianceForEachPoint(self):
        for point in self.points:
            point.variance = round(pow(((point.all_values[2] - point.all_values[0])/6), 2), 2)
    
    def FindCriticalPath(self):
        critical_path = []
        for i in range(self.point_num):
            point = self.points[i]
            if point.ES == point.LS:  # Critical path condition
                critical_path.append(i)
                self.total_variance += point.variance
                

        # Sort critical path based on successors
        sorted_path = self.SortCriticalPath(critical_path)
        return sorted_path
    
    def GetStandardDeviation(self):
        return round(np.sqrt(self.total_variance), 2)