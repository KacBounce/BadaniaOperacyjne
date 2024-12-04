from Graph import Graph
from Point import Point
import heapq

class MinPath(Graph):

    def __init__(self, point_num, connections):
        self.point_num = int(point_num)
        self.connections = connections
        self.points = []  

    def PrepareConnections(self):
        for i in range(self.point_num):
            z = Point(0)
            self.points.append(z)
        for i in range(self.point_num):
            for j in range(self.point_num):
                if int(self.connections[i][j]) != 0:
                    self.points[i].value = int(self.connections[i][j])
                    self.points[i].successors.append(j)
                    self.points[j].predecessors.append(i)  # Add predecessor relationship


    def FindShortestPath(self):
        dist = {i: float('inf') for i in range(self.point_num)}  
        dist[0] = 0 
        
        pq = [(0, 0)]
        predecessors = {i: None for i in range(self.point_num)}
        
        while pq:
            current_dist, current_point = heapq.heappop(pq)
            
            if current_dist > dist[current_point]:
                continue
            
            for successor in self.points[current_point].successors:
                weight = self.connections[current_point][successor]
                new_dist = current_dist + int(weight)
                
                if new_dist < dist[successor]:
                    dist[successor] = new_dist
                    predecessors[successor] = current_point
                    heapq.heappush(pq, (new_dist, successor))
        
        paths = {}
        for i in range(self.point_num):
            path = []
            current = i
            while current is not None:
                path.append(current)
                current = predecessors[current]
            if dist[i] == float('inf'):
                paths[i] = [-1] 
            else:
                paths[i] = path[::-1]
        
        return dist, paths
