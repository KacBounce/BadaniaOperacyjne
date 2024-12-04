from Point import Point
from MinPath import MinPath
import heapq
from collections import deque

class MaxFlow(MinPath):

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
                    self.points[j].predecessors.append(i)

    def bfs(self, rGraph, s, t, parent):
        visited = [False] * self.point_num
        q = deque()
        q.append(s)
        visited[s] = True
        parent[s] = -1

        while q:
            u = q.popleft()
            for v in range(self.point_num):
                if not visited[v] and rGraph[u][v] > 0:
                    q.append(v)
                    parent[v] = u
                    visited[v] = True

        return visited[t]

    def fordFulkerson(self, graph, s, t):
        rGraph = [[0] * self.point_num for _ in range(self.point_num)]
        flowGraph = [[0] * self.point_num for _ in range(self.point_num)]  # To track the flow
        flowPaths = []  # To store flow paths and their contributions

        for u in range(self.point_num):
            for v in range(self.point_num):
                rGraph[u][v] = graph[u][v]

        parent = [-1] * self.point_num
        max_flow = 0

        while self.bfs(rGraph, s, t, parent):
            path_flow = float('inf')
            v = t
            path = []

            while v != s:
                u = parent[v]
                path_flow = min(path_flow, rGraph[u][v])
                path.append(v)
                v = u
            path.append(s)

            v = t
            while v != s:
                u = parent[v]
                rGraph[u][v] -= path_flow
                rGraph[v][u] += path_flow
                flowGraph[u][v] += path_flow
                v = u

            max_flow += path_flow
            flowPaths.append((path[::-1], path_flow))  # Store path and its flow

        return max_flow, flowGraph, flowPaths
