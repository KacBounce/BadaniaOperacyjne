from Point import Point

class Graph():
    def __init__(self, point_num, connection_num, point_weights, connections):
        self.point_num = int(point_num) 
        self.connection_num = int(connection_num) 
        self.point_weights = [int(w) for w in point_weights] 
        self.connections = connections  
        self.points = []  


    def GetAllPoints(self):
        for weight in self.point_weights:
            x = Point(int(weight))
            self.points.append(x)

    def PrepareConnections(self):
        new_connections = []
        for conn in self.connections:
            new_connections.append([int(x) - 1 for x in conn.split(" ")])
        self.connections = new_connections

    def GetAllPredecessors(self):
        for i in self.connections:
            self.points[i[1]].predecessors.append(i[0])  # i[1] -> current point, i[0] -> predecessor

    def GetAllSuccessors(self):
        for i in self.connections:
            self.points[i[0]].successors.append(i[1])  # i[0] -> current point, i[1] -> successor

    def ForwardPass(self):
        # Initial setup for ES and EF
        for i in range(self.point_num):
            point = self.points[i]
            point.ES = 0 if point.predecessors == [] else float('-inf')  # If no predecessors, ES = 0
            point.EF = 0  # Initialize EF

        # Iterate until no changes are made
        while True:
            updated = False
            for i in range(self.point_num):
                point = self.points[i]
                # Update ES and EF based on predecessors
                if point.predecessors: 
                    new_es = max(self.points[p].EF for p in point.predecessors)
                    if new_es > point.ES:
                        point.ES = new_es
                        updated = True
                rounded_value = round(point.ES + point.value, 5)
                point.EF = rounded_value
            if not updated:
                break  # Exit when no updates occur

    def BackwardPass(self):
        # Initialize LS and LF
        max_ef = max(point.EF for point in self.points)  # Find the maximum EF from the forward pass
        for point in self.points:
            point.LF = max_ef  # Initialize LF to maximum EF
            rounded_value = round(point.LF - point.value, 5)
            point.LS = rounded_value

        # Iterate until no changes are made
        while True:
            updated = False
            for i in range(self.point_num):
                point = self.points[i]
                for successor in point.successors:
                    if self.points[successor].LS < point.LF - point.value:  # If a new LS is found
                        self.points[successor].LS = point.LF - point.value
                        updated = True
                    # Update LF based on successors
                    if point.LF > self.points[successor].LS:
                        point.LF = self.points[successor].LS
                point.LS = round(point.LF - point.value, 5)
            if not updated:
                break  # Exit when no updates occur

    def PrintCPMParameters(self):
        for i in range(self.point_num):
            point = self.points[i]
            print(f"Point {i + 1} (Duration: {point.value}): ES = {point.ES}, EF = {point.EF}, "
                  f"LS = {point.LS}, LF = {point.LF}")

    def FindCriticalPath(self):
        critical_path = []
        for i in range(self.point_num):
            point = self.points[i]
            if point.ES == point.LS:  # Critical path condition
                critical_path.append(i)

        # Sort critical path based on successors
        sorted_path = self.SortCriticalPath(critical_path)
        return sorted_path

    def SortCriticalPath(self, critical_path):
        """ Sorts the critical path based on successors. """
        # Find the starting point (with no predecessors)
        start_points = [i for i in critical_path if not self.points[i].predecessors]
        sorted_path = []

        def traverse(point_index):
            """ Recursive function to traverse the path. """
            sorted_path.append(point_index + 1)  # Store 1-indexed point
            for successor in self.points[point_index].successors:
                if successor in critical_path and (successor + 1) not in sorted_path:
                    traverse(successor)

        # Start from each starting point found
        for start in start_points:
            traverse(start)

        return sorted_path