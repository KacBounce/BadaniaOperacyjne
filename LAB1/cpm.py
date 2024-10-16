from Graph import Graph

data = open("C:\\Users\\kbednarski\\OneDrive - Anegis\\Desktop\\Badania operacyne\\LAB1\\data.txt", "r")

numbers = data.readline().replace("\n", "").split(" ")

weights = data.readline().replace("\n", "").split(" ")

connections = data.readline().replace("\n", "").split("  ")

g = Graph(numbers[0], numbers[1], weights, connections)

g.GetAllPoints()  
g.PrepareConnections()  
g.GetAllPredecessors()  
g.GetAllSuccessors()  
g.ForwardPass()  
g.BackwardPass()  
g.PrintCPMParameters()  

# Optional: Find and print the critical path
critical_path = g.FindCriticalPath()
print(f"Critical Path: {critical_path}")
print(f"Total time : {g.points[critical_path[len(critical_path) - 1] - 1].EF}")