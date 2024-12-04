from MinPath import MinPath
from Point import Point

def run_min():

    data = open("C:\\Users\\kbednarski\\OneDrive - Anegis\\Desktop\\Badania operacyne\\LAB3\\data.txt", "r")

    number_of_points = int(data.readline())

    data_array = []

    for i in range(number_of_points):
        z = data.readline().replace("\n", "").split("  ")
        data_array.append(z)

    print(data_array)

    x = MinPath(number_of_points, data_array)

    x.PrepareConnections()

    for i in x.points:
        if len(i.successors) == 0:
            i.ES = 0
            i.EF = 0
        print(f"{i.value} successor : {i.successors}")

    z = x.FindShortestPath()

    print(z)

run_min()