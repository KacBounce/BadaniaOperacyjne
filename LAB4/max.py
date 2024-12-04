from MaxFlow import MaxFlow
from Point import Point

def run_min():

    data = open("C:\\Users\\kbednarski\\OneDrive - Anegis\\Desktop\\Badania operacyne\\LAB4\\data.txt", "r")

    number_of_points = int(data.readline())

    data_array = []

    for i in range(number_of_points):
        z = data.readline().replace("\n", "").split("  ")
        for i in range(len(z)):
            z[i] = int(z[i])
        data_array.append(z)

    print(data_array)

    x = MaxFlow(number_of_points, data_array)

    x.PrepareConnections()

    for i in x.points:
        if len(i.successors) == 0:
            i.ES = 0
            i.EF = 0
        print(f"{i.value} successor : {i.successors}")

    z = x.FindShortestPath()

    short_path = z[-1][len(z[-1]) - 1]

    print(short_path, end=" Max Flow : ")

    flow = []
    for i in short_path:
        f = x.points[i].value
        flow.append(f)
    
    print(min(flow))

    mf, fg, fp = x.fordFulkerson(data_array, 0, number_of_points - 1)

    print(f"Max flow : {mf}")
    print("Flow graphs : ")
    for i in range(len(fg)):
        print(f"{i + 1} {fg[i]}")
    print("Flow paths : ")
    for i in range(len(fp)):
        print(f"{fp[i]}")

    


run_min()