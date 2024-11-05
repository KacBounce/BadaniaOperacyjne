from PertGraph import PertGraph
import random
from scipy.stats import norm
from math import sqrt

def z_value(probability):
    # Zwraca wartość Z dla podanego prawdopodobieństwa
    return norm.ppf(probability/100)

def calculate_total_time():
    data = open("C:\\Users\\kacpe\\Downloads\\Compressed\\Badania operacyne\\Badania operacyne\\LAB2\\data.txt", "r")
    x = data.readline()
    weights = data.readline().replace("\n", "").split("   ")
    total = 0
    for i in weights:
        i = i.split(" ")
        i = [int(i[x]) for x in range(len(i))]
        total+=i[1]
    return total

def run_pert():
    data = open(
        "C:\\Users\\kacpe\\Downloads\\Compressed\\Badania operacyne\\Badania operacyne\\LAB2\\data.txt", "r")

    numbers = data.readline().replace("\n", "").split(" ")

    weights = data.readline().replace("\n", "").split("   ")
    #weights = [[random.randint(1,9) for x in range(3)] for x in range(int(numbers[0]))]

    connections = data.readline().replace("\n", "").split("   ")

    g = PertGraph(numbers[0], numbers[1], weights, connections)


    g.GetAllPoints()  
    g.PrepareConnections() 
    g.GetVarianceForEachPoint() 
    g.GetAllPredecessors()  
    g.GetAllSuccessors()  
    g.ForwardPass()  
    g.BackwardPass()  
    g.PrintCPMParameters()  

    # Optional: Find and print the critical path
    critical_path = g.FindCriticalPath()
    deviation = round(sqrt(g.total_variance), 2)
    if (len(critical_path)):
        total_time = g.points[critical_path[len(critical_path) - 1] - 1].EF
    else:
        total_time = -100
    print(f"Critical Path: {critical_path}")
    print(f"Total time : {total_time}")
    print(f"Total variance : {g.total_variance}")
    print(f"Standard deviation : {deviation}")

    return total_time

times = []
samples = 1000
threshold = 15
percentage = 50

i = 0
while (i != samples):
    x = run_pert()
    if (x == -100):
            continue
    else:
        times.append(x)
        i += 1

filtered_arr = [x for x in times if x < threshold]

wynik = len(filtered_arr)/samples

times_sum = 0

for i in times:
    times_sum += i

average_time = times_sum/samples
deviation = 1.05
z = z_value(percentage)
time_to_process = average_time + (z * deviation)

print(f"Szansa ze wykona sie ponizej {threshold} : {wynik}\nSredni czas trwania projektu : {average_time}\nCzas potrzebny na wykonanie w {percentage}% : {time_to_process}")
