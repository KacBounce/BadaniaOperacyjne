import random
import math
import matplotlib.pyplot as plt

# Generate random cities in 2D space
def generate_cities(num_cities, width=100, height=100):
    return [(int(random.uniform(0, width)), int(random.uniform(0, height))) for _ in range(num_cities)]

# Calculate Euclidean distance between two points
def euclidean_distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return round((dx ** 2 + dy ** 2) ** 0.5)

# Nearest Neighbour Algorithm
def nearest_neighbour(cities):
    unvisited = cities[:]
    path = [unvisited.pop(0)]  # Start from the first city
    while unvisited:
        current_city = path[-1]
        next_city = min(unvisited, key=lambda city: euclidean_distance(current_city, city))
        path.append(next_city)
        unvisited.remove(next_city)
    return path

def furthest_insertion(cities):
    if len(cities) < 2:
        return cities, 0  # Return the cities and length as 0 if fewer than two cities

    unvisited = cities[:]
    
    # Step 1: Start with the two cities that are farthest apart
    max_dist = -1
    start, end = 0, 1
    for i in range(len(unvisited)):
        for j in range(i + 1, len(unvisited)):
            dist = euclidean_distance(unvisited[i], unvisited[j])
            if dist > max_dist:
                max_dist = dist
                start, end = i, j

    # Initialize path with the two farthest cities
    path = [unvisited[start], unvisited[end]]
    del unvisited[end]  # Remove in reverse order to avoid indexing issues
    del unvisited[start]

    # Step 2: Add cities based on furthest distance from the current path
    while unvisited:
        # Find the city furthest from the current path
        furthest_city = None
        max_dist_to_path = -1
        for city in unvisited:
            dist_to_path = min(euclidean_distance(city, p) for p in path)
            if dist_to_path > max_dist_to_path:
                max_dist_to_path = dist_to_path
                furthest_city = city

        # Find the best position to insert the furthest city in the current path
        best_position = None
        best_cost = float('inf')
        for i in range(len(path)):
            prev_city = path[i - 1] if i > 0 else path[-1]
            cost = (
                euclidean_distance(prev_city, furthest_city) +
                euclidean_distance(furthest_city, path[i]) -
                euclidean_distance(prev_city, path[i])
            )
            if cost < best_cost:
                best_cost = cost
                best_position = i
        
        # Insert the furthest city at the best position
        path.insert(best_position, furthest_city)
        unvisited.remove(furthest_city)

    return path


# Calculate total path length
def path_length(path):
    return sum(euclidean_distance(path[i], path[(i + 1) % len(path)]) for i in range(len(path)))

# Visualize paths
def visualize_path(cities, path, title):
    x, y = zip(*cities)
    px, py = zip(*(path + [path[0]]))  # Close the path
    plt.scatter(x, y, color='red', label='Cities')
    plt.plot(px, py, label='Path')
    plt.title(title)
    plt.legend()
    plt.show()

# Test environment
def test_algorithms(num_cities=10, width=100, height=100):
    #cities = generate_cities(num_cities, width, height)

    data = "37 52   49 49   52 64   20 26   40 30   21 47   17 63   31 62   52 33   51 21   42 41   31 32    5 25   12 42   36 16   52 41   27 23   17 33   13 13   57 58   62 42   42 57   16 57    8 52    7 38   27 68   30 48   43 67   58 48   58 27   37 69   38 46   46 10   61 33   62 63   63 69   32 22   45 35   59 15    5  6   10 17   21 10    5 64   30 15   39 10   32 39   25 32   25 55   48 28   56 37   30 40"

    # Split the string into a list of integers
    numbers = list(map(int, data.split()))

    # Create a list of tuples by pairing consecutive numbers
    cities = [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers), 2)]
    
    # Run Nearest Neighbour
    nn_path = nearest_neighbour(cities)
    nn_length = path_length(nn_path)
    
    # Run Insertion Algorithm
    ia_path = furthest_insertion(cities)
    ia_length = path_length(ia_path)
    
    # Visualize and compare
    print(f"Nearest Neighbour Path Length: {nn_length:.2f}\nPath : {nn_path}")
    visualize_path(cities, nn_path, "Nearest Neighbour Algorithm")
    
    print(f"Insertion Algorithm Path Length: {ia_length:.2f}\nPath : {len(ia_path)}")
    visualize_path(cities, ia_path, "Insertion Algorithm")
    

    # Compare Results
    if nn_length < ia_length:
        print("Nearest Neighbour performed better.")
    elif ia_length < nn_length:
        print("Insertion Algorithm performed better.")
    else:
        print("Both algorithms performed equally well.")

    to_path_nn = two_opt(nn_path)
    to_path_ia = two_opt(ia_path)

    to_path_nn_length = path_length(to_path_nn)

    to_path_ia_length = path_length(to_path_ia)

    # Visualize and compare
    print(f"Nearest Neighbour Path Length (after two opt): {to_path_nn_length:.2f}\nPath : {to_path_nn}")
    visualize_path(cities, to_path_nn, "Nearest Neighbour Algorithm")
    
    print(f"Insertion Algorithm Path Length: {to_path_ia_length:.2f}\nPath : {to_path_ia}")
    visualize_path(cities, to_path_ia, "Insertion Algorithm")

    print(f"Poprawa NN : {(nn_length/to_path_nn_length - 1) * 100:.2f}%\nPoprawa INS : {(ia_length/to_path_ia_length - 1) * 100:.2f}%\nPO TWO OPT")



def two_opt(path):
    n = len(path)
    improved = True

    while improved:
        improved = False

        for i in range(n - 1):
            for j in range(i + 2, n):  # Only consider valid non-overlapping segments
                # Calculate the cost before and after reversing the segment
                old_distance = (
                    euclidean_distance(path[i], path[i + 1]) +
                    euclidean_distance(path[j], path[(j + 1) % n])
                )
                new_distance = (
                    euclidean_distance(path[i], path[j]) +
                    euclidean_distance(path[i + 1], path[(j + 1) % n])
                )

                # If reversing the segment reduces the total path distance
                if new_distance < old_distance:
                    # Reverse the segment
                    path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                    improved = True
 
    return path


def simulated_annealing(cities, initial_temperature=100000, cooling_rate=0.99999, stopping_temperature=1):
    # Initialize with a random solution
    current_path = cities[:]
    random.shuffle(current_path)
    current_length = path_length(current_path)
    
    best_path = current_path[:]
    best_length = current_length
    
    temperature = initial_temperature
    
    while temperature > stopping_temperature:
        # Generate a new candidate path by swapping two cities
        new_path = current_path[:]
        i, j = random.sample(range(len(new_path)), 2)
        new_path[i], new_path[j] = new_path[j], new_path[i]
        
        new_length = path_length(new_path)
        
        # Decide whether to accept the new solution
        if new_length < current_length or random.uniform(0, 1) < math.exp((current_length - new_length) / temperature):
            current_path = new_path
            current_length = new_length
            
            # Update the best solution if the new one is better
            if current_length < best_length:
                best_path = current_path
                best_length = current_length
        
        # Cool down the temperature
        temperature *= cooling_rate
    
    return best_path, best_length

def tabu_search(cities, initial_solution, max_iterations=100, tabu_size=10):
    def swap_edges(path, i, j):
        """Perform a 2-opt swap."""
        return path[:i] + list(reversed(path[i:j+1])) + path[j+1:]

    def calculate_path_length(path):
        """Calculate the total path length."""
        return sum(euclidean_distance(path[i], path[(i + 1) % len(path)]) for i in range(len(path)))

    # Initialize variables
    current_solution = initial_solution
    best_solution = current_solution[:]
    best_distance = calculate_path_length(best_solution)
    tabu_list = []  # To store tabu moves (i, j swaps)
    tabu_tenures = {}  # Optional: Use tenures to expire moves

    for iteration in range(max_iterations):
        neighborhood = []  # Generate neighbors
        for i in range(len(current_solution) - 1):
            for j in range(i + 1, len(current_solution)):
                if i != j:
                    new_solution = swap_edges(current_solution, i, j)
                    distance = calculate_path_length(new_solution)
                    neighborhood.append((new_solution, distance, (i, j)))

        # Find the best move
        neighborhood = sorted(neighborhood, key=lambda x: x[1])  # Sort by distance
        for neighbor in neighborhood:
            new_solution, new_distance, move = neighbor
            # Check tabu list
            if move not in tabu_list or new_distance < best_distance:
                # Aspiration criterion: Accept if it's better than the best solution
                current_solution = new_solution
                current_distance = new_distance
                tabu_list.append(move)  # Add move to tabu list
                if len(tabu_list) > tabu_size:  # Keep the tabu list within size limit
                    tabu_list.pop(0)
                break  # Only accept one move per iteration

        # Update the best solution
        if current_distance < best_distance:
            best_solution = current_solution[:]
            best_distance = current_distance

        # Optional: Print iteration progress
        print(f"Iteration {iteration + 1}, Best Distance: {best_distance:.2f}")

    return best_solution, best_distance

# Integrating Simulated Annealing into the test environment
def test_with_simulated_annealing(num_cities=10, width=100, height=100):
    # Generate cities or use predefined ones
    data = "37 52   49 49   52 64   20 26   40 30   21 47   17 63   31 62   52 33   51 21   42 41   31 32    5 25   12 42   36 16   52 41   27 23   17 33   13 13   57 58   62 42   42 57   16 57    8 52    7 38   27 68   30 48   43 67   58 48   58 27   37 69   38 46   46 10   61 33   62 63   63 69   32 22   45 35   59 15    5  6   10 17   21 10    5 64   30 15   39 10   32 39   25 32   25 55   48 28   56 37   30 40"
    numbers = list(map(int, data.split()))
    cities = [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers), 2)]
    
    # Nearest Neighbour for initial comparison
    nn_path = nearest_neighbour(cities)
    nn_length = path_length(nn_path)
    visualize_path(cities, nn_path, "Nearest Neighbour Algorithm")
    print(f"Nearest Neighbour Path Length: {nn_length:.2f}")
    
    # Simulated Annealing
    sa_path, sa_length = simulated_annealing(cities)
    visualize_path(cities, sa_path, "Simulated Annealing Algorithm")
    print(f"Simulated Annealing Path Length: {sa_length:.2f}")
    
    # Comparison
    improvement = (nn_length - sa_length) / nn_length * 100
    print(f"Improvement with Simulated Annealing: {improvement:.2f}%")


def test_all_algorithms(num_cities=10, width=100, height=100):
    # Define or generate cities
    data = "37 52   49 49   52 64   20 26   40 30   21 47   17 63   31 62   52 33   51 21   42 41   31 32    5 25   12 42   36 16   52 41   27 23   17 33   13 13   57 58   62 42   42 57   16 57    8 52    7 38   27 68   30 48   43 67   58 48   58 27   37 69   38 46   46 10   61 33   62 63   63 69   32 22   45 35   59 15    5  6   10 17   21 10    5 64   30 15   39 10   32 39   25 32   25 55   48 28   56 37   30 40"
    numbers = list(map(int, data.split()))
    cities = [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers), 2)]

    # Run Nearest Neighbour Algorithm
    nn_path = nearest_neighbour(cities)
    nn_length = path_length(nn_path)
    print(f"Nearest Neighbour Path Length: {nn_length:.2f}")
    visualize_path(cities, nn_path, "Nearest Neighbour Algorithm")
    
    # Run Furthest Insertion Algorithm
    fi_path = furthest_insertion(cities)
    fi_length = path_length(fi_path)
    print(f"Furthest Insertion Path Length: {fi_length:.2f}")
    visualize_path(cities, fi_path, "Furthest Insertion Algorithm")

    # Optimize with Two-Opt
    nn_optimized = two_opt(nn_path)
    nn_optimized_length = path_length(nn_optimized)
    fi_optimized = two_opt(fi_path)
    fi_optimized_length = path_length(fi_optimized)
    print(f"Nearest Neighbour (Two-Opt) Path Length: {nn_optimized_length:.2f}")
    visualize_path(cities, nn_optimized, "Nearest Neighbour with Two-Opt")
    print(f"Furthest Insertion (Two-Opt) Path Length: {fi_optimized_length:.2f}")
    visualize_path(cities, fi_optimized, "Furthest Insertion with Two-Opt")

    # Run Simulated Annealing
    sa_path, sa_length = simulated_annealing(cities)
    print(f"Simulated Annealing Path Length: {sa_length:.2f}")
    visualize_path(cities, sa_path, "Simulated Annealing Algorithm")

    # Run Tabu Search
    initial_solution = random.sample(cities, len(cities))
    ts_path, ts_length = tabu_search(cities, initial_solution)
    print(f"Tabu Search Path Length: {ts_length:.2f}")
    visualize_path(cities, ts_path, "Tabu Search Algorithm")


    # Compare Results
    print("\n--- Comparison of Results ---")
    print(f"Nearest Neighbour Length: {nn_length:.2f}")
    print(f"Furthest Insertion Length: {fi_length:.2f}")
    print(f"Two-Opt on Nearest Neighbour: {nn_optimized_length:.2f}")
    print(f"Two-Opt on Furthest Insertion: {fi_optimized_length:.2f}")
    print(f"Simulated Annealing Length: {sa_length:.2f}")
    print(f"Tabu Search Length: {ts_length:.2f}")

    # Determine the best algorithm
    results = {
        "Nearest Neighbour": nn_length,
        "Furthest Insertion": fi_length,
        "Nearest Neighbour (Two-Opt)": nn_optimized_length,
        "Furthest Insertion (Two-Opt)": fi_optimized_length,
        "Simulated Annealing": sa_length,
        "Tabu Search": ts_length,
    }
    best_algorithm = min(results, key=results.get)
    print(f"\nBest Algorithm: {best_algorithm} with Length: {results[best_algorithm]:.2f}")


# # Run the test environment
# test_with_simulated_annealing()

# test_algorithms(num_cities=100, width=100, height=100)

#test_all_algorithms()



import random
import math
import matplotlib.pyplot as plt
import numpy as np
import time
# Wprowadzone funkcje (Nearest Neighbour, Furthest Insertion, Two-Opt, Simulated Annealing) są już w kodzie

def run_experiment():
    results = {
        'NN': [],
        'FI': [],
        'SA': [],
        '2-Opt-NN': [],
        '2-Opt-FI': []
    }
    
    runtimes = {
        'NN': [],
        'FI': [],
        'SA': [],
        '2-Opt-NN': [],
        '2-Opt-FI': []
    }
    
    city_sizes = list(range(100, 1001, 50))  # Cities from 100 to 1000 with step of 50
    iterations = 5  # Number of iterations for each algorithm
    
    for num_cities in city_sizes:
        nn_avg = []
        fi_avg = []
        sa_avg = []
        nn_opt_avg = []
        fi_opt_avg = []
        
        nn_times = []
        fi_times = []
        sa_times = []
        nn_opt_times = []
        fi_opt_times = []
        
        for _ in range(iterations):
            cities = generate_cities(num_cities)
            
            # Nearest Neighbour
            start_time = time.time()
            nn_path = nearest_neighbour(cities)
            nn_avg.append(path_length(nn_path))
            nn_times.append(time.time() - start_time)
            
            # Furthest Insertion
            start_time = time.time()
            fi_path = furthest_insertion(cities)
            fi_avg.append(path_length(fi_path))
            fi_times.append(time.time() - start_time)
            
            # Simulated Annealing
            start_time = time.time()
            sa_path, sa_length = simulated_annealing(cities)
            sa_avg.append(sa_length)
            sa_times.append(time.time() - start_time)

            # Two-Opt on NN
            start_time = time.time()
            nn_opt_path = two_opt(nn_path)  # Pass an initial path
            nn_opt_avg.append(path_length(nn_opt_path))
            nn_opt_times.append(time.time() - start_time)
            
            # Two-Opt on FI
            start_time = time.time()
            fi_opt_path = two_opt(fi_path)  # Pass an initial path
            fi_opt_avg.append(path_length(fi_opt_path))
            fi_opt_times.append(time.time() - start_time)
        
        # Calculate average for each algorithm
        results['NN'].append(np.mean(nn_avg))
        results['FI'].append(np.mean(fi_avg))
        results['SA'].append(np.mean(sa_avg))
        results['2-Opt-NN'].append(np.mean(nn_opt_avg))
        results['2-Opt-FI'].append(np.mean(fi_opt_avg))
        
        # Store average runtimes
        runtimes['NN'].append(np.mean(nn_times))
        runtimes['FI'].append(np.mean(fi_times))
        runtimes['SA'].append(np.mean(sa_times))
        runtimes['2-Opt-NN'].append(np.mean(nn_opt_times))
        runtimes['2-Opt-FI'].append(np.mean(fi_opt_times))
    
    return results, runtimes, city_sizes


def plot_exp_results(results, runtimes, city_sizes):
    # Plot the path lengths
    plt.figure(figsize=(10, 6))
    plt.plot(city_sizes, results['NN'], label="Nearest Neighbour")
    plt.plot(city_sizes, results['FI'], label="Furthest Insertion")
    plt.plot(city_sizes, results['SA'], label="Simulated Annealing")
    plt.plot(city_sizes, results['2-Opt-NN'], label="Two-Opt-NN")
    plt.plot(city_sizes, results['2-Opt-FI'], label="Two-Opt-FI")
    
    plt.xlabel('Number of Cities')
    plt.ylabel('Average Path Length')
    plt.title('Algorithm Comparison for Different City Sizes')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot the runtime measurements
    plt.figure(figsize=(10, 6))
    plt.plot(city_sizes, runtimes['NN'], label="Nearest Neighbour")
    plt.plot(city_sizes, runtimes['FI'], label="Furthest Insertion")
    plt.plot(city_sizes, runtimes['SA'], label="Simulated Annealing")
    plt.plot(city_sizes, runtimes['2-Opt-NN'], label="Two-Opt-NN")
    plt.plot(city_sizes, runtimes['2-Opt-FI'], label="Two-Opt-FI")
    
    plt.xlabel('Number of Cities')
    plt.ylabel('Average Runtime (Seconds)')
    plt.title('Algorithm Runtime Comparison for Different City Sizes')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the experiment
results, runtimes, city_sizes = run_experiment()
plot_exp_results(results, runtimes, city_sizes)
