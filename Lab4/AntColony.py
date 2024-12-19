import numpy as np
import random

# Parameters
num_ants = 10
num_iterations = 100
alpha = 1  # Importance of pheromone
beta = 2   # Importance of heuristic information
evaporation_rate = 0.5
pheromone_constant = 1.0
num_nodes = 20  # Number of nodes or elements in the problem space

# Initialize pheromone matrix to zero
pheromone_matrix = np.zeros((num_nodes, num_nodes))  # Explicitly set to 0

# Heuristic information (problem-specific, this should be adapted)
def heuristic_info(i, j):
    # Placeholder heuristic: in a real problem, replace this with problem-specific values
    return 1.0 / (abs(i - j) + 1e-10)

# Initialize ants' paths and lengths
def initialize_ants(num_ants, num_nodes):
    return [random.sample(range(num_nodes), num_nodes) for _ in range(num_ants)]

# Evaluate the fitness of a path (problem-specific)
def fitness_function(path):
    # Placeholder fitness function: in a real problem, define the cost/fitness of a path
    return sum(abs(path[i] - path[i+1]) for i in range(len(path) - 1))

# Update pheromones
def update_pheromones(pheromone_matrix, ants, fitnesses):
    # Evaporate pheromones
    pheromone_matrix *= (1 - evaporation_rate)
    
    # Deposit new pheromones based on the fitness of each ant's path
    for ant, fitness in zip(ants, fitnesses):
        for i in range(len(ant) - 1):
            pheromone_matrix[ant[i]][ant[i+1]] += pheromone_constant / (fitness + 1e-10)

# Ant decision rule: choose the next node based on pheromone and heuristic
def choose_next_node(current_node, visited, pheromone_matrix, alpha, beta):
    probabilities = []
    for next_node in range(num_nodes):
        if next_node not in visited:
            pheromone = (pheromone_matrix[current_node][next_node] + 1e-10) ** alpha
            heuristic = heuristic_info(current_node, next_node) ** beta
            probabilities.append((next_node, pheromone * heuristic))
        else:
            probabilities.append((next_node, 0))
    
    total = sum(prob for _, prob in probabilities)
    probabilities = [(node, prob / total) if total > 0 else (node, 0) for node, prob in probabilities]
    chosen_node = random.choices([node for node, _ in probabilities], weights=[prob for _, prob in probabilities])[0]
    return chosen_node

# Main ACO function
def ant_colony_optimization():
    best_path = None
    best_fitness = float('inf')
    
    for iteration in range(num_iterations):
        # Generate paths for all ants
        ants = []
        for _ in range(num_ants):
            path = []
            visited = set()
            current_node = random.randint(0, num_nodes - 1)
            path.append(current_node)
            visited.add(current_node)
            
            while len(path) < num_nodes:
                next_node = choose_next_node(current_node, visited, pheromone_matrix, alpha, beta)
                path.append(next_node)
                visited.add(next_node)
                current_node = next_node
            
            ants.append(path)
        
        # Evaluate paths and update best solution
        fitnesses = [fitness_function(path) for path in ants]
        for path, fitness in zip(ants, fitnesses):
            if fitness < best_fitness:
                best_fitness = fitness
                best_path = path
        
        # Update pheromones
        update_pheromones(pheromone_matrix, ants, fitnesses)
        
        # Print iteration details
        print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")
    
    print(f"Best Path: {best_path}")
    print(f"Best Fitness: {best_fitness}")

# Run the ACO algorithm
ant_colony_optimization()

