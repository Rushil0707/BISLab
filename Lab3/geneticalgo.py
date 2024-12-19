import random
import numpy as np

def fitness_function(x):
    return x**2

population_size = 10
mutation_rate = 0.01
crossover_rate = 0.8
num_generations = 10
gene_length = 10

def create_population(size, gene_length):
    return [np.random.randint(0, 2, gene_length).tolist() for _ in range(size)]

def binary_to_decimal(binary):
    binary_str = ''.join(str(bit) for bit in binary)
    return int(binary_str, 2) / ((2**gene_length) - 1) * 10 - 5

def evaluate_population(population):
    return [fitness_function(binary_to_decimal(individual)) for individual in population]

def select(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probs = [fitness / total_fitness for fitness in fitness_scores]
    return population[np.random.choice(range(len(population)), p=selection_probs)]

def mutate(individual):
    for i in range(gene_length):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        point = random.randint(1, gene_length - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1, parent2

def genetic_algorithm():
    population = create_population(population_size, gene_length)

    for generation in range(num_generations):
        fitness_scores = evaluate_population(population)
        best_fitness = max(fitness_scores)
        best_individual = population[fitness_scores.index(best_fitness)]
        
        print(f"Generation {generation}: Best Fitness = {best_fitness:.4f}")

        new_population = []
        while len(new_population) < population_size:
            parent1 = select(population, fitness_scores)
            parent2 = select(population, fitness_scores)
            offspring = crossover(parent1, parent2)
            new_population.extend([mutate(child) for child in offspring])

        population = new_population[:population_size]
        best_fitness = max(fitness_scores)
        best_individual = population[fitness_scores.index(best_fitness)]
    
    best_solution = binary_to_decimal(best_individual)
    print(f"Best Solution: {best_solution}")

# Run the genetic algorithm
genetic_algorithm()
