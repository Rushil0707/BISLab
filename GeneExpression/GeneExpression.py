

import numpy as np

# Define the target function
def target_function(x):
    return x**2  # The function to optimize

# Define the fitness function
def fitness_function(expression, target, x_value):
    """
    Evaluate the fitness of an expression.
    :param expression: The candidate solution (as a string).
    :param target: Target output to achieve (e.g., x^2).
    :param x_value: The value of x to plug into the function.
    :return: Fitness value (higher is better).
    """
    try:
        result = eval(expression)  # Evaluate the expression
        return -abs(result - target_function(x_value))  # Closer to x^2, better the fitness
    except:
        return float('-inf')  # Invalid expressions get very low fitness

# Gene Expression Algorithm
class GeneExpressionAlgorithm:
    def __init__(self, population_size, gene_length, target, generations, mutation_rate, x_value):
        self.population_size = population_size
        self.gene_length = gene_length
        self.target = target
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.x_value = x_value
        self.operators = ['+', '-', '*', '/', '**']
        self.variables = ['x']
        self.constants = ['1', '2', '3', '4', '5']
        self.population = self._initialize_population()

    def _initialize_population(self):
        """
        Generate a random initial population.
        """
        population = []
        for _ in range(self.population_size):
            gene = ''.join(
                np.random.choice(self.variables + self.operators + self.constants, self.gene_length)
            )
            population.append(gene)
        return population

    def _mutate(self, gene):
        """
        Apply random mutation to a gene.
        """
        gene = list(gene)
        for i in range(len(gene)):
            if np.random.rand() < self.mutation_rate:
                gene[i] = np.random.choice(self.variables + self.operators + self.constants)
        return ''.join(gene)

    def evolve(self):
        """
        Evolve the population to optimize the function.
        """
        for generation in range(self.generations):
            # Evaluate fitness for each gene in the population
            fitness = [fitness_function(gene, self.target, self.x_value) for gene in self.population]

            # Select the best-performing genes
            sorted_indices = np.argsort(fitness)[::-1]  # Descending sort
            self.population = [self.population[i] for i in sorted_indices[:self.population_size // 2]]

            # Generate offspring by mutating the best genes
            offspring = [self._mutate(gene) for gene in self.population]
            self.population += offspring

            # Print the best gene of the generation
            print(f"Generation {generation + 1}: Best Gene = {self.population[0]}, Fitness = {fitness[sorted_indices[0]]}")

        # Return the best solution
        best_gene = self.population[0]
        return best_gene

# Main Execution
if __name__ == "__main__":
    # Parameters
    population_size = 20
    gene_length = 5
    target = 25  # Target value of x^2 for x=5
    x_value = 5  # Use x = 5 for optimization
    generations = 10
    mutation_rate = 0.2

    # Initialize and run the algorithm
    gep = GeneExpressionAlgorithm(population_size, gene_length, target, generations, mutation_rate, x_value)
    best_solution = gep.evolve()
    print(f"Best Solution: {best_solution}")
