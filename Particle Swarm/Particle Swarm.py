import numpy as np

num_particles = 30
num_iterations = 10
inertia_weight = 0.5
cognitive_coefficient = 1.5
social_coefficient = 1.5
solution_dim = 2

def fitness_function(position):
    return -sum(x**2 for x in position)

particles_position = np.random.uniform(-10, 10, (num_particles, solution_dim))
particles_velocity = np.random.uniform(-1, 1, (num_particles, solution_dim))
personal_best_position = particles_position.copy()
personal_best_fitness = np.array([fitness_function(p) for p in particles_position])
global_best_position = personal_best_position[np.argmax(personal_best_fitness)]
global_best_fitness = np.max(personal_best_fitness)

for iteration in range(num_iterations):
    for i in range(num_particles):
        fitness = fitness_function(particles_position[i])
        if fitness > personal_best_fitness[i]:
            personal_best_position[i] = particles_position[i].copy()
            personal_best_fitness[i] = fitness
        if fitness > global_best_fitness:
            global_best_position = particles_position[i].copy()
            global_best_fitness = fitness

    for i in range(num_particles):
        r1, r2 = np.random.rand(2)
        cognitive_velocity = cognitive_coefficient * r1 * (personal_best_position[i] - particles_position[i])
        social_velocity = social_coefficient * r2 * (global_best_position - particles_position[i])
        particles_velocity[i] = inertia_weight * particles_velocity[i] + cognitive_velocity + social_velocity
        particles_position[i] += particles_velocity[i]

    print(f"Iteration {iteration + 1}: Best Fitness = {global_best_fitness}, Best Position = {global_best_position}")

print("\nBest Solution after Iterations:")
print("Best Position:", global_best_position)
print("Best Fitness:", global_best_fitness)
