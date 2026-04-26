
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# GENETIC ALGORITHM TUTORIAL (CLEAN VERSION)
# =====================================================

# -----------------------------------------------------
# Problem Definition
# Maximize: y = sum(w_i * x_i)
#Task is to find best w_i
# -----------------------------------------------------
equation_inputs = np.array([4, -2, 3.5, 5, -11, -4.7])
num_weights = len(equation_inputs)

# -----------------------------------------------------
# GA Parameters
# -----------------------------------------------------
population_size = 10
num_parents = 2
num_generations = 20
mutation_rate = 0.3 #Each offspring has a 30% chance of being mutated 70% chance → no change

# Store best fitness per generation (for plotting)
fitness_history = []


# -----------------------------------------------------
# Initialize Population
# Each row is a chromosome (solution)
# -----------------------------------------------------
def initialize_population():
    return np.random.uniform(-4, 4, (population_size, num_weights))


# -----------------------------------------------------
# Fitness Function
# Computes dot product: w.x
# -----------------------------------------------------
def fitness(population):

    # Default objective
    return np.sum(population * equation_inputs, axis=1)

    # =================================================
    # EXERCISE 3: Target vector optimization
    # Uncomment below and comment above line
    # Goal: converge to [2,2,2,2,2,2]
    # =================================================
    # return -np.sum((population - 2)**2, axis=1)

    # =================================================
    # EXERCISE 4: Multi-objective (weighted sum)
    # Balance output and small weights
    # =================================================
    # f1 = np.sum(population * equation_inputs, axis=1)
    # f2 = -np.sum(population**2, axis=1)
    # return f1 + 0.1 * f2


# -----------------------------------------------------
# Selection: choose best individuals
# -----------------------------------------------------
def select_parents(population, fitness_values):
    parents = np.empty((num_parents, num_weights))
    fitness_copy = fitness_values.copy()

    for i in range(num_parents):
        idx = np.argmax(fitness_copy)
        parents[i] = population[idx]
        fitness_copy[idx] = -np.inf  # avoid selecting again

    return parents


# -----------------------------------------------------
# Crossover: combine two parents
# -----------------------------------------------------
def crossover(parents):
    offspring = []
    crossover_point = num_weights // 2

    for i in range(population_size - len(parents)):
        p1 = parents[i % len(parents)]
        p2 = parents[(i + 1) % len(parents)]

        child = np.concatenate([p1[:crossover_point],
                                p2[crossover_point:]])
        offspring.append(child)

    return np.array(offspring)


# -----------------------------------------------------
# Mutation: randomly modify one gene
# -----------------------------------------------------
def mutation(offspring):
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            gene_idx = np.random.randint(0, num_weights)
            offspring[i][gene_idx] += np.random.uniform(-1, 1)
    return offspring


# -----------------------------------------------------
# Constraint Handling (Exercise 5)
# -----------------------------------------------------
def apply_constraints(population, fitness_values):

    # =================================================
    # EXERCISE 5: Add penalty if weights exceed limits
    # Uncomment below
    # =================================================
    # for i in range(len(population)):
    #     if np.any(np.abs(population[i]) > 10):
    #         fitness_values[i] -= 100

    return fitness_values


# -----------------------------------------------------
# Main GA Loop
# -----------------------------------------------------
population = initialize_population()

for generation in range(num_generations):

    fit = fitness(population)

    # Apply constraints if enabled
    fit = apply_constraints(population, fit)

    best_fit = np.max(fit)
    fitness_history.append(best_fit)

    print("Generation", generation, "| Best Fitness:", round(best_fit, 4))

    parents = select_parents(population, fit)
    offspring = crossover(parents)
    offspring = mutation(offspring)

    # Create new population
    population[:len(parents)] = parents
    population[len(parents):] = offspring


# -----------------------------------------------------
# Final Result
# -----------------------------------------------------
fit = fitness(population)
best_idx = np.argmax(fit)

print("\nBest Solution:", population[best_idx])
print("Best Fitness:", fit[best_idx])


# -----------------------------------------------------
 

plt.plot(fitness_history) 
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("GA Convergence")
plt.show()


# -----------------------------------------------------
# EXERCISE 1. Mutation Study
# Try changing mutation_rate above:
# mutation_rate = 0.0   # no mutation
# mutation_rate = 0.9   # too random
# mutation_rate = 0.3   # balanced
# -----------------------------------------------------

# =================================================
# EXERCISE 2: Population Size Study
# Try different values and compare convergence
# =================================================
# population_size = 5     # Very small → fast but poor exploration
# population_size = 50    # Better exploration
# population_size = 200   # Very slow but stable search

# =================================================
# EXERCISE 3. Number of Parents
# =================================================
# num_parents = 1   # Extreme elitism (almost cloning)
# num_parents = 2   # Default
# num_parents = 5   # More diversity
# =================================================


# EXERCISE 4. Tournament Selection
# Replace select_parents function
# =================================================
#Instead of picking global best (greedy), pick best from a random mini-group, prevents same individuals from dominating Maintains diversity
# def select_parents(population, fitness_values):
#     parents = []
#     k = 3  # tournament size
#    
#     for _ in range(num_parents):
#         indices = np.random.choice(len(population), k, replace=False)
#         best_idx = indices[np.argmax(fitness_values[indices])]
#         parents.append(population[best_idx])
#
#     return np.array(parents)

# =================================================
# EXERCISE 5. Early stopping, practical optimization stopping
# =================================================
# if generation > 5:
#     if abs(fitness_history[-1] - fitness_history[-5]) < 1e-3:
#         print("Early stopping triggered")
#         break



