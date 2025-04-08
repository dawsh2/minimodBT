"""
Genetic Algorithm module for optimizing trading rule weights.

This simplified version maintains compatibility with your existing code
while providing a cleaner interface.
"""

import numpy as np

def cal_pop_fitness(equation_inputs, pop, opt = 0):
    """
    Calculate the fitness value of each solution in the current population.
    
    Args:
        equation_inputs: Trading rule features with the first column being returns
        pop: Population of potential solutions (weights)
        opt: Optimization criterion (0 for default optimization)
        
    Returns:
        Array of fitness values for each solution
    """
    # Extract log returns (first column)
    logr = equation_inputs[:, 0]
    
    # Calculate weighted positions
    positions = pop @ equation_inputs[:, 1:].T
    
    # Calculate strategy returns
    port_r = (positions * logr).astype(np.float64)
    
    # Calculate Sharpe Ratio-like metric (SSR)
    # Mean return / Std deviation / Negative sum of negative returns
    # This rewards high returns, low volatility, and small drawdowns
    mean_returns = np.mean(port_r, axis=1)
    std_returns = np.std(port_r, axis=1)
    negative_returns = np.sum(port_r * (port_r < 0), axis=1)
    
    # Handle division by zero
    std_returns = np.where(std_returns == 0, 1e-8, std_returns)
    negative_returns = np.where(negative_returns == 0, -1e-8, negative_returns)
    
    SSR = mean_returns / std_returns / (-negative_returns)
    
    return SSR

def select_mating_pool(pop, fitness, num_parents):
    """
    Select the best individuals for mating.
    
    Args:
        pop: Population of solutions
        fitness: Fitness values for each solution
        num_parents: Number of parents to select
        
    Returns:
        Selected parents for breeding next generation
    """
    parents = np.empty((num_parents, pop.shape[1]))
    
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999  # Ensure this parent isn't selected again
    
    return parents

def crossover(parents, offspring_size):
    """
    Create offspring through crossover of pairs of parents.
    
    Args:
        parents: Selected parents for breeding
        offspring_size: Size of the offspring to produce
        
    Returns:
        Offspring created through crossover
    """
    offspring = np.empty(offspring_size)
    
    # The point at which crossover takes place between two parents
    crossover_point = np.uint8(offspring_size[1] / 2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate
        parent2_idx = (k + 1) % parents.shape[0]
        
        # First half from first parent
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # Second half from second parent
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    
    return offspring

def mutation(offspring_crossover, num_mutations=1):
    """
    Apply random mutations to the offspring.
    
    Args:
        offspring_crossover: Offspring created through crossover
        num_mutations: Number of mutations to apply
        
    Returns:
        Mutated offspring
    """
    mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
    
    # Mutation changes a number of genes as defined by the num_mutations argument
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for _ in range(num_mutations):
            # Add a random value to the gene
            random_value = np.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    
    return offspring_crossover
