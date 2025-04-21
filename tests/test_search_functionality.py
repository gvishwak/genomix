import pytest
import numpy as np
from genomix.genetic_algorithm import GeneticAlgorithm


def test_basic_search():
    """Test a basic search on a simple objective function"""
    
    def sphere_function(**kwargs):
        """Simple sphere function for testing (minimum at origin)"""
        individual = kwargs['ga_individual']
        return [sum(x**2 for x in individual)]
    
    space = {
        'x1': {'uniform': [-5, 5], 'mutation': [0, 0.5]},
        'x2': {'uniform': [-5, 5], 'mutation': [0, 0.5]}
    }
    
    ga = GeneticAlgorithm(
        objective_function=sphere_function,
        objective_function_params={},
        space=space,
        fitness=("Min",),
        population_size=20,
        crossover_size=10,
        mutation_size=10,
        algorithm=1
    )
    
    # Run for just a few generations
    best_ind_df, best_ind_dict = ga.search(
        batch_mode=False,
        n_generations=5
    )
    
    # Check that we have results
    assert len(best_ind_df) > 0
    assert len(best_ind_dict) == 2
    
    # Check that keys in best_ind_dict match space keys
    assert set(best_ind_dict.keys()) == set(space.keys())
    
    # Check that fitness values are improving (or at least not getting worse)
    fitness_values = best_ind_df['Fitness_values'].tolist()
    assert fitness_values[-1] <= fitness_values[0]


def test_batch_mode():
    """Test batch mode where evaluations are done outside the GA loop"""
    
    def dummy_objective(**kwargs):
        individual = kwargs['ga_individual']
        return [sum(individual)]
    
    space = {
        'x': {'uniform': [0, 1], 'mutation': [0, 0.1]},
        'y': {'uniform': [0, 1], 'mutation': [0, 0.1]}
    }
    
    ga = GeneticAlgorithm(
        objective_function=dummy_objective,
        objective_function_params={},
        space=space,
        fitness=("Min",),
        population_size=10,
        algorithm=1
    )
    
    # First call to get initial population
    pop = ga.search(batch_mode=True)
    assert len(pop) == 10
    
    # Create fitness dictionary
    fitness_dict = {}
    for ind in pop:
        fitness_dict[ind] = [ind[0] + ind[1]]  # Simple sum
    
    # Second call with fitness_dict
    new_pop = ga.search(batch_mode=True, fitness_dict=fitness_dict)
    assert len(new_pop) > 0
    assert all(ind not in fitness_dict for ind in new_pop)  # New individuals should not be in fitness_dict

