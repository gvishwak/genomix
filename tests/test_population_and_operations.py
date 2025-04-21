import pytest
import numpy as np
from genomix.genetic_algorithm import GeneticAlgorithm



@pytest.fixture
def simple_ga():
    """Create a simple GA instance for testing"""
    def dummy_objective(**kwargs):
        individual = kwargs['ga_individual']
        return [sum(individual)]
    
    space = {
        'x': {'uniform': [0, 1], 'mutation': [0, 0.1]},
        'y': {'int': [1, 10]},
    }
    
    return GeneticAlgorithm(
        objective_function=dummy_objective,
        objective_function_params={},
        space=space,
        fitness=("Min",),
        population_size=20,
        algorithm=1
    )



def test_population_generation(simple_ga):
    """Test that population is generated correctly"""
    population = simple_ga.generate_population(10)
    
    assert len(population) == 10
    assert all(isinstance(ind, tuple) for ind in population)
    assert all(len(ind) == 2 for ind in population)
    
    # Check that values are within bounds
    for ind in population:
        x, y = ind
        assert 0 <= x <= 1
        assert 1 <= y <= 10
        assert isinstance(y, int)



def test_crossover_operations(simple_ga):
    """Test different crossover operations"""
    parent1 = (0.3, 5)
    parent2 = (0.7, 8)
    
    # Create a fitness dict for blend crossover
    fitness_dict = {
        parent1: [1.0],
        parent2: [2.0]
    }
    
    # Test SinglePointCrossover
    child1, child2 = simple_ga.SinglePointCrossover(parent1, parent2)
    assert child1 != parent1 and child1 != parent2
    assert child2 != parent1 and child2 != parent2
    assert len(child1) == 2 and len(child2) == 2
    
    # Test BlendCrossover
    simple_ga.crossover_type = "Blend"
    child1, child2 = simple_ga.blend(parent1, parent2, fitness_dict)
    assert len(child1) == 2 and len(child2) == 2
    assert isinstance(child1[1], int) and isinstance(child2[1], int)
    
    # Test UniformCrossover
    simple_ga.crossover_type = "Uniform"
    child1, child2 = simple_ga.UniformCrossover(parent1, parent2)
    assert len(child1) == 2 and len(child2) == 2
    assert all(val in (parent1[i], parent2[i]) for i, val in enumerate(child1))
    assert all(val in (parent1[i], parent2[i]) for i, val in enumerate(child2))



def test_selection(simple_ga):
    """Test selection operation"""
    population = [(0.1, 3), (0.2, 5), (0.3, 7), (0.4, 9)]
    fitness_dict = {
        (0.1, 3): [0.5],  # Best (lowest value for minimization)
        (0.2, 5): [1.0],
        (0.3, 7): [1.5],
        (0.4, 9): [2.0]   # Worst
    }
    
    # Test best selection
    selected = simple_ga.select(population, fitness_dict, 2, choice="best")
    assert len(selected) == 2
    assert (0.1, 3) in selected  # Best individual should be selected
    
    # Test roulette selection
    selected = simple_ga.select(population, fitness_dict, 2, choice="Roulette")
    assert len(selected) == 2



