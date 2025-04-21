import pytest
import numpy as np
from genomix.genetic_algorithm import GeneticAlgorithm


def test_ga_initialization():
    """Test basic initialization of the GeneticAlgorithm class"""
    
    def dummy_objective(**kwargs):
        individual = kwargs['ga_individual']
        return [sum(individual)]
    
    space = {
        'x': {'uniform': [0, 1], 'mutation': [0, 0.1]},
        'y': {'int': [1, 10]},
        'z': {'choice': ['a', 'b', 'c']}
    }
    
    ga = GeneticAlgorithm(
        objective_function=dummy_objective,
        objective_function_params={},
        space=space,
        fitness=("Min",),
        population_size=20,
        algorithm=1
    )
    
    assert ga.pop_size == 20
    assert ga.algo == 1
    assert ga.fit_val == [-1]  # Should be -1 for minimization
    assert set(ga.hyp_space.keys()) == set(['x', 'y', 'z'])



def test_validation_errors():
    """Test that appropriate errors are raised for invalid inputs"""
    
    def dummy_objective(**kwargs):
        return [0]
    
    # Test invalid space
    with pytest.raises(TypeError):
        GeneticAlgorithm(
            objective_function=dummy_objective,
            objective_function_params={},
            space={
                'x': {'invalid_type': [0, 1]}  # Invalid type
            },
            fitness=("Min",)
        )
    
    # Test invalid crossover type
    with pytest.raises(ValueError):
        GeneticAlgorithm(
            objective_function=dummy_objective,
            objective_function_params={},
            space={'x': {'uniform': [0, 1], 'mutation': [0, 0.1]}},
            crossover_type="InvalidType",  # Invalid crossover type
            fitness=("Min",)
        )


