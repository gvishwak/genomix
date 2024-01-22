import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from copy import deepcopy
import random, time, math, itertools, os


class GeneticAlgorithm():
    """
    A python implementation of real-valued, genetic algorithm for solving optimization problems.
    Refer to https://chemrxiv.org/engage/chemrxiv/article-details/60c7445a337d6c2849e26d98 paper for further details about the implementation.

    Parameters
    ----------
    objective_function : function
        The objective function that is to be optimized. It should only have **kwargs as the argument, which are then unpacked within the function.
        For instance, 
        def objective_function(**kwargs):
            <vars> = list(kwargs.values())
        
        The final variable that is unpacked is the list of the trial values of the hyperparameters in the order in which they are declared in the space variable. The objective function can return one or more than one values depending on the type of optimization: single objective or multi-objective.

    objective_function_params: dict
        A dictionary of parameters to be passed to the objective_function. 
        For instance: Use the following to pass the descriptors (X) and target property (Y) to the objective_function:
        
        objective_function_params={'x_data': X, 'y_data': Y}

        The objective function is modified accordingly:

        def objective_function(**kwargs):
            X, Y, individual = list(kwargs.values())
                OR
            X, Y, individual = kwargs['x_data'], kwargs['y_data'], kwargs['ga_individual']

    space : tuple,
        A tuple of dict objects specifying the hyper-parameter space to search in.
        Each hyper-parameter should be a python dict object with the name of the hyper-parameter as the key.
        Value is also a dict object with one mandatory key among: 'uniform', 'int' and 'choice' for defining floating point, integer and choice variables respectively.
        Values for these keys should be a list defining the valid hyper-parameter search space (lower and upper bounds for 'int' and 'uniform', and all valid choices for 'choice').
        For uniform, a 'mutation' key is also required for which the value is [mean, standard deviation] for the gaussian distribution.
        Example:
                ({'alpha': {'uniform': [0.001, 1],
                            'mutation': [0, 1]}},
                {'layers': {'int': [1, 3]}},
                {'neurons': {'choice': range(0, 200, 20)}})

    fitness : tuple, optional (default = ('Max',)
        A tuple of string(s) for Maximizing (Max) or minimizing (Min) the objective function(s). The length of the tuple is the same as the number of variables returned by the objective function.

    population_size : integer, optional (default = 50)
        Size of the population

    crossover_size : int, optional (default = 30)
        Number of individuals to select for crossover.

    mutation_size : int, optional (default = 20)
        Number of individuals to select for mutation.

    crossover_type : string, optional (default = "Blend")
        Type of crossover: SinglePoint, DoublePoint, Blend, Uniform, Fused

    fused_cutoff: int
        The cutoff to use for splitting the individuals for fused crossover

    mutation_fraction : float, optional (default = 0.4)
        Probability of mutation.

    algorithm : int, optional (default=1)
        The algorithm to use for the search. Look at the 'search' method for a description of the various algorithms.

            - Algorithm 1:
                Initial population is instantiated.
                Roulette wheel selection is used for selecting individuals for crossover and mutation.
                The initial population, crossovered and mutated individuals form the pool of individuals from which the best
                n members are selected as the initial population for the next generation, where n is the size of population.

            - Algorithm 2:
                Same as algorithm 1 but when selecting individuals for next generation, n members are selected using Roulette wheel selection.

            - Algorithm 3:
                Same as algorithm 1 but when selecting individuals for next generation, best members from each of the three pools (initital population, crossover and mutation) are selected according to the input parameters in the search method.

            - Algorithm 4:
                Same as algorithm 1 but mutation population is selected from the crossover population and not from the parents directly.

    initial_population : list, optional (default=None)
        The initial population for the algorithm to start with. If not provided, initial population is randomly generated.

    """

    def __init__(self,
                objective_function,
                objective_function_params,
                space,
                optimized_space={},
                fitness=("Max", ),
                population_size=50,
                crossover_size=30,
                mutation_size=20,
                crossover_type="Blend",
                fused_cutoff = 1,
                mutation_fraction=0.75,
                algorithm=2,
                ) -> None:

        self.hyp_space = self._validate_space(space)
        self.optimized_space = self._validate_optimized_space(space=optimized_space)

        if crossover_type not in ['SinglePoint', 'DoublePoint', 'Blend', 'Uniform', 'Fused']:
            raise ValueError('Incorrect crossover type provided. Aborting.')
        if len(self.hyp_space.keys()) <2 and crossover_type in ["SinglePoint", "Fused"]: 
            raise Exception('Single point or Fused crossovers are not possible for only 1 variable in the space parameter.')
        if len(self.hyp_space.keys()) <3 and crossover_type == "DoublePoint": 
            raise Exception('Double point crossover not possible for less than 3 variables in the space parameter.')


        self.fit_val = [1 if i.lower() == 'max' else -1 for i in fitness]
        self.evaluate = objective_function
        self.evaluate_params = objective_function_params
        self.pop_size = population_size
        self.crossover_size = crossover_size
        self.mutation_size = mutation_size
        self.algo = algorithm
        self.mutation_prob = mutation_fraction
        self.crossover_type = crossover_type
        self.fused_cutoff = fused_cutoff


    @staticmethod
    def _validate_space(space):
        """
        space = {'r_max':           {'choice': [i/10 for i in range(1*10, 7*10, 1)]}},
                'num_layers':       {'choice': [3, 4, 5]}},
                'layers':           {'int': [1, 3]},
                'parity':           {'choice': [True, False]}},
                'num_features':     {'choice': list(range(32, 50, 4))}},
                'learning_rate':    {'uniform': [np.log(0.0001), np.log(0.1)], 'mutation': [0, 1]},
                }
        """
        if not isinstance(space, dict):
            raise TypeError("Hyperparameter space variable should be a python dictionary.")
        
        space_types = [h.keys() for h in space.values()]
        space_ranges = [h.values() for h in space.values()]

        check_space_types = [[True if g in ['int', 'choice', 'uniform', 'mutation'] else False for g in h] for h in space_types]
        check_uniform_type = [h for h in space_types if 'uniform' in h]
        check_uniform_type = [True if 'mutation' in h else False for h in check_uniform_type]

        if False in [c for s in check_space_types for c in s]:
            raise TypeError("Incorrect space types provided. Valid options are: 'int', 'choice', 'uniform', 'mutation'")
        
        if False in check_uniform_type:
            raise TypeError("A 'uniform' hyperparameter type must have a 'mutation' parameter to define the [mean, standard deviation] for the gaussian distribution.")


        check_space_values = []

        return space


    def _validate_optimized_space(self, space):
        if space != {}:
            for optimized_key in space:
                if optimized_key not in self.hyp_space.keys():
                    raise NameError("Incorrect optimized space name provided. Names should be consistent with those used in the space parameter.")

                type_key = [i for i in self.hyp_space[optimized_key].keys() if i != 'mutation'][0]
                if type_key == 'int' or type_key == 'uniform': 
                    if not self.hyp_space[optimized_key][type_key][0] < space[optimized_key] < self.hyp_space[optimized_key][type_key][1]:
                        raise ValueError("The optimized hyperparameter value should lie within the range specified in the space parameter.")
                else:
                    if space[optimized_key] not in self.hyp_space[optimized_key][type_key]:
                        raise ValueError("The optimized hyperparameter can only take on values from the list of choices specified in the space parameter.")

        return space


    def generate_population(self, n):
        population = []
        while len(population) < n:
            individual = self.chromosome_generator(len(population), n)
            if individual not in population:
                population.append(individual)

        return population


    def chromosome_generator(self, idx, n):
        individual = []
        for h_name in self.hyp_space:
            h_key = self.hyp_space[h_name].keys()
            if 'uniform' in h_key:
                individual.append(np.linspace(self.hyp_space[h_name]['uniform'][0], self.hyp_space[h_name]['uniform'][1], n)[idx])
            elif 'int' in h_key:
                individual.append(random.randint(self.hyp_space[h_name]['int'][0], self.hyp_space[h_name]['int'][1]))
            elif 'choice' in h_key:
                individual.append(random.choice(self.hyp_space[h_name]['choice']))

        return tuple(individual)


    def SinglePointCrossover(self, x1, x2):
        x1, x2 = list(x1), list(x2)
        nVar=len(x1)
        c = random.randint(1,nVar-1)
        y1=x1[0:c]				
        y1=y1+x2[c:nVar]      
        y2=x2[0:c]
        y2=y2+x1[c:nVar]
        return tuple(deepcopy(y1)), tuple(deepcopy(y2))


    def DoublePointCrossover(self, x1, x2):
        x1, x2 = list(x1), list(x2)
        nVar = len(x1)
        cc = random.sample(range(1,nVar), 2)   
        c1 = min(cc)
        c2 = max(cc)
        y1 = x1[0:c1]+x2[c1:c2]+x1[c2:nVar]				
        y2 = x2[0:c1]+x1[c1:c2]+x2[c2:nVar]      
        return tuple(deepcopy(y1)), tuple(deepcopy(y2))


    def UniformCrossover(self, x1, x2):
        parents = [x1,x2]
        ind1, ind2 = [parents[random.randint(0, 1)][i] for i in range(len(x1))], [parents[random.randint(0, 1)][i] for i in range(len(x1))]   
        return tuple(deepcopy(ind1)), tuple(deepcopy(ind2))


    def blend(self, ind1, ind2, fitness_dict, z=0.4, alpha=0.5, beta=0.1):
        # rank all individuals in fitness dict
        ranked_list = self.select([i for i in fitness_dict], fitness_dict, len(fitness_dict.items()), choice="best")
        # determine the better individual among the two for implementing the alpha-beta crossover
        better, worse = (deepcopy(list(ind1)), deepcopy(list(ind2))) if ranked_list.index(ind1) < ranked_list.index(ind2) else (deepcopy(list(ind2)), deepcopy(list(ind1)))

        for h_index, h_name in enumerate(self.hyp_space.keys()):
            if h_name not in self.optimized_space.keys():
                h_key = [i for i in self.hyp_space[h_name].keys() if i != 'mutation'][0]

                if h_key == 'choice':
                    better[h_index], worse[h_index] = worse[h_index], better[h_index]

                else:
                    while True:
                        absolute_diff = abs(better[h_index] - worse[h_index])

                        if better[h_index] <= worse[h_index]:
                            lower_cap, upper_cap = better[h_index] - absolute_diff * alpha, worse[h_index] + absolute_diff * beta
                        else:
                            lower_cap, upper_cap = worse[h_index] - absolute_diff * beta, better[h_index] + absolute_diff * alpha


                        alpha_beta_crossover_1, alpha_beta_crossover_2 = [lower_cap + random.random() * (upper_cap - lower_cap) for _ in range(2)]
                        # check new values with user-defined bounds
                        if False not in [self.hyp_space[h_name][h_key][0] <= val <= self.hyp_space[h_name][h_key][1] for val in [alpha_beta_crossover_1, alpha_beta_crossover_2]]:
                            break

                    better[h_index], worse[h_index] = alpha_beta_crossover_1, alpha_beta_crossover_2
                    if h_key == 'int':
                        better[h_index], worse[h_index] = int(better[h_index]), int(worse[h_index])

            else:
                better[h_index], worse[h_index] = self.optimized_space[h_name], self.optimized_space[h_name]

    
        return tuple(deepcopy(better)), tuple(deepcopy(worse))


    def fused(self, ind1, ind2, fitness_dict):
        ind1, ind2 = list(ind1), list(ind2)
        x_ind1, y_ind1 = ind1[:self.fused_cutoff], ind1[self.fused_cutoff:]
        x_ind2, y_ind2 = ind2[:self.fused_cutoff], ind2[self.fused_cutoff:]
        x_ind1, x_ind2 = self.blend(x_ind1, x_ind2, fitness_dict)
        y_ind1, y_ind2 = self.UniformCrossover(y_ind1, y_ind2)
        return tuple(deepcopy(list(x_ind1) + list(y_ind1))), tuple(deepcopy(list(x_ind2) + list(y_ind2)))


    def custom_mutate(self, indi, fitness_dict, failed_attempts=0):
        # rank all individuals in fitness dict
        ranked_list = self.select([i for i in fitness_dict], fitness_dict, len(fitness_dict.items()), choice="best")
        # calculate parameter to adjust Gaussian distribution according to individual's rank for uniform type
        parent_fit_param = (ranked_list.index(indi) + 1)*2/len(ranked_list)
        original_individual, indi = deepcopy(list(indi)), list(indi)

        for h_index, h_name in enumerate(self.hyp_space.keys()):
            if h_name not in self.optimized_space.keys():
                h_key = [i for i in self.hyp_space[h_name].keys() if i != 'mutation'][0]

                if h_key == 'uniform':
                    if random.random() < self.mutation_prob:
                        while True:
                            # modify the gaussian mean and standard deviation according to individual's rank
                            add = random.gauss(self.hyp_space[h_name]['mutation'][0], parent_fit_param * self.hyp_space[h_name]['mutation'][1]) + indi[h_index]
                            # check validity of new value in the user-defined range
                            if self.hyp_space[h_name][h_key][0] <= add <= self.hyp_space[h_name][h_key][1]:
                                break

                        indi[h_index] = add

                elif h_key == 'int':
                    if random.random() < self.mutation_prob:
                        indi[h_index] = random.randint(self.hyp_space[h_name][h_key][0], self.hyp_space[h_name][h_key][1])

                elif h_key == 'choice':
                    if random.random() < self.mutation_prob:
                        indi[h_index] = random.choice(list(set(self.hyp_space[h_name][h_key]) - set([indi[h_index]])))

            else:
                indi[h_index] = self.optimized_space[h_name]


        # recursive function call to ensure generation of a different individual
        if self._individual_in_existing_keys(individual=tuple(indi), fitness_dict=fitness_dict):     # tuple(indi) in fitness_dict.keys()
            if failed_attempts < 50:
                failed_attempts += 1
                indi = self.custom_mutate(tuple(original_individual), fitness_dict, failed_attempts=failed_attempts)
            else:
                return None

        return tuple(indi)


    def _deprecated_crossover_and_mutation():
        # # if there is no uniform type hyperparamter in the space variable, run the 'if' condition below to select from a pre-defined superlist of mutations and crossovers.
        # if self.global_cm_list is not None:
        #     try:
        #         if self.chromosome_length == 1:
        #             indi = random.choice(list(set(self.global_cm_list)-set([i[0] for i in list(fitness_dict.keys())])))
        #             if isinstance(indi, int): return indi,
        #             else: return tuple(indi)
        #         else:
        #             indi = random.choice(list(set(self.global_cm_list)-set(list(fitness_dict.keys()))))
        #             return tuple(indi)
        #     except:
        #         return None
    
        # # for type 'choice' apply roulette wheel selection by biasing the wheel towards the two current values
                # if len(self.hyp_space[h_name][h_key]) == 2:
                #     better[h_index], worse[h_index] = worse[h_index], better[h_index]
                # else:
                #     scores = [3 if val == better[h_index] or val == worse[h_index] else 1 for val in self.hyp_space[h_name][h_key]]
                #     sc_dict = {self.hyp_space[h_name][h_key][j]: scores[j] for j in range(len(scores))}
                #     better[h_index], worse[h_index] = self.select(self.hyp_space[h_name][h_key], sc_dict, 2)

        # # if there is no uniform type in the space parameter, use a pre-defined crossover and mutation list as follows
        # self.global_cm_list = None
        # if uni == 0:
        #     if len(space) == 1: 
        #         self.global_cm_list = self.bit_limits[0]
        #     else:
        #         gcl = [list(range(i[0], i[1]+1)) if t == 'int' else i for i, t in zip(self.bit_limits, self.chromosome_type)]
        #         self.global_cm_list = list(itertools.product(*gcl))

        pass


    def select(self, population, fit_dict, num, choice="Roulette"):
        if num >= len(population): return population
        o_fits = [fit_dict[i] for i in population]

        df_fits = pd.DataFrame(o_fits)
        # scale all values in range 1-2
        df2 = [((df_fits[i] - df_fits[i].min()) / (df_fits[i].max() - df_fits[i].min())) + 1 for i in range(df_fits.shape[1])]
        # inverse min columns
        df2 = pd.DataFrame([df2[i]**self.fit_val[i] for i in range(len(df2))]).T
        # rescale all values in range 1-2
        df2 = pd.DataFrame([((df2[i] - df2[i].min()) / (df2[i].max() - df2[i].min())) + 1 for i in range(df2.shape[1])])
        
        fitnesses = list(df2.sum())

        if choice == "Roulette":
            total_fitness = float(sum(fitnesses))
            rel_fitness = [f/total_fitness for f in fitnesses]
            # Generate probability intervals for each individual
            probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]
            # Draw new population
            new_population = []
            for _ in range(num):
                r = random.random()
                for i, individual in enumerate(population):
                    if r <= probs[i]:
                        new_population.append(deepcopy(individual))
                        break
            return new_population
        else:
            fits_sort = sorted(fitnesses, reverse=True)
            best = [deepcopy(population[fitnesses.index(fits_sort[i])]) for i in range(min(num, len(population)))]
            return best


    def fit_eval(self, invalid_ind, fitness_dict):
        invalid_ind = [i for i in invalid_ind if i not in fitness_dict.keys()]
        if invalid_ind:
            fitnesses = [self.evaluate(**{**self.evaluate_params, **{'ga_individual': x}}) for x in invalid_ind]

            if isinstance(fitnesses[0], list) or isinstance(fitnesses[0], tuple):
                if len(self.fit_val) != len(fitnesses[0]):
                    raise ValueError("The objective function is returning a different number of values than the variables that have to be optimized. Please ensure the objective function returns the same number of arguments provided in the fitness variable.")
            else:
                if len(self.fit_val) > 1:
                    raise ValueError("The objective function is returning a different number of values than the variables that have to be optimized. Please ensure the objective function returns the same number of arguments provided in the fitness variable.")

            for ind, fit in zip(invalid_ind, fitnesses):
                fitness_dict[tuple(ind)] = fit

        return fitness_dict


    def _gen_crossover(self, fitness_dict, pop, total_pop):
        # Generate crossover population
        crossover_final_pop = []
        crossover_init_pop = self.select(pop, fitness_dict, int(math.ceil(self.crossover_size)))
        crossover_init_pop = list(itertools.combinations(list(set(crossover_init_pop)), 2)) + list(itertools.combinations(list(set(pop + total_pop)), 2))

        failed_attempts = 0
        while len(crossover_final_pop) < int(math.ceil(self.crossover_size)) and failed_attempts < 50:
            for child1, child2 in crossover_init_pop:
                if len(crossover_final_pop) >= int(math.ceil(self.crossover_size)):
                    break
                if self.crossover_type == "SinglePoint":
                    c1, c2 = self.SinglePointCrossover(child1, child2)
                elif self.crossover_type == "DoublePoint":
                    c1, c2 = self.DoublePointCrossover(child1, child2)
                elif self.crossover_type == "Blend":
                    c1, c2 = self.blend(child1, child2, fitness_dict)
                elif self.crossover_type == "Fused":
                    c1, c2 = self.fused(child1, child2, fitness_dict)
                elif self.crossover_type == "Uniform":
                    c1, c2 = self.UniformCrossover(child1, child2)

                if not self._individual_in_existing_keys(individual=c1, fitness_dict=fitness_dict) and c1 not in crossover_final_pop:
                    crossover_final_pop.append(c1)
                
                if not self._individual_in_existing_keys(individual=c2, fitness_dict=fitness_dict) and c2 not in crossover_final_pop:
                    crossover_final_pop.append(c2)

            failed_attempts += 1

        return list(set(crossover_final_pop))


    def _gen_mutation(self, fitness_dict, pop, crossover_final_pop):
        # Generate mutation population
        mutation_final_pop, flag = [], False
        m_select = crossover_final_pop if self.algo == 4 else pop
        mutation_initial_pop = self.select(m_select, fitness_dict, int(math.ceil(self.mutation_size)))

        for mutant in mutation_initial_pop:
            a = self.custom_mutate(mutant, fitness_dict)
            if a is not None:
                mutation_final_pop.append(a)
            else: 
                print("All combinations exhausted. Stopping genetic algorithm iterations.")
                flag = True
                break

        return list(set(mutation_final_pop)), flag


    def _select_next_gen_pop(self, fitness_dict, total_pop, init_ratio, crossover_ratio, crossover_final_pop, mutation_final_pop, c_gen=None, n_generations=None):
        if self.algo == 2 and c_gen != n_generations - 1:
            pop = self.select(total_pop, fitness_dict, self.pop_size)
        elif self.algo == 3:
            p1 = self.select(pop, fitness_dict, int(init_ratio*self.pop_size), choice="best")
            p2 = self.select(crossover_final_pop, fitness_dict, int(crossover_ratio*self.pop_size), choice="best")
            p3 = self.select(mutation_final_pop, fitness_dict, self.crossover_size+self.mutation_size-len(p1)-len(p2), choice="best")
            pop = p1 + p2 + p3
        else:
            pop = self.select(total_pop, fitness_dict, self.pop_size, choice="best")

        return pop


    def _run_generations(self, n_generations, early_stopping, fitness_dict, pop, init_ratio, crossover_ratio):
        best_indi_per_gen, best_indi_fitness_values, timer, total_pop, convergence, flag = [], [], [], [], 0, False
        for c_gen in range(n_generations):
            if convergence >= early_stopping:
                print("The search converged with convergence criteria = ", early_stopping)
                break

            st_time = time.time()

            crossover_final_pop = self._gen_crossover(fitness_dict=fitness_dict, pop=pop, total_pop=total_pop)
            fitness_dict = self.fit_eval(crossover_final_pop, fitness_dict)

            mutation_final_pop, flag = self._gen_mutation(fitness_dict=fitness_dict, pop=pop, crossover_final_pop=crossover_final_pop)
            fitness_dict = self.fit_eval(mutation_final_pop, fitness_dict)

            # Select the next generation individuals
            total_pop = pop + crossover_final_pop + mutation_final_pop
            pop = self._select_next_gen_pop(fitness_dict=fitness_dict, total_pop=total_pop, init_ratio=init_ratio, crossover_ratio=crossover_ratio, crossover_final_pop=crossover_final_pop, mutation_final_pop=mutation_final_pop, c_gen=c_gen, n_generations=n_generations)
            
            # Storing the best individuals after each generation
            best_individual = pop[0]
            best_indi_per_gen.append(best_individual)
            best_indi_fitness_values.append(fitness_dict[best_individual])
            tot_time = (time.time() - st_time)/(60*60)
            timer.append(tot_time)
            if c_gen != 0:
                if best_indi_per_gen[-1] == best_indi_per_gen[-2]: 
                    convergence += 1
                else: 
                    convergence = 0

            if flag: 
                break

    
        b1 = pd.Series(best_indi_per_gen, name='Best_individual')
        b2 = pd.Series(best_indi_fitness_values, name='Fitness_values')
        b3 = pd.Series(timer, name='Time (hours)')
        best_ind_df = pd.concat([b1, b2, b3], axis=1)
    

        self.population = pop    # stores best individuals of last generation
        self.fitness_dict = fitness_dict
        best_ind_dict = {name: val for name, val in zip(self.var_names, best_individual)}

        return best_ind_df, best_ind_dict


    def _validate_fitness_dict(self, fitness_dict, batch_mode):
        if fitness_dict is None or fitness_dict == {}:
            pop = self.generate_population(n=self.pop_size)       # list of tuples
            fitness_dict = {}

        else:
            check_keys = [True if isinstance(i, list) or isinstance(i, tuple) else False for i in fitness_dict.keys()]
            # TODO: also check key length and key values
            if False in check_keys:
                raise ValueError("Incorrect format of fitness dictionary provided.")

            loss_values = fitness_dict.values()
            if None in loss_values:
                if batch_mode:
                    raise ValueError("Loss values cannot be None in batch mode.")
                
                uneval_pop = [i for i, j in fitness_dict.items() if j is None]
                # evaluate initial pop provided
                fitness_dict = {key: value for key, value in fitness_dict if key not in uneval_pop}
                fitness_dict = self.fit_eval(uneval_pop, fitness_dict)
    
            check_vals = [True if isinstance(i, int) or isinstance(i, float) else False for i in loss_values]
            if False in check_vals:
                raise ValueError("Loss values should only be int, float or None type.")

            pop = list(fitness_dict.keys())

        return pop, fitness_dict


    def _individual_in_existing_keys(self, individual, fitness_dict):
        exists = True if individual in fitness_dict.keys() else False

        # if self.optimized_space == {}:
        #     exists = True if individual in fitness_dict.keys() else False

        # else:
        #     reduced_indices = [idx for idx, i in enumerate(self.hyp_space.keys()) if i not in self.optimized_space.keys()]
        #     reduced_fitness_keys = [[k[x] for x in reduced_indices] for k in fitness_dict.keys()]
        #     reduced_individual = [individual[x] for x in reduced_indices]
        #     exists = True if reduced_individual in reduced_fitness_keys else False

        return exists
    

    @staticmethod
    def _random_sample(superset, size):
        return random.sample(population=superset, k=size) if len(superset) > size else [random.sample(population=superset, k=1) for _ in range(size)]


    def search(self, batch_mode, fitness_dict=None, n_generations=20, early_stopping=10, init_ratio = 0.35, crossover_ratio = 0.35):
        """
        Parameters
        ----------
        batch_mode: bool
            True if genetic algorithm is to be run in batch mode else False. When True, the objective function is not evaluated but is instead left to the user.
            Use True when objective function is too expensive to calculate and/or objective function cannot be run in a continuous mode

        fitness_dict: dict
            A dictionary providing VALID and COMPATIBLE population of individuals for this code. In case of reruns, provide the same dictionary returned by the code in previous iterations.
            Can also be used to provide initial population for the code to work with; in this case the dictionary values should be None.

        n_generations : integer, optional (default = 20)
                An integer for the number of generations to evolve the population for.

        early_stopping : int, optional (default=10)
                Integer specifying the maximum number of generations for which the algorithm can select the same best individual, after which 
                the search terminates.

        init_ratio : float, optional (default = 0.4)
            Fraction of initial population to select for next generation. Required only for algorithm 3.

        crossover_ratio : float, optional (default = 0.3)
            Fraction of crossover population to select for next generation. Required only for algorithm 3.

        
        Attributes
        ----------
        population : list,
            list of individuals from the final generation

        fitness_dict : dict,
            dictionary of all individuals evaluated by the algorithm


        Returns
        -------
        ------------------ if batch_mode is False ------------------
        best_ind_df :  pandas dataframe
            A pandas dataframe of best individuals of each generation

        best_ind :  dict,
            The best individual after the last generation.


        ------------------ if batch_mode is True ------------------
        pop: list
            list of new individuals created by genetic algorithm.
            The user assigns the objective function values 
        """

        if init_ratio >=1 or crossover_ratio >=1 or (init_ratio+crossover_ratio)>=1: 
            raise Exception("Sum of parameters init_ratio and crossover_ratio should be in the range (0,1)")

        pop, fitness_dict = self._validate_fitness_dict(fitness_dict=fitness_dict, batch_mode=batch_mode)


        if batch_mode:
            # # Algorithm 2 uses 'roulette' filter instead of 'best' for final generation, thus does not make sense for batch mode
            # self.algo = 1 if self.algo == 2 else self.algo

            if len(fitness_dict.keys()) == 0:
                return pop

            else:
                crossover_pop_previous_gen = self._random_sample(superset=pop, size=self.crossover_size)
                mutation_pop_previous_gen = self._random_sample(superset=pop, size=self.mutation_size)
                pop = self._select_next_gen_pop(fitness_dict=fitness_dict, total_pop=list(fitness_dict.keys()), init_ratio=init_ratio, crossover_ratio=crossover_ratio, crossover_final_pop=crossover_pop_previous_gen, mutation_final_pop=mutation_pop_previous_gen)

                # generate new crossover and mutation population 
                crossover_final_pop = self._gen_crossover(fitness_dict=fitness_dict, pop=pop, total_pop=list(fitness_dict.keys()))
                mutation_final_pop, _ = self._gen_mutation(fitness_dict=fitness_dict, pop=pop, crossover_final_pop=self._random_sample(superset=pop, size=self.crossover_size))

                return crossover_final_pop + mutation_final_pop

        else:
            # Evaluate the initial population
            fitness_dict = self.fit_eval(pop, fitness_dict)
            best_ind_df, best_ind_dict = self._run_generations(n_generations=n_generations, early_stopping=early_stopping, fitness_dict=fitness_dict, pop=pop, init_ratio=init_ratio, crossover_ratio=crossover_ratio)

            return best_ind_df, best_ind_dict


    def hyperparameter_statistics(self, fitness_dict, save_path, batch_mode):
        if fitness_dict is None or fitness_dict == {}:
            raise ValueError('Incorrect fitness dict provided.')


        total_pop, fitness_dict = self._validate_fitness_dict(fitness_dict=fitness_dict, batch_mode=batch_mode)

        pop = self.select(total_pop, fitness_dict, self.pop_size, choice="best")
        reference = pop[0]
        ref_loss = fitness_dict[reference]
        print('best hyp loss value: ', ref_loss)

        for h_index, h_name in enumerate(self.hyp_space.keys()):
            h_key = [i for i in self.hyp_space[h_name].keys() if i != 'mutation'][0]
            h_valid_values = self.hyp_space[h_name][h_key]

        # for hp_index, name, h_key, limits in zip(range(len(self.var_names)), self.var_names, self.chromosome_type, self.bit_limits):

            ref_hp_value = reference[h_index]

            hp_instances_list = [[i for i in fitness_dict.keys() if i[h_index] == j] for j in h_valid_values]
            hp_instances_loss_values = [[fitness_dict[j] for j in i] for i in hp_instances_list]
            # relative_loss = [[(i-ref_loss)/ref_loss for i in j] for j in hp_instances_loss_values]
            relative_loss = hp_instances_loss_values

            if h_key == 'choice':
                pass

            elif h_key == 'int':
                # relative_hp_value = [[for h in i] for i in hp_instances_list]
                pass

            elif h_key == 'uniform':
                pass

            value_edited = []
            for l, b in zip(relative_loss, h_valid_values):
                for _ in range(len(l)):
                    if isinstance(b, tuple) or isinstance(b, list):
                        value_edited.append('-'.join([str(a) for a in b]))  
                    else:
                        value_edited.append(b)

            df_loss = pd.DataFrame({'relative loss': list(itertools.chain.from_iterable(relative_loss)), 'value': value_edited})
            df_loss = df_loss[df_loss['relative loss'] != 0]    # -1
            df_loss = df_loss.append({'relative loss': ref_loss, 'value': 'reference'}, ignore_index=True)

            plt.figure()
            strip_plot = sns.stripplot(data=df_loss, x='value', y='relative loss', hue='value')
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
            plt.xticks(rotation=50)
            plt.title(h_name)
            plt.tight_layout()
            # plt.show()

            # strip_plot.figure.savefig(os.path.join(save_path, name + '.png'))
            plt.savefig(os.path.join(save_path, h_name + '.png'))

