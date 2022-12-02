import itertools
import math
import random
import numpy as np


class PartyAllocationGA:
    def __init__(self, num_friends, population_size, seed=0):
        self._num_friends = num_friends
        self._population_size = population_size
        self._interaction_matrix = self._create_interaction_matrix(seed)
        self._population = self._create_initial_population(population_size, seed)
        self._pop_fitness = [self._calculate_fitness(i) for i in self._population]

    @property
    def population(self):
        return self._population

    @property
    def pop_fitness(self):
        return self._pop_fitness

    def get_best_individual(self):
        best_individuals = np.argmax(self._pop_fitness)
        return self._pop_fitness[best_individuals], self._population[best_individuals]

    def _create_interaction_matrix(self, seed=0):
        """Initialise interaction matrix.

        :param seed: (int) Seed for rng. 0 indicates no seed should be set (default = 0).
        :return: Initialised interaction matrix.
        """
        if seed:
            np.random.seed(seed)  # Set seed for testing based on same interaction matrix
        temp_matrix = np.random.choice([-1, 1], size=(self._num_friends, self._num_friends))  # Random interaction matrix
        np.fill_diagonal(temp_matrix, 0) # Set diagonals to 0 to remove interactions with self
        return temp_matrix

    def _create_initial_population(self, pop_size, seed=0):
        """Initialise population randomly.

        :param pop_size: (int) Size of the population.
        :param seed: (int) Seed for rng. 0 indicates no seed should be set (default = 0).
        :return: Initialised population.
        """
        if seed:
            np.random.seed(seed)  # Set seed for testing based on same interaction matrix
        initial_pop = np.random.choice([-1, 1], size=[pop_size, self._num_friends])  # Initialise population randomly
        return initial_pop

    def _calculate_fitness(self, solution):
        """
        :param solution: A single genotype for evaluation (i.e. an allocation of friends at pubs)
        :return: Fitness value for genotype (int)
        """
        return np.matmul(np.matmul(solution, self._interaction_matrix), np.transpose(solution))

    def brute_force_optimal_solution(self):
        """Find the best solution using brute force search.

        :return: (int) best fitness, (list) genotype of best solution
        """
        best_fitness = -math.inf
        best_genotype = None
        for j in [list(i) for i in itertools.product([-1, 1], repeat=self._num_friends)]:
            fitness = self._calculate_fitness(j)
            if fitness > best_fitness:
                best_fitness = fitness
                best_genotype = j
        return best_fitness, best_genotype

    def run(self, n_tournaments, deme_size, mutation_rate, include_crossover=False, recombination_rate=0):
        """Run the genetic algorithm.

        :param n_tournaments: (int) Number of tournaments to run for.
        :param deme_size: (int) Size of deme to consider when choosing individuals for tournaments.
        :param mutation_rate: (int) Probability of mutating each gene in a genotype.
        :param include_crossover: (bool) Indicate whether crossover should be used or not (default = False).
        :param recombination_rate: (int) Probability of crossover for each gene of loser (default = 0).
        :return: (int) 0 if optimal solution not found, 1 if optimal solution found.
        """
        best_solution_fitness = self.brute_force_optimal_solution()[0]
        start_index = np.random.choice(self._population_size)  # Set starting index for tournament selection
        for i in range(0, n_tournaments):
            if max(self._pop_fitness) == best_solution_fitness:
                print("Optimal solution found")
                yield 1
            # Choose two individuals within deme_size for tournament
            a_index = (start_index + i) % self._population_size
            b_index = (a_index + 1 + np.random.choice(deme_size)) % self._population_size
            a = self._population[a_index]
            b = self._population[b_index]
            # Compare chosen individuals fitness
            if self._pop_fitness[a_index] > self._pop_fitness[b_index]:
                winner = a
                loser = b
                loser_index = b_index
            else:
                winner = b
                loser = a
                loser_index = a_index
            # Perform crossover if crossover is included
            if include_crossover:
                for j in range(self._num_friends):
                    if random.uniform(0, 1) < recombination_rate:
                        loser[j] = winner[j]
            # Perform mutation
            for j in range(self._num_friends):
                if random.uniform(0, 1) < mutation_rate:
                    if loser[j] == 1:
                        loser[j] = -1
                    else:
                        loser[j] = 1
            # Update fitness of loser
            self._pop_fitness[loser_index] = self._calculate_fitness(loser)
            yield 0

    def reset(self, seed=0):
        """Reset the population and population fitness ready for another run of GA.

        :param seed: seed for rng. If 0 don't set seed.
        :return: None
        """
        self._population = self._create_initial_population(self._population_size, seed)
        self._pop_fitness = [self._calculate_fitness(i) for i in self._population]
