import numpy as np
import random
import operator
from past.builtins import range

random.seed()

Nd = 9  # Number of digits, this is a solver to solve a 9x9 Sudoku.


class Population(object):
    """ possible solutions of the sudoku puzzle """

    def __init__(self):
        self.candidates = []
        return

    def seed(self, Nc, given):
        self.candidates = []

        # Determine the valid values.
        helper = Candidate()

        # The helper is a 3d list, two of them determine the position and the other one is a list of valid numbers
        helper.values = [[[] for _ in range(0, Nd)] for _ in range(0, Nd)]
        for row in range(0, Nd):
            for column in range(0, Nd):
                for value in range(1, 10):
                    # Check if it is allowed to put a number into the position
                    # And check if the new number will cause duplicate in this row/column/block
                    if (given.values[row][column] == 0) and not given.is_column_duplicate(column,
                                                                                          value) and not given.is_block_duplicate(
                            row, column, value) and not given.is_row_duplicate(row, value):
                        helper.values[row][column].append(value)
                    elif given.values[row][column] != 0:
                        # Given/known value from file.
                        helper.values[row][column].append(given.values[row][column])
                        break

        for _ in range(0, Nc):
            g = Candidate()
            for i in range(0, Nd):  # New row in candidate.
                row = np.zeros(Nd)

                # Fill in the givens.
                for j in range(0, Nd):

                    # We can only fill the position where the number in it is 0 in given puzzle
                    if given.values[i][j] != 0:
                        row[j] = given.values[i][j]
                    elif given.values[i][j] == 0:
                        row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]

                k = 0
                # use len(list(set(row))) to eliminate the duplicate candidates
                while len(list(set(row))) != Nd:
                    k += 1

                    # Set a threshold to stop
                    if k > 500000:
                        return 0
                    for j in range(0, Nd):
                        if given.values[i][j] == 0:
                            row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]

                g.values[i] = row
            self.candidates.append(g)
        # Compute the fitness of all candidates in the population.
        self.update_fitness()

        return 1

    def update_fitness(self):
        for candidate in self.candidates:
            candidate.update_fitness()
        return

    def sort(self):
        self.candidates = sorted(self.candidates, key=operator.attrgetter('fitness'))
        return


class Candidate(object):
    """ A possible solution of a sudoku puzzle"""

    def __init__(self):
        self.values = np.zeros((Nd, Nd))
        self.fitness = None
        return

    def update_fitness(self):
        """To calculate the fitness of a solution
        the number which appears only once in a column/block will contribute to the fitness"""

        column_count = np.zeros(Nd)
        block_count = np.zeros(Nd)
        column_sum = 0
        block_sum = 0

        self.values = self.values.astype(int)
        # Calculate the fitness in every column
        for j in range(0, Nd):
            for i in range(0, Nd):
                column_count[self.values[i][j] - 1] += 1

            for k in range(len(column_count)):
                if column_count[k] == 1:
                    column_sum += (1 / Nd) / Nd
            column_count = np.zeros(Nd)

        # Calculate the fitness in every block
        for i in range(0, Nd, 3):
            for j in range(0, Nd, 3):
                block_count[self.values[i][j] - 1] += 1
                block_count[self.values[i][j + 1] - 1] += 1
                block_count[self.values[i][j + 2] - 1] += 1

                block_count[self.values[i + 1][j] - 1] += 1
                block_count[self.values[i + 1][j + 1] - 1] += 1
                block_count[self.values[i + 1][j + 2] - 1] += 1

                block_count[self.values[i + 2][j] - 1] += 1
                block_count[self.values[i + 2][j + 1] - 1] += 1
                block_count[self.values[i + 2][j + 2] - 1] += 1

                for k in range(len(block_count)):
                    if block_count[k] == 1:
                        block_sum += (1 / Nd) / Nd
                block_count = np.zeros(Nd)

        # The fitness will be 1 only when the fitness of columns and blocks are 1
        if int(column_sum) == 1 and int(block_sum) == 1:
            fitness = 1.0
        else:
            fitness = column_sum * block_sum

        self.fitness = fitness
        return

    def mutate(self, mutation_rate, given):

        success = False
        if random.uniform(0, 1) < mutation_rate:
            while not success:
                row = random.randint(0, 8)

                # Select two column randomly and they can not be same
                from_column = random.randint(0, 8)
                to_column = random.randint(0, 8)
                while from_column == to_column:
                    from_column = random.randint(0, 8)
                    to_column = random.randint(0, 8)

                    # If numbers in these positions are both 0 in the given puzzle, they can swap
                if given.values[row][from_column] == 0 and given.values[row][to_column] == 0:
                    # we are not causing a duplicate in the row/column/block.
                    if not given.is_column_duplicate(to_column,
                                                     self.values[row][from_column]) and not given.is_column_duplicate(
                        from_column, self.values[row][to_column]) and not given.is_block_duplicate(row, to_column,
                                                                                                   self.values[row][
                                                                                                       from_column]) and not given.is_block_duplicate(
                        row, from_column, self.values[row][to_column]):
                        # Swap numbers.
                        temp = self.values[row][to_column]
                        self.values[row][to_column] = self.values[row][from_column]
                        self.values[row][from_column] = temp
                        success = True

        return success


class Fixed(Candidate):

    def __init__(self, values):
        self.values = values
        return

    def is_row_duplicate(self, row, value):
        """ Check duplicate in a row. """
        for column in range(0, Nd):
            if self.values[row][column] == value:
                return True
        return False

    def is_column_duplicate(self, column, value):
        """ Check duplicate in a column. """
        for row in range(0, Nd):
            if self.values[row][column] == value:
                return True
        return False

    def is_block_duplicate(self, row, column, value):
        """ Check duplicate in a block. """
        i = 3 * (int(row / 3))
        j = 3 * (int(column / 3))

        if ((self.values[i][j] == value)
                or (self.values[i][j + 1] == value)
                or (self.values[i][j + 2] == value)
                or (self.values[i + 1][j] == value)
                or (self.values[i + 1][j + 1] == value)
                or (self.values[i + 1][j + 2] == value)
                or (self.values[i + 2][j] == value)
                or (self.values[i + 2][j + 1] == value)
                or (self.values[i + 2][j + 2] == value)):
            return True
        else:
            return False


class Tournament(object):
    """
    Select a parent for crossover. The selection rate allow us to select a weaker parent.
    """

    def __init__(self):
        return

    def compete(self, candidates):
        """ Pick 2 random candidates"""
        c1 = candidates[random.randint(0, len(candidates) - 1)]
        c2 = candidates[random.randint(0, len(candidates) - 1)]
        f1 = c1.fitness
        f2 = c2.fitness

        # Compete
        if f1 > f2:
            better = c1
            weaker = c2
        else:
            better = c2
            weaker = c1

        selection_rate = 0.80

        # If the number is smaller than selection rate, we choose the better one; otherwise, the weaker one
        if random.uniform(0, 1) < selection_rate:
            return better
        else:
            return weaker


class Crossover(object):
    def __init__(self):
        return

    def crossover(self, p1, p2, given, crossover_rate):
        c1 = Candidate()
        c2 = Candidate()
        c1.values = np.copy(p1.values)
        c2.values = np.copy(p2.values)

        # Mutate when the random number less than the crossover rate.
        if random.uniform(0, 1) < crossover_rate:
            num_row = random.randint(0, len(given) - 1)
            row = given[num_row]
            crossover_mask = np.zeros(len(row))
            for i in range(len(row)):
                if row[i] == 0:
                    crossover_mask[i] = 1
            crossover_sum = sum(crossover_mask)

            flag = 0
            while flag < crossover_sum / 2:
                num = random.randint(0, len(row) - 1)
                while crossover_mask[num] == 0:
                    num = random.randint(0, len(row) - 1)
                c1.values[num_row][num], c2.values[num_row][num] = c2.values[num_row][num], c1.values[num_row][num]
                crossover_mask[num] = 0
                flag += 1
        return c1, c2


class Sudoku(object):
    """ Solves a given Sudoku puzzle using a genetic algorithm. """

    def __init__(self):
        self.given = None
        return

    def load(self, p):
        self.given = Fixed(p)
        return

    def solve(self):
        Numc = 1000  # The population size.
        Nume = int(0.05 * Numc)  # Number of elites which will be preserved.
        Numg = 10000  # Number of generations.
        Numm = 0  # Number of mutations.

        # Mutation parameters.
        mutation_rate = 0.1

        # Seed initial population.
        self.population = Population()
        print("create an initial population.")
        if self.population.seed(Numc, self.given) != 1:
            return -1, 1

        # For up to Numg generations
        stale = 0
        for generation in range(0, Numg):

            # Check for a solution.
            best_fitness = 0.0
            for c in range(0, Numc):
                fitness = self.population.candidates[c].fitness
                if fitness == 1:
                    print("Solution found at generation %d!" % generation)
                    return generation, self.population.candidates[c]

                # Find the best fitness
                if fitness > best_fitness:
                    best_fitness = fitness

            print("Generation:", generation, " Best fitness:", best_fitness)

            # Create the next generation.
            next_generation = []

            # Select elites and preserve them.
            self.population.sort()
            elites = []
            for e in range(0, Nume):
                elite = Candidate()
                elite.values = np.copy(self.population.candidates[e].values)
                elites.append(elite)

            # Create the rest of the candidates.
            for _ in range(Nume, Numc, 2):
                # Select 2 parents.
                t = Tournament()
                parent1 = t.compete(self.population.candidates)
                parent2 = t.compete(self.population.candidates)

                # Cross-over.
                cc = Crossover()
                child1, child2 = cc.crossover(parent1, parent2, self.given.values, crossover_rate=1.0)

                # Mutate child1.
                child1.update_fitness()
                success = child1.mutate(mutation_rate, self.given)
                child1.update_fitness()
                if success:
                    Numm += 1

                # Mutate child2.
                child2.update_fitness()
                success = child2.mutate(mutation_rate, self.given)
                child2.update_fitness()
                if success:
                    Numm += 1

                # Add children to new population.
                next_generation.append(child1)
                next_generation.append(child2)

            # Add elites to the population.
            # They will not have been affected by crossover or mutation.
            for e in range(0, Nume):
                next_generation.append(elites[e])

            # Select next generation.
            self.population.candidates = next_generation
            self.population.update_fitness()

            # Check for stale population.
            self.population.sort()
            if self.population.candidates[0].fitness != self.population.candidates[1].fitness:
                stale = 0
            else:
                stale += 1

            # If in 100 generations the fitness of the fittest two candidates do not change
            # we can assume the population has gone stale, so we generate new population
            if stale >= 100:
                print("The population has gone stale. Re-seeding...")
                self.population.seed(Numc, self.given)
                stale = 0
                mutation_rate = 0.06

        print("No solution found.")
        return -2, 1
