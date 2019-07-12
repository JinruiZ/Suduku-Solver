import numpy as np
import random
import operator
from past.builtins import range

random.seed()

Nd = 9  # Number of digits, this is a solver to solve a 9x9 Sudoku.

class Population(object):
    """ A set of candidate solutions to the Sudoku puzzle.
    These candidates are also known as the chromosomes in the population. """

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
                    if (given.values[row][column] == 0) and not (given.is_column_duplicate(column, value) or given.is_block_duplicate(row, column, value) or given.is_row_duplicate(row, value)):
                        helper.values[row][column].append(value)
                    elif given.values[row][column] != 0:
                        helper.values[row][column].append(given.values[row][column])
                        break

        # Seed a new population.
        for _ in range(0, Nc):
            g = Candidate()
            for i in range(0, Nd):  # New row in candidate.
                row = np.zeros(Nd)

                # Fill in the givens.
                for j in range(0, Nd):  # New column j value in row i.

                    # We can only fill the position where the number in it is 0 in given puzzle
                    if given.values[i][j] != 0:
                        row[j] = given.values[i][j]
                    elif given.values[i][j] == 0:
                        row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j]) - 1)]

                k = 0
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
        """ Update fitness of every candidate/chromosome. """
        for candidate in self.candidates:

            candidate.update_fitness()
        return

    def sort(self):
        """ Sort the population based on fitness. """
        self.candidates = sorted(self.candidates, key=operator.attrgetter('fitness'))
        return

class Candidate(object):
    """ A candidate solutions to the Sudoku puzzle. """

    def __init__(self):
        self.values = np.zeros((Nd, Nd))
        self.fitness = None
        return

    def update_fitness(self):
        """ The fitness of a candidate solution is determined by how close it is to being the actual solution to the puzzle. """

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
                    column_sum += (1/Nd)/Nd
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
                        block_sum += (1/Nd)/Nd
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
        if random.uniform(0, 1) < mutation_rate:  # Mutate.
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
                    if not given.is_column_duplicate(to_column, self.values[row][from_column]) and not given.is_column_duplicate(from_column, self.values[row][to_column]) and not given.is_block_duplicate(row, to_column, self.values[row][from_column]) and not given.is_block_duplicate(row, from_column, self.values[row][to_column]):
                        # Swap values.
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
        """ Check duplicate in a 3 x 3 block. """
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


class CycleCrossover(object):
    """ These function is trying to crossover two chromosome without causing new confication
    It is used here (see e.g. A. E. Eiben, J. E. Smith.
    Introduction to Evolutionary Computing. Springer, 2007). """

    def __init__(self):
        return

    def crossover(self, parent1, parent2, crossover_rate):
        """ Create two new child candidates by crossing over parent genes. """
        child1 = Candidate()
        child2 = Candidate()

        # Make a copy of the parent genes.
        child1.values = np.copy(parent1.values)
        child2.values = np.copy(parent2.values)

        # Perform crossover.
        if random.uniform(0, 1) < crossover_rate:
            # Pick a crossover point. Crossover must have at least 1 row (and at most Nd-1) rows.
            crossover_point1 = random.randint(0, 8)
            crossover_point2 = random.randint(1, 9)
            while crossover_point1 == crossover_point2:
                crossover_point1 = random.randint(0, 8)
                crossover_point2 = random.randint(1, 9)

            if crossover_point1 > crossover_point2:
                temp = crossover_point1
                crossover_point1 = crossover_point2
                crossover_point2 = temp

            for i in range(crossover_point1, crossover_point2):
                child1.values[i], child2.values[i] = self.crossover_rows(child1.values[i], child2.values[i])

        return child1, child2

    def crossover_rows(self, row1, row2):
        child_row1 = np.zeros(Nd)
        child_row2 = np.zeros(Nd)

        remaining = range(1, Nd + 1)
        cycle = 0

        while (0 in child_row1) and (0 in child_row2):  # While child rows not complete...
            if cycle % 2 == 0:  # Even cycles.
                # Assign next unused value.
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row1[index]
                child_row2[index] = row2[index]
                next = row2[index]

                while next != start:  # While cycle not done
                    index = self.find_value(row1, next)
                    child_row1[index] = row1[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row2[index]
                    next = row2[index]

                cycle += 1

            else:  # Odd cycle - flip values.
                index = self.find_unused(row1, remaining)
                start = row1[index]
                remaining.remove(row1[index])
                child_row1[index] = row2[index]
                child_row2[index] = row1[index]
                next = row2[index]

                while next != start:  # While cycle not done
                    index = self.find_value(row1, next)
                    child_row1[index] = row2[index]
                    remaining.remove(row1[index])
                    child_row2[index] = row1[index]
                    next = row2[index]

                cycle += 1

        return child_row1, child_row2

    def find_unused(self, parent_row, remaining):
        for i in range(0, len(parent_row)):
            if parent_row[i] in remaining:
                return i

    def find_value(self, parent_row, value):
        for i in range(0, len(parent_row)):
            if parent_row[i] == value:
                return i


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
        phi = 0
        sigma = 1
        mutation_rate = 0.06

        # Seed initial population.
        self.population = Population()
        print("create an initial population.")
        if self.population.seed(Numc, self.given) ==  1:
            pass
        else:
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

            # Create the next population.
            next_population = []

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
                cc = CycleCrossover()
                child1, child2 = cc.crossover(parent1, parent2, crossover_rate=1.0)

                # Mutate child1.
                child1.update_fitness()
                old_fitness = child1.fitness
                success = child1.mutate(mutation_rate, self.given)
                child1.update_fitness()
                if (success):
                    Numm += 1
                    if (child1.fitness > old_fitness):  # Used to calculate the relative success rate of mutations.
                        phi = phi + 1

                # Mutate child2.
                child2.update_fitness()
                old_fitness = child2.fitness
                success = child2.mutate(mutation_rate, self.given)
                child2.update_fitness()
                if (success):
                    Numm += 1
                    if (child2.fitness > old_fitness):  # Used to calculate the relative success rate of mutations.
                        phi = phi + 1

                # Add children to new population.
                next_population.append(child1)
                next_population.append(child2)

            # Add elites to the population.
            # They will not have been affected by crossover or mutation.
            for e in range(0, Nume):
                next_population.append(elites[e])

            # Select next generation.
            self.population.candidates = next_population
            self.population.update_fitness()

            # Dynamic mutation rate.
            # If the mutated child has better fitness than before, we consider it is a successful mutation
            # or it is a failed one.
            # Considering successful_mutations/total_mutations as the successful mutation rate
            # If it is bigger than a threshold (0.2 there), we can assume that more mutation can introduce
            # higher fitness to the whole population. So we increase the mutation rate slightly.
            # Otherwise, we decrease the mutation rate.
            if Numm == 0:
                phi = 0  # Avoid divide by zero.
            else:
                phi = phi / Numm

            if phi > 0.2:
                sigma = sigma / 0.998
            elif phi < 0.2:
                sigma = sigma * 0.998

            mutation_rate = abs(np.random.normal(loc=0.0, scale=sigma, size=None))

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
                sigma = 1
                phi = 0
                mutation_rate = 0.06

        print("No solution found.")
        return -2, 1
