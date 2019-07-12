import numpy as np
import random
import math

random.seed()

Nd = 9  # Number of digits (in the case of standard Sudoku puzzles, this is 9x9).


class Material(object):

    def __init__(self, temp_min, temp_max, alpha):
        self.state = np.zeros((Nd, Nd))
        self.fitness = 0
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.temp = temp_max
        self.alpha = alpha

    def rand_state(self, given):

        # Determine the legal values that each square can take.
        helper = [[[] for _ in range(0, Nd)] for _ in range(0, Nd)]
        for row in range(0, Nd):
            for column in range(0, Nd):
                for value in range(1, 10):
                    if given[row][column] == 0 and not (
                            self.is_column_duplicate(given, column, value) or self.is_block_duplicate(given, row,
                                                                                                      column,
                                                                                                      value) or self.is_row_duplicate(
                            given, row, value)):
                        # Value is available.
                        helper[row][column].append(value)
                    elif given[row][column] != 0:
                        # Given/known value from file.
                        helper[row][column].append(given[row][column])
                        break

        # Seed a new state.
        g = np.zeros((Nd, Nd))
        for i in range(0, Nd):  # New row in candidate.
            row = np.zeros(Nd)

            # Fill in the givens.
            for j in range(0, Nd):  # New column j value in row i.

                # If value is already given, don't change it.
                if given[i][j] != 0:
                    row[j] = given[i][j]
                # Fill in the gaps using the helper board.
                elif given[i][j] == 0:
                    row[j] = helper[i][j][random.randint(0, len(helper[i][j]) - 1)]

            # If we don't have a valid board, then try again. max iteration 500,000
            # There must be no duplicates in the row.
            ii = 0
            while len(list(set(row))) != Nd:
                ii += 1
                if ii > 500000:
                    return 0
                for j in range(0, Nd):
                    if given[i][j] == 0:
                        row[j] = helper[i][j][random.randint(0, len(helper[i][j]) - 1)]
            g[i] = row
        # Compute the fitness of all candidates in the population.
        self.state = g
        self.update_fitness()
        return 1

    def is_row_duplicate(self, values, row, value):
        """ Check duplicate in a row. """
        for column in range(0, Nd):
            if values[row][column] == value:
                return True
        return False

    def is_column_duplicate(self, values, column, value):
        """ Check duplicate in a column. """
        for row in range(0, Nd):
            if values[row][column] == value:
                return True
        return False

    def is_block_duplicate(self, values, row, column, value):
        """ Check duplicate in a 3 x 3 block. """
        i = 3 * (int(row / 3))
        j = 3 * (int(column / 3))

        if ((values[i][j] == value)
                or (values[i][j + 1] == value)
                or (values[i][j + 2] == value)
                or (values[i + 1][j] == value)
                or (values[i + 1][j + 1] == value)
                or (values[i + 1][j + 2] == value)
                or (values[i + 2][j] == value)
                or (values[i + 2][j + 1] == value)
                or (values[i + 2][j + 2] == value)):
            return True
        else:
            return False

    def update_fitness(self):

        # Calculate overall fitness.
        column_count = np.zeros(Nd)
        block_count = np.zeros(Nd)
        column_sum = 0
        block_sum = 0

        self.state = self.state.astype(int)
        for j in range(0, Nd):
            for i in range(0, Nd):
                column_count[self.state[i][j] - 1] += 1

            for k in range(len(column_count)):
                if column_count[k] == 1:
                    column_sum += (1 / Nd) / Nd
            column_count = np.zeros(Nd)

        # For each block...
        for i in range(0, Nd, 3):
            for j in range(0, Nd, 3):
                block_count[self.state[i][j] - 1] += 1
                block_count[self.state[i][j + 1] - 1] += 1
                block_count[self.state[i][j + 2] - 1] += 1

                block_count[self.state[i + 1][j] - 1] += 1
                block_count[self.state[i + 1][j + 1] - 1] += 1
                block_count[self.state[i + 1][j + 2] - 1] += 1

                block_count[self.state[i + 2][j] - 1] += 1
                block_count[self.state[i + 2][j + 1] - 1] += 1
                block_count[self.state[i + 2][j + 2] - 1] += 1

                for k in range(len(block_count)):
                    if block_count[k] == 1:
                        block_sum += (1 / Nd) / Nd
                block_count = np.zeros(Nd)

        if int(column_sum) == 1 and int(block_sum) == 1:
            fitness = 1.0
        else:
            fitness = column_sum * block_sum

        self.fitness = fitness

        return

    def interfered_state(self, given):
        new_m = Material(self.temp_min, self.temp_max, self.alpha)
        new_m.state = self.state.copy()
        row = random.randint(0, len(given) - 1)

        r1 = random.randint(0, len(given[row]) - 1)
        r2 = random.randint(0, len(given[row]) - 1)
        while r1 == r2 or given[row][r1] != 0 or given[row][r2] != 0:
            r1 = random.randint(0, len(given[row]) - 1)
            r2 = random.randint(0, len(given[row]) - 1)

        new_m.state[row][r1], new_m.state[row][r2] = new_m.state[row][r2], new_m.state[row][r1]
        new_m.update_fitness()
        return new_m

    def update_state(self, new):
        self.state = new.state
        self.fitness = new.fitness

    def annealing(self, given):
        new_m = self.interfered_state(given)
        dE = self.fitness - new_m.fitness
        if dE <= 0:
            self.update_state(new_m)
        else:
            p = math.exp(-(dE / (.001 * self.temp)))
            if random.uniform(0, 1.0) < p:
                self.temp = self.alpha * self.temp
                if self.temp < self.temp_min:
                    return -1
                self.update_state(new_m)
        return self.temp


class Sudoku(object):
    """ Solves a given Sudoku puzzle using a simulated annealing algorithm. """

    def __init__(self):
        self.given = None
        return

    def load(self, given):
        self.given = given
        return

    def solve(self):
        max_step = 100000
        min_temp = 0.1
        max_temp = 10000
        alpha = 0.998

        m = Material(min_temp, max_temp, alpha)
        m.rand_state(self.given)
        print("Generate new state.")

        for step in range(max_step):
            if m.fitness == 1:
                return step, m
            temp = m.annealing(self.given)
            print("Step:", step, " Temperature:", temp)

        return -1, 1
