import numpy as np
import random
from past.builtins import range

random.seed()

Nd = 9  # Number of digits (in the case of standard Sudoku puzzles, this is 9x9).


class Particle(object):
    def __init__(self, valid, given, weights, lazy_threshold):
        self.weights = weights
        self.valid = valid
        self.lazy_threshold = lazy_threshold
        self.given = given
        self.position = np.zeros((Nd, Nd))
        self.rand_position()
        self.valid = np.zeros((Nd, Nd))

        self.fitness = 0
        self.personal_best_fitness = 0
        self.global_best_fitness = 0
        self.lazy = 0

        self.personal_best_position = self.position.copy()
        self.global_best_position = np.zeros((Nd, Nd))
        self.velocity = np.zeros((Nd, Nd))
        for row in range(Nd):
            for column in range(Nd):
                if given[row][column] == 0:
                    self.velocity[row][column] = random.uniform(-2, 2)
        self.update_fitness()

    def mutate(self, mutate_rate):
        for row in range(len(self.position)):

            if random.uniform(0, 1) < mutate_rate:
                from_column = random.randint(0, 8)
                to_column = random.randint(0, 8)
                while from_column == to_column or self.given[row][from_column] != 0 or self.given[row][to_column] != 0:
                    from_column = random.randint(0, 8)
                    to_column = random.randint(0, 8)

                self.position[row][from_column], self.position[row][to_column] = self.position[row][to_column], \
                                                                                 self.position[row][from_column]

    def rand_position(self):
        for i in range(0, Nd):  # New row in candidate.

            # Fill in the givens.
            for j in range(0, Nd):  # New column j value in row i.

                # If value is already given, don't change it.
                if self.given[i][j] != 0:
                    self.position[i][j] = self.given[i][j]
                # Fill in the gaps using the helper board.
                elif self.given[i][j] == 0:
                    self.position[i][j] = self.valid[i][j][random.randint(0, len(self.valid[i][j]) - 1)]

    def update_lazy(self):
        if self.fitness == self.global_best_fitness:
            self.lazy += 1
            if self.lazy >= self.lazy_threshold:
                self.rand_position()
                self.personal_best_position = self.position.copy()
                self.update_fitness()
        else:
            self.lazy = 0

    def update_fitness(self):

        # Calculate overall fitness.
        column_count = np.zeros(Nd)
        block_count = np.zeros(Nd)
        column_sum = 0
        block_sum = 0

        self.position = self.position.astype(int)
        for j in range(0, Nd):
            for i in range(0, Nd):
                column_count[self.position[i][j] - 1] += 1

            for k in range(len(column_count)):
                if column_count[k] == 1:
                    column_sum += (1 / Nd) / Nd
            column_count = np.zeros(Nd)

        # For each block...
        for i in range(0, Nd, 3):
            for j in range(0, Nd, 3):
                block_count[self.position[i][j] - 1] += 1
                block_count[self.position[i][j + 1] - 1] += 1
                block_count[self.position[i][j + 2] - 1] += 1

                block_count[self.position[i + 1][j] - 1] += 1
                block_count[self.position[i + 1][j + 1] - 1] += 1
                block_count[self.position[i + 1][j + 2] - 1] += 1

                block_count[self.position[i + 2][j] - 1] += 1
                block_count[self.position[i + 2][j + 1] - 1] += 1
                block_count[self.position[i + 2][j + 2] - 1] += 1

                for k in range(len(block_count)):
                    if block_count[k] == 1:
                        block_sum += (1 / Nd) / Nd
                block_count = np.zeros(Nd)

        if int(column_sum) == 1 and int(block_sum) == 1:
            fitness = 1.0
        else:
            fitness = column_sum * block_sum

        self.fitness = fitness

        # Sync best position and fitness.

        if self.fitness > self.personal_best_fitness:
            self.personal_best_fitness = self.fitness
            self.personal_best_position = self.position.copy()
        if self.fitness > self.global_best_fitness:
            self.global_best_fitness = self.fitness
            self.global_best_position = self.position.copy()
        return

    def update_velocity(self):
        for i in range(Nd):
            for j in range(Nd):
                t1 = self.weights[0] * self.velocity[i][j]
                t2 = self.weights[1] * random.uniform(0, 1) * (self.personal_best_position[i][j] - self.given[i][j])
                t3 = self.weights[2] * random.uniform(0, 1) * (self.global_best_position[i][j] - self.given[i][j])
                self.velocity[i][j] = t1 + t2 + t3
                if self.velocity[i][j] > 3:
                    self.velocity[i][j] = -3
                if self.velocity[i][j] < -3:
                    self.velocity[i][j] = 3

    def update_position(self):
        for i in range(Nd):
            for j in range(Nd):
                self.position[i][j] = self.position[i][j] + self.velocity[i][j]
                if self.position[i][j] >= 9.5:
                    self.position[i][j] = round(self.position[i][j]) - 9
                if self.position[i][j] < .5:
                    self.position[i][j] = round(self.position[i][j]) + 9
        self.mutate(0.5)


class Swarm(object):
    globalbestvalid = None

    def __init__(self, given, weights, lazy_threshold):
        self.nparticles = 500
        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = 0
        self.step_best = 0

        valid = self.get_valid(given)
        for _ in range(self.nparticles):
            p = Particle(valid, given, weights, lazy_threshold)
            self.particles.append(p)

        self.update_global_best()

    def get_valid(self, given):

        # Determine the legal values that each square can take.
        helper = [[[] for j in range(0, Nd)] for i in range(0, Nd)]
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

        return helper

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

    def update_global_best(self):
        best = self.global_best_fitness
        self.step_best = 0
        for p in self.particles:
            if p.fitness > self.step_best:
                self.step_best = p.fitness
            if p.global_best_fitness > best:
                best = p.global_best_fitness
                self.global_best_position = p.global_best_position

        self.global_best_fitness = best
        for p in self.particles:
            p.global_best_fitness = best
            p.global_best_position = self.global_best_position

    def optimize(self):
        for p in self.particles:
            p.update_velocity()
            p.update_position()
            p.update_fitness()
            p.update_lazy()
        self.update_global_best()


class Sudoku(object):
    """ Solves a given Sudoku puzzle using a Particle Swarm Optimization Algorithm. """

    def __init__(self):
        self.given = None
        return

    def load(self, given):
        self.given = given
        return

    def solve(self):
        Ns = 2000  # Number of the moving steps

        s = Swarm(self.given, [.3, .7, .5], 10)
        print("Generate particle swarm.")

        for step in range(Ns):
            if s.global_best_fitness == 1:
                return step, s
            else:
                s.optimize()
                print("Step:", step, " Best fitness:", s.step_best)

        return -1, 1
