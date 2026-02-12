import matplotlib.pyplot as plt
import string
import random
import pandas as pd


def mask_X(X, mask):
    return X


# --- CONFIGURATION ---
TARGET_PHRASE = "MENDEL AND DARWIN LINKED"
POPULATION_SIZE = 200
MUTATION_RATE = 0.01
GENES = string.ascii_uppercase + " "


class Pindividual:
    def __init__(self, chromosome):
        self.chromosome = list(chromosome)  # Make mutable
        self.fitness = 0
        # We calculate fitness immediately upon birth
        self.calculate_fitness()

    def calculate_fitness(self):
        score = 0
        for i in range(len(self.chromosome)):
            if self.chromosome[i] == TARGET_PHRASE[i]:
                score += 1
        self.fitness = score
        return score

    def crossover(self, partner):
        midpoint = random.randint(0, len(self.chromosome))
        child1 = self.chromosome[:midpoint] + partner.chromosome[midpoint:]
        child2 = self.chromosome[midpoint:] + partner.chromosome[:midpoint]
        return Pindividual(child1), Pindividual(child2)
