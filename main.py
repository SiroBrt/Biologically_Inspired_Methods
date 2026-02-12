# import matplotlib.pyplot as plt
import random
# import pandas as pd
import numpy as np


def mask_X(X, mask):
    return X


# --- CONFIGURATION ---
POPULATION_SIZE = 200
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.7
GENES = 23


class Pindividual:
    def __init__(self, chromosome):
        self.chromosome = list(chromosome)
        self.fitness = 0
        # self.calculate_fitness()

    def __str__(self):
        return f"{self.chromosome}, fitness: {self.calculate_fitness()}"

    def calculate_fitness(self):
        self.fitness = sum(self.chromosome)
        return self.fitness

    def crossover(self, partner):
        midpoint = random.randint(0, len(self.chromosome))
        child1 = self.chromosome[:midpoint] + partner.chromosome[midpoint:]
        child2 = self.chromosome[midpoint:] + partner.chromosome[:midpoint]
        return Pindividual(child1), Pindividual(child2)

    def punctual_mutation(self, prob):
        for i in range(len(self.chromosome)):
            if random.random() < prob:
                self.chromosome[i] = 1 - self.chromosome[i]

    def transposition_mutation(self, prob):
        if random.random() > prob:
            return
        MAXSIZE = np.floor(GENES / 4)
        size = random.randint(1, MAXSIZE)
        place1 = random.randint(0, GENES - size - 1)
        place2 = place1
        while (abs(place1 - place2) < size):
            place2 = random.randint(0, GENES - size - 1)
        if place1 > place2:
            aux = place2
            place2 = place1
            place1 = aux
        print(size, place1, place2)
        interposon1 = self.chromosome[place1:place1 + size]
        interposon2 = self.chromosome[place2:place2 + size]
        self.chromosome = self.chromosome[:place1] + interposon2 + self.chromosome[place1 + size:place2] + interposon1 + self.chromosome[place2 + size:]


yo = Pindividual([0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1])
# yo = Pindividual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
print(yo)
yo.transposition_mutation(1)
print(yo)
# for i in range(20):
#     print(random.randint(0, GENES - 1))

