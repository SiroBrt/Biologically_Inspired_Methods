# Binary genome

# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import random
import bisect
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

POPULATION_SIZE = 100
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.7
GENERATIONS = 20
SEED = 1


class Pindividual:
    def __init__(self, chromosome):
        length = len(list(chromosome))
        if length > GENES:
            # print("WARNING: truncating chromosome")
            ch = list(chromosome)[:GENES]
        elif length < GENES:
            # print("WARNING: chromosome too short")
            ch = list(chromosome) + [random.randint(0, 1) for i in range(GENES - length)]
        else:
            ch = list(chromosome)
        self.chromosome = [bool(i) for i in ch]
        self.fitness = 0

    def __str__(self):
        # return f"fitness: {self.calculate_fitness()}"
        string = ""
        for i in self.chromosome:
            if i:
                string = string + "1"
            else:
                string = string + "0"
        return f"{string}, fitness: {self.calculate_fitness()}, sum: {sum(self.chromosome)}"

    def copy(self):
        return Pindividual(self.chromosome)

    def calculate_fitness(self):
        self.fitness = linreg(mask_X(X, self.chromosome), y)
        return self.fitness

    def __lt__(self, other):
        if self.fitness == 0:
            self.calculate_fitness()
        if other.fitness == 0:
            other.calculate_fitness()

        return self.fitness < other.fitness

    def crossover(self, partner):
        midpoint = random.randint(0, len(self.chromosome))
        child1 = self.chromosome[:midpoint] + partner.chromosome[midpoint:]
        child2 = self.chromosome[midpoint:] + partner.chromosome[:midpoint]
        return Pindividual(child1), Pindividual(child2)

    def punctual_mutation(self, prob):
        for i in range(len(self.chromosome)):
            if random.random() < prob:
                self.chromosome[i] = not self.chromosome[i]
        self.fitness = 0

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
        self.fitness = 0


def mask_X(X, mask):
    return X.iloc[:, mask]


def random_mask(n):
    return np.random.binomial(1, 0.5, n).astype(bool)


def linreg(M, v):
    X_train, X_test, y_train, y_test = train_test_split(M, v, test_size=0.2, random_state=0)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_ = lr.predict(X_test)
    return mean_squared_error(y_test, y_)


def direct_selection(population, ark_capacity, best):
    # take best
    saved = []
    for i in population:
        if len(saved) < ark_capacity:
            bisect.insort(saved, i)
            continue
        if i < saved[-1]:
            bisect.insort(saved, i)
        if len(saved) > ark_capacity:
            saved.pop()
    print(saved[0])
    best.append(saved[0].fitness)

    # keep saved
    new_popu = []
    for i in saved:
        elem = i.copy()
        new_popu.append(elem)

    # aber cojan
    crossings = int((POPULATION_SIZE - ark_capacity) * CROSSOVER_RATE / 2)
    for i in range(crossings):
        # from OG (exploitation)
        # parent1 = random.randint(0, ark_capacity - 1)
        # parent2 = random.randint(0, ark_capacity - 1)
        # child1, child2 = saved[parent1].crossover(saved[parent2])
        # from new (exploration)
        parent1 = random.randint(0, len(new_popu) - 1)
        parent2 = random.randint(0, len(new_popu) - 1)
        child1, child2 = new_popu[parent1].crossover(new_popu[parent2])

        new_popu.append(child1)
        new_popu.append(child2)

    # fill with mutants
    while len(new_popu) < POPULATION_SIZE:
        pos = random.randint(0, len(new_popu) - 1)
        elem = new_popu[pos].copy()
        elem.punctual_mutation(MUTATION_RATE)
        new_popu.append(elem)

    return new_popu


def tournament(population, selection_size, best):
    aux_popu = []
    for i in range(POPULATION_SIZE):
        selected = []
        for i in range(selection_size):
            selected.append(population[random.randint(0, len(population) - 1)])
        aux_popu.append(min(selected).copy())
    new_popu = []
    i = 0
    while len(new_popu) < POPULATION_SIZE:
        ch1, ch2 = aux_popu[i].crossover(aux_popu[i + 1])
        new_popu.append(ch1)
        new_popu.append(ch2)
        i = i + 2
    fittest = min(new_popu)
    print(fittest)
    best.append(fittest.fitness)
    return new_popu


def evolve(gen, selection, param, best):
    t1 = time.time()
    popu = [Pindividual([]) for _ in range(POPULATION_SIZE)]
    for i in range(gen):
        print(f"gen {i} best: ", end='')
        popu = selection(popu, param, best)

    print(f"gen {gen} best: {min(popu)}")
    t2 = time.time()
    print(f"{t2-t1}s")
    return popu


print("Reading")
data = pd.read_excel('data.xlsx')
data = data.dropna(axis=0)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
GENES = X.shape[1]
print("Done")


random.seed(SEED)
state = random.getstate()
best1 = []
evolved = evolve(GENERATIONS, tournament, 5, best1)
plt.semilogy(best1)
random.setstate(state)
best2 = []
evolved = evolve(GENERATIONS, direct_selection, POPULATION_SIZE / 10, best2)
plt.semilogy(best2)
plt.show()
