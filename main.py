# Binary genome

# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import random
import bisect
import pandas as pd
import numpy as np

POPULATION_SIZE = 100
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.7
SEED = 1
random.seed(SEED)


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
        return linreg(mask_X(X, self.chromosome), y)

    def __lt__(self, other):
        return self.calculate_fitness() < other.calculate_fitness()

    def crossover(self, partner):
        midpoint = random.randint(0, len(self.chromosome))
        child1 = self.chromosome[:midpoint] + partner.chromosome[midpoint:]
        child2 = self.chromosome[midpoint:] + partner.chromosome[:midpoint]
        return Pindividual(child1), Pindividual(child2)

    def punctual_mutation(self, prob):
        for i in range(len(self.chromosome)):
            if random.random() < prob:
                self.chromosome[i] = not self.chromosome[i]

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


# def tournament(population, prob)

def direct_selection(population, ark_capacity):
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

    # keep saved
    new_popu = []
    for i in saved:
        elem = i.copy()
        # elem.punctual_mutation(MUTATION_RATE)
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


def evolve(init, gen):
    popu = init[:]
    for i in range(gen):
        print(f"gen {i} best: {min(popu)}")
        popu = direct_selection(popu, POPULATION_SIZE / 10)
    print(f"gen {gen} best: {min(popu)}")
    return popu


print("Reading")
data = pd.read_excel('data.xlsx')
data = data.dropna(axis=0)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
GENES = X.shape[1]
print("Done")

popu = [Pindividual([]) for _ in range(POPULATION_SIZE)]
# for i in popu:
#     print(i)

evolved = evolve(popu, 20)
# for i in evolved:
#     print(i)
