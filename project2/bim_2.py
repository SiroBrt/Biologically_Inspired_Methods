import numpy as np
import pandas as pd
import sympy as sp
import pygraphviz as pgv
import random
import bisect

import deap
import operator

from deap import base
from deap import creator
from deap import gp
from deap import tools

MUTATION_RATE = 0.05
POPULATION_SIZE = 100
CROSSOVER_RATE = 0.7

class Pindividual:
    def __init__(self, expr):
        self.expr = expr
        self.tree = gp.PrimitiveTree(expr)
        self.function = gp.compile(self.tree,pset)
        self.fitness = 0

    def __str__(self):
        return str(self.tree)
    
    def print_tree(self,file):
        nodes, edges, labels = gp.graph(self.expr)
        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")
        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]
        g.draw(file)

    def copy(self):
        return Pindividual(self.expr)

    def calculate_fitness(self):
        try:
            y_ = np.array([self.function(x_) for x_ in x])
            self.fitness = np.mean(np.abs(y_ - y))
        except Exception:
            print(f"\nupsi: {self}\n")
            self.fitness = -1
        return self.fitness
    

    def __lt__(self, other):
        if self.fitness == 0:
            self.calculate_fitness()
        if other.fitness == 0:
            other.calculate_fitness()

        return self.fitness < other.fitness

    def crossover(self, partner):
        expr1, expr2 = gp.cxOnePoint(self.tree, partner.tree)
        return Pindividual(expr1), Pindividual(expr2)

    def mut_prune(self):
        return Pindividual(list(list(gp.mutShrink(self.tree))[0]))

    # keeps structure but changes some leaves and nodes
    def mut_reroll(self):
        # change some leaves
        changes = int(1+MUTATION_RATE*len(list(self.tree))*abs(random.gauss(0,1)))
        for i in range(changes):
            tree = gp.mutEphemeral(self.tree,"one")[0]
        # change some nodes
        changes = int(1+MUTATION_RATE*len(list(self.tree))*abs(random.gauss(0,1)))
        for i in range(changes):
            tree = gp.mutNodeReplacement(tree,pset)[0]
        return Pindividual(list(tree))

    def phenotypic_dist(self,other):
        val = np.mean([abs(self.function(i)-other.function(i)) for i in x])
        return val

def direct_selection(population, ark_capacity):
    # take best
    saved = []
    for i in population:
        # print(i)
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
        elem.mut_reroll()
        new_popu.append(elem)

    return new_popu


def evolve_var_mut(popu, threshold, selection, param, best):
    t1 = time.time()
    popu = selection(popu, param)
    t2 = time.time()
    best.append(min(popu))
    print(f"gen 1, time: {t2-t1}s")

    t1 = time.time()
    popu = selection(popu, param)
    t2 = time.time()
    best.append(min(popu))
    print(f"gen 2, time: {t2-t1}s")

    improvement = (best[-1].fitness - best[-2].fitness) / best[-1].fitness

    i = 2
    while (improvement > threshold or improvement < 0):
        print(f"gen {i}, time: ", end='')
        t1 = time.time()
        popu = selection(popu, param)
        t2 = time.time()
        print(f"{t2-t1}s")
        best.append(min(popu))
        i += 1
        improvement = (best[-1].fitness - best[-2].fitness) / best[-1].fitness
    print(min(popu))
    return popu




data = pd.read_csv('pi.csv', header=None)

x = data.iloc[1:,0].values
y = data.iloc[1:,1].values
#
# def gauss_legendre(x):
#   return x / np.log(x)
#
# def li(x):
#   return sp.li(x).evalf()
#
# def fitness(f):
#   y_ = np.array([f(x_) for x_ in x])
#   return np.mean(np.abs(y_ - y))
#
# print(f'Fitness Gauss-Legendre: {fitness(gauss_legendre)}')
# print(f'Fitness Li: {fitness(li)}')


min_init_depth = 2
max_init_depth = 5
pset = gp.PrimitiveSet("MAIN", 1)
pset.renameArguments(ARG0="x")

def safe_div(a,b):
    if b == a:
        return 1
    if b == 0:
        return np.inf
    return a/b

def safe_log(a):
    if a == 0:
        return np.inf
    return np.log(float(abs(a)))

def safe_round(a):
    if abs(a) == np.inf:
        return a
    return round(a)

def safe_pow(a,b):
    # no complex numbers
    if a < 0 and (int(b)-b!=0):
        result = (-a)**b
        return result
    return a**b


pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safe_div, 2)
pset.addPrimitive(safe_pow, 2)
pset.addPrimitive(safe_log, 1)
pset.addPrimitive(safe_round, 1)

def efimeros():
    p = random.random()
    threshold = 0.1 # for constants
    constants = [np.pi,np.e]
    num = len(constants)
    if p<1-threshold:
        return round(random.uniform(-10, 10),1)
    for i in range(num):
        if p<1-threshold*(num-i-1)/num:
            return constants[i]

pset.addEphemeralConstant("num",efimeros)

random.seed(1)
population = []
for i in range(POPULATION_SIZE):
    population.append(Pindividual(gp.genHalfAndHalf(pset, min_=min_init_depth, max_=max_init_depth)))

direct_selection(population,10)
# population[1].print_tree(f"tree.png")
# print(safe_round(np.inf))
# evolved = evolve_var_mut(population, 0.0001, direct_selection, int(POPULATION_SIZE / 10), best_direct)

# for i in range(10):
#     population[i].print_tree(f"tree{i}.png")


