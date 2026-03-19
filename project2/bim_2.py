import numpy as np
import pandas as pd
import sympy as sp
import pygraphviz as pgv
import random

import deap
import operator

from deap import base
from deap import creator
from deap import gp
from deap import tools

MUTATION_RATE = 0.05
POPULATION_SIZE = 100

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
        y_ = np.array([self.function(x_) for x_ in x])
        self.fitness = np.mean(np.abs(y_ - y))
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
        changes = int(1+MUTATION_RATE*len(list(self.tree))*abs(random.gauss(0,2)))
        for i in range(changes):
            tree = gp.mutEphemeral(self.tree,"one")[0]
        # change some nodes
        changes = int(1+MUTATION_RATE*len(list(self.tree))*abs(random.gauss(0,2)))
        for i in range(changes):
            tree = gp.mutNodeReplacement(tree,pset)[0]
        return Pindividual(list(tree))

    def phenotypic_dist(self,other):
        val = np.mean([abs(self.function(i)-other.function(i)) for i in x])
        return val



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

def div(a,b):
    if b == a:
        return 1
    if b == 0:
        return np.inf
    return a/b

pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(div, 2)
pset.addPrimitive(operator.pow, 2)
pset.addPrimitive(np.log, 1)

def efimeros():
    p = random.random()
    threshold = 0.05 # for constants
    constants = [np.pi,np.e]
    num = len(constants)
    if p<1-threshold:
        return round(random.uniform(-10, 10),1)
    for i in range(num):
        if p<1-threshold*(num-i)/num:
            return constants[i]

pset.addEphemeralConstant("num",efimeros)

random.seed(1)
population = []
for i in range(POPULATION_SIZE):
    population.append(Pindividual(gp.genHalfAndHalf(pset, min_=min_init_depth, max_=max_init_depth)))
for i in range(10):
    population[i].print_tree(f"tree{i}.png")

print(population[0].phenotypic_dist(population[2]))

