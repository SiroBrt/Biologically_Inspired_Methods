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

    # def calculate_fitness(self):
    #     self.fitness = 0

    def __lt__(self, other):
        if self.fitness == 0:
            self.calculate_fitness()
        if other.fitness == 0:
            other.calculate_fitness()

        return self.fitness < other.fitness

    def crossover(self, partner):
        expr1, expr2 = gp.cxOnePoint(self.tree, partner.tree)
        return Pindividual(expr1), Pindividual(expr2)

    def punctual_mutation(self, prob):
        p = random.random()




# data = pd.read_csv('pi.csv', header=None)
#
# x = data.iloc[1:,0].values
# y = data.iloc[1:,1].values
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

def div(a,b):
    if b == 0:
        return np.inf
    return a/b

min_init_depth = 3
max_init_depth = 3
pset = gp.PrimitiveSet("MAIN", 1)
pset.renameArguments(ARG0="x")

pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.pow, 2)
pset.addPrimitive(div, 2)
pset.addPrimitive(np.log, 1)

for i in range(10): pset.addTerminal(i)
pset.addTerminal(np.e)
pset.addTerminal(np.pi)


tree1 = Pindividual(gp.genHalfAndHalf(pset, min_=min_init_depth, max_=max_init_depth))
tree1.print_tree("tree1.png")
print(tree1)

