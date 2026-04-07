
import pandas as pd
import numpy as np
import pygraphviz as pgv
import array
import random
import operator

from deap import base
from deap import creator
from deap import tools
from deap import gp

MUTATION_RATE = 0.05
POPULATION_SIZE = 50
CROSSOVER_RATE = 0.7
min_init_depth = 2
max_init_depth = 5

random.seed(10)


#     def phenotypic_dist(self,other):
#         val = np.mean([abs(self.function(i)-other.function(i)) for i in x])
#         return val
#

# def evolve_var_mut(popu, threshold, selection, param, best):
#     t1 = time.time()
#     popu = selection(popu, param)
#     t2 = time.time()
#     best.append(min(popu))
#     print(f"gen 1, time: {t2-t1}s")
#
#     t1 = time.time()
#     popu = selection(popu, param)
#     t2 = time.time()
#     best.append(min(popu))
#     print(f"gen 2, time: {t2-t1}s")
#
#     improvement = (best[-1].fitness - best[-2].fitness) / best[-1].fitness
#
#     i = 2
#     while (improvement > threshold or improvement < 0):
#         print(f"gen {i}, time: ", end='')
#         t1 = time.time()
#         popu = selection(popu, param)
#         t2 = time.time()
#         print(f"{t2-t1}s")
#         best.append(min(popu))
#         i += 1
#         improvement = (best[-1].fitness - best[-2].fitness) / best[-1].fitness
#     print(min(popu))
#     return popu



# create pset
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

# deap things
toolbox = base.Toolbox()
toolbox.register("compile", gp.compile, pset=pset)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree,fitness=creator.FitnessMin, prim_set=pset)
toolbox.register("expr", gp.genHalfAndHalf,pset=pset, min_=min_init_depth, max_=max_init_depth)
toolbox.register("individual", tools.initIterate,container=creator.Individual,generator=toolbox.expr)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)



# evaluate
data = pd.read_csv('pi.csv', header=None)

x = data.iloc[1:,0].values
y = data.iloc[1:,1].values

def calc_fitness(ind):
    f = toolbox.compile(expr=ind)
    try:
        y_ = np.array([f(x_) for x_ in x])
        fit = np.mean((y_ - y)**2)
    except Exception:
        ind.fitness.values = tuple([np.inf])
        return tuple([np.inf])
    ind.fitness.values =  tuple([fit])
    return tuple([fit])

toolbox.register("evaluate",calc_fitness)
toolbox.register("select", tools.selTournament, tournsize=5)
# toolbox.register("select", tools.selBest, k=int(POPULATION_SIZE/10))


#     def crossover(self, partner):
#         expr1, expr2 = gp.cxOnePoint(self.tree, partner.tree)
#         return Pindividual(expr1), Pindividual(expr2)
toolbox.register("mate", gp.cxOnePoint)


def random_mutation(ind):
    if random.random() < 0.5:
        gp.mutShrink(ind)
    else:
        # at least 1 change, more changes for bigger trees depending on MUTATION_RATE
        changes = int(1+MUTATION_RATE*len(ind)*abs(random.gauss(0,1)))
        for i in range(changes):
            ind = gp.mutNodeReplacement(ind,pset)[0]

toolbox.register("mutate", random_mutation)

hof = tools.HallOfFame(10)

popu = toolbox.population(n=POPULATION_SIZE)
fit = list(map(toolbox.evaluate, popu))
hof_popu = []
for i in range(POPULATION_SIZE):
    if fit[i] != np.inf:
        hof_popu.append(popu[i])
hof.update(hof_popu)

print("\nEvolving")
g = 0
while g < 2:
    # A new generation
    g += 1
    print(f"Gen {g}")
    offspring = toolbox.select(popu, POPULATION_SIZE)
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CROSSOVER_RATE:
            ch1, ch2 = toolbox.mate(child1, child2)

    for mutant in offspring:
        if random.random() < MUTATION_RATE:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    popu = tools.selBest(popu+offspring,POPULATION_SIZE)
    
    fit = list(map(calc_fitness, popu))
    hof_popu = []
    for i in range(POPULATION_SIZE):
        if fit[i] != np.inf:
            hof_popu.append(popu[i])
    hof.update(hof_popu)


def print_tree(ind,file):
    nodes, edges, labels = gp.graph(ind)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")
    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]
    g.draw(file)



ind = hof[0]

f = toolbox.compile(expr=ind)
y_ = np.array([f(x_) for x_ in x])
fit = np.mean((y_ - y)**2)

print([f(x_) for x_ in list(np.array(range(2,9999)))])

# print_tree(ind,f"tree.png")
print(ind)
print(ind.fitness.values[0])
print(np.nan)



# for i in range(len(hof)):
#     print(hof[i].fitness,hof[i])
#     # print_tree(hof[i],f"tree{i}.png")

