# import numpy as np
import math
import random


def encode(ordered_list, mode):
    copia = alfabeto[:]
    result = 0
    multiplier = 1
    for i in ordered_list[:-1]:
        pos = copia.index(i)
        if mode == 'print':
            print(copia, end=', ')
            print(f"{result} + {pos} * {multiplier}", end='')
        result += pos * multiplier
        if mode == 'print':
            print(f" = {result}")
        multiplier = multiplier * len(copia)
        copia = copia[:pos] + copia[pos + 1:]
    return result


def decode(number, mode):
    copia = alfabeto[:]
    result = []
    for i in range(len(alfabeto)):
        mod = number % len(copia)
        if mode == 'print':
            print(f"{number} mod {len(copia)} = {mod}", end='')
        number = number // len(copia)
        result.append(copia.pop(mod))
        if mode == 'print':
            print(f", new value = {number}, {result}")
    return result


alfabeto = [i for i in range(5)]
possible = math.factorial(len(alfabeto))
bits = math.ceil(math.log2(possible))
print(f"alphabet = {alfabeto}")
print(f"possible sequences: {possible}")
print(f"best encoding uses {bits} bits")

orden = random.sample(alfabeto, len(alfabeto))
# orden = [0, 5, 2, 1, 3]
print(f"sequence = {orden}\n")
encoding = encode(orden, 'print')
print(f"encoding = {encoding} -> {bin(encoding)[2:]}\n")
print(f"decoding = {decode(encoding,'print')}")





