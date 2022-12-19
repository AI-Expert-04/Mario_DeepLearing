import random
import math
import numpy as np
import csv
import matplotlib.pyplot as plt
import time

MUTATION_RATE = 40
MUTATION_COUNT = 2
THRESHOLD = 35000
UNIFORMCROSSOVER_RATE = 0.5

csvfile = 'TSP.csv'


def read_csv(csvf):  # csv 파일 읽기, csv file read
    City = np.genfromtxt(open(csvf, "rb"), dtype=float, delimiter=",", skip_header=0)
    print(City)
    return City


cityCoordinates = read_csv(csvfile)
citySize = len(read_csv(csvfile))


class Genome():
    chromosomes = []
    fitness = 100000

    def __init__(self, numberOfchromosomes=None):
        if numberOfchromosomes is not None:
            self.chromosomes = list(range(numberOfchromosomes))
            randShuffle(self.chromosomes)


def randShuffle(listToShuffle):
    return random.shuffle(listToShuffle)


def init_population(size):
    initial_population = []
    for i in range(size):
        newGenome = Genome()
        newGenome.chromosomes = random.sample(range(1, citySize), citySize - 1)
        newGenome.chromosomes.insert(0, 0)
        newGenome.chromosomes.append(0)
        newGenome.fitness = Evaluate(newGenome.chromosomes)
        initial_population.append(newGenome)
    return initial_population


def Evaluate(chromosomes):
    Fitness = 0
    for i in range(len(chromosomes) - 1):
        p1 = cityCoordinates[chromosomes[i]]
        p2 = cityCoordinates[chromosomes[i + 1]]
        Fitness += Euclidean_distance(p1, p2)
    Fitness = np.round(Fitness, 2)
    return Fitness


def Euclidean_distance(x, y):
    dist = np.linalg.norm(np.array(x) - np.array(y))
    return dist


def findBestGenome(population):
    allFitness = [i.fitness for i in population]
    bestFitness = min(allFitness)
    return population[allFitness.index(bestFitness)]


# 선택 연산 _ Selection
'''
def roulette_wheel_selection(population):
    population_fitness = sum([chromosome.fitness for chromosome in population])
    chromosome_probabilities = [chromosome.fitness / population_fitness for chromosome in population]
    chromosome_probabilities = 1 - np.array(chromosome_probabilities)
    chromosome_probabilities /= chromosome_probabilities.sum()
    return np.random.choice(population, p=chromosome_probabilities)
'''


def TournamentSelection(population, k):
    select = [population[random.randrange(0, len(population))] for i in range(k)]
    bestGenome = findBestGenome(select)
    return bestGenome


def Reproduction(population):
    parent1 = TournamentSelection(population, 15).chromosomes
    parent2 = TournamentSelection(population, 15).chromosomes
    while parent1 == parent2:
        parent2 = TournamentSelection(population, 15).chromosomes

    return OrderCrossover(parent1, parent2)


def randRange(first, last):
    return random.randint(first, last)


# 교차 연산 _ Crossover
'''
def SinglePointCrossover(parent1, parent2) :
    child = Genome(None)
    child.chromosomes = []
    point = random.randint(0, len(parent1))
    child.chromosomes =parent1[:point] +parent2[point:]
    if random.randrange(0, 100) < MUTATION_RATE:
        child.chromosomes = SwapMutation(child.chromosomes)
    child.fitness = Evaluate(child.chromosomes)
    return child
def TwoPointCrossover (parent1, parent2) :
    child = Genome(None)
    child.chromosomes = []
    point1 = random.randrange(1,len(parent1) -1)
    while True :
        point2 = random.randrange(1, len(parent1)-1)
        if point1 != point2 :
            break
    child.chromosomes = parent1[:min(point1, point2)] + parent2[min(point1,point2):max(point1, point2)] + parent1[max(point1,point2):]
    if random.randrange(0, 100) < MUTATION_RATE:
        child.chromosomes = SwapMutation(child.chromosomes)
    child.fitness = Evaluate(child.chromosomes)
    return child
def PMXCrossover (parent1, parent2) :
    child = Genome(None)
    child.chromosomes = []
    size = len(parent1)
    p1, p2 = [0] * size, [0] * size
    for i in range(size) :
        p1[parent1[i]] = i
        p2[parent1[i]] = i
    point1 = random.randint(0,size)
    point2 = random.randint(0, size - 1)
    if point2 >= point1 :
        point2 += 1
    else :
        point1, point2 = point2, point1
    for i in range(point1, point2) :
        temp1 = parent1[i]
        temp2 = parent2[i]
        parent1[i], parent1[p1[temp2]] = temp2, temp1
        parent2[i], parent2[p2[temp1]] = temp1, temp2
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]
    child.chromosomes = parent1[:min(point1, point2)] + parent1[min(point1,point2):max(point1, point2)] + parent1[max(point1,point2):]
    if random.randrange(0, 100) < MUTATION_RATE:
        child.chromosomes = SwapMutation(child.chromosomes)
    child.fitness = Evaluate(child.chromosomes)
    return child
def UniformCrossover(parent1, parent2) :
    size = len(parent1)
    child = [-1] * size
    child[0], child[size - 1] = 0, 0
    for i in range(1, size - 1):
        if random.randrange(0, 100) < UNIFORMCROSSOVER_RATE:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    if random.randrange(0, 100) < MUTATION_RATE:
        child = SwapMutation(child)
    newGenome = Genome()
    newGenome.chromosomes = child
    newGenome.fitness = Evaluate(child)
    return newGenome
'''


def OrderCrossover(parent1, parent2):
    child = Genome(None)
    child.chromosomes = []
    firstIndex = randRange(0, len(parent1) - 1)
    secondIndex = randRange(firstIndex, len(parent1) - 1)
    innerSet = parent1[firstIndex:secondIndex]
    startSet = []
    endSet = []
    for _, value in enumerate([item for item in parent2 if item not in innerSet]):
        if len(startSet) < firstIndex:
            startSet.append(value)
        else:
            endSet.append(value)
    child.chromosomes = startSet + innerSet + endSet

    if random.randrange(0, 100) < MUTATION_RATE:
        child.chromosomes = InversionMutation(child.chromosomes)

    child.fitness = Evaluate(child.chromosomes)
    return child


# 변이 연산 _ Mutation
'''
def SwapMutation(chromo):
    for x in range(MUTATION_COUNT):
        p1, p2 = [random.randrange(1, len(chromo) - 1) for i in range(2)]
        while p1 == p2:
            p2 = random.randrange(1, len(chromo) - 1)
        log = chromo[p1]
        chromo[p1] = chromo[p2]
        chromo[p2] = log
    return chromo
def ScrambleMutation(chromo) :
    for x in range(MUTATION_COUNT):
        p1, p2 = [random.randrange(1, len(chromo) - 1) for i in range(2)]
        while p1 == p2 or p1 > p2:
            p1 = random.randint(0, len(chromo) - 1)
            p2 = random.randint(0, len(chromo) - 1)
        log = chromo[p1:p2]
        random.shuffle(log)
        chromo = chromo[:p1] + log +chromo[p2:]
    return chromo
'''


def InversionMutation(chromo):
    for x in range(MUTATION_COUNT):
        p1, p2 = [random.randrange(1, len(chromo) - 1) for i in range(2)]
        while p1 == p2 or p1 > p2:
            p1 = random.randint(0, len(chromo) - 1)
            p2 = random.randint(0, len(chromo) - 1)
        log = chromo[p1:p2]
        log = log[::-1]
        chromo = chromo[:p1] + log + chromo[p2:]
    return chromo


# 시각화 _ Visualization
def fitness_plot(generation, allBestFitness):
    plt.plot(range(0, generation), allBestFitness, c='blue')
    plt.xlabel('Generations')
    plt.ylabel('Best Fitness')
    plt.title('Fitness Function')
    plt.show()


def city_visualize(bestGenome, city):
    start = None
    for x, y in city:
        if start is None:  # 시작지점이면 표시
            start = city[0]
            plt.scatter(start[0], start[1], c="green", marker=">")
            plt.annotate("Start", (x + 2, y - 2), color='red')
        else:  # 시작지점 아니면
            plt.scatter(x, y, marker='.', s=10, c="black")

    # edge 표현을 위한 x, y 범위
    x_edge = [city[i][0] for i in bestGenome.chromosomes]
    y_edge = [city[i][1] for i in bestGenome.chromosomes]

    plt.plot(x_edge, y_edge, color="blue", linewidth=0.07, linestyle="-")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('City Edges')
    plt.show()


def GeneticAlgorithm(populationSize, Generation_Count):
    allBestFitness = []
    population = init_population(populationSize)
    generation = 0
    TotalBestFitness = 100000
    TotalBestPath = []

    start = time.time()

    while generation < Generation_Count:
        generation += 1

        for i in range(populationSize):
            population.append(Reproduction(population))

        for genom in population:
            if genom.fitness > THRESHOLD:
                population.remove(genom)

        averageFitness = round(np.sum([genom.fitness for genom in population]) / len(population), 2)
        bestGenome = findBestGenome(population)
        if bestGenome.fitness < TotalBestFitness:
            TotalBestFitness = bestGenome.fitness
            TotalBestPath = bestGenome.chromosomes
        print("\n" * 5)
        print("Generation: {0}\nPopulation Size: {1}\t Average Fitness: {2}\nBest Fitness: {3}"
              .format(generation, len(population), averageFitness,
                      bestGenome.fitness))

        allBestFitness.append(bestGenome.fitness)

    print("\nTotal Best Fitness : ", TotalBestFitness)

    end = time.time()

    print("Total time : ", end - start)  # 소요 시간 표기, Working Time

    # 시각화
    fitness_plot(generation, allBestFitness)
    city_visualize(bestGenome, cityCoordinates)

    # csv 파일 경로 저장
    f = open('temp.csv', 'w', newline='')
    wr = csv.writer(f)
    for i in range(0, citySize):
        wr.writerow([TotalBestPath[i]])
    f.close()


if __name__ == "__main__":
    GeneticAlgorithm(populationSize=15, Generation_Count=5000)  # Population size, Generation Count 입력