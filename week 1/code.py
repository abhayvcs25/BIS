import random

START = (0, 0)
GOAL = (9, 9)

OBSTACLES = {
    (1, 1), (2, 3), (2, 4),
    (4, 5), (5, 5), (6, 5),
    (7, 2), (8, 2)
}

POPULATION_SIZE = 40
GENERATIONS = 100
MUTATION_RATE = 0.1
PATH_LENGTH = 15


def create_path():
    path = []
    x, y = START

    for _ in range(PATH_LENGTH):
        move = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])

        x = max(0, min(9, x + move[0]))
        y = max(0, min(9, y + move[1]))

        path.append((x, y))

    return path


def fitness(path):
    score = 0

    for point in path:
        if point in OBSTACLES:
            score -= 100

    last = path[-1]

    distance = abs(GOAL[0] - last[0]) + abs(GOAL[1] - last[1])

    score -= distance

    if GOAL in path:
        score += 200

    return score


def selection(population):
    population.sort(key=fitness, reverse=True)
    return population[:5]


def crossover(parent1, parent2):
    point = random.randint(1, PATH_LENGTH - 1)

    child = parent1[:point] + parent2[point:]

    return child


def mutation(path):
    for i in range(len(path)):
        if random.random() < MUTATION_RATE:
            x, y = path[i]

            move = random.choice([
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1)
            ])

            x = max(0, min(9, x + move[0]))
            y = max(0, min(9, y + move[1]))

            path[i] = (x, y)

    return path


population = [create_path() for _ in range(POPULATION_SIZE)]

for generation in range(GENERATIONS):

    parents = selection(population)

    new_population = parents.copy()

    while len(new_population) < POPULATION_SIZE:
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)

        child = crossover(parent1, parent2)

        child = mutation(child)

        new_population.append(child)

    population = new_population

best_path = max(population, key=fitness)

print("Best Path:")
print(START)

for point in best_path:
    print("↓", point)

print("Goal:", GOAL)
print("Fitness:", fitness(best_path))