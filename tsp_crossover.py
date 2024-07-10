import random


def cycle_crossover(parent1, parent2, cost_matrix=None):
    size = len(parent1)
    used = [False] * size
    child1 = [-1] * size
    child2 = [-1] * size

    while True:
        start = None
        for i in range(size):
            if not used[i]:
                start = i
                break
        if start is None:
            break

        index = start
        while True:
            child1[index] = parent2[index]
            child2[index] = parent1[index]
            used[index] = True
            index = parent1.index(parent2[index])
            if index == start:
                break

    for i in range(size):
        if child1[i] == -1:
            child1[i] = parent1[i]
        if child2[i] == -1:
            child2[i] = parent2[i]

    return child1, child2


def edge_recombination(parent1, parent2, cost_matrix=None):
    size = len(parent1)
    edge_map = {i: set() for i in range(size)}

    def add_edge(a, b):
        edge_map[a].add(b)
        edge_map[b].add(a)

    def get_neighbors(node):
        neighbors = list(edge_map[node])
        if neighbors:
            return random.choice(neighbors)
        return None

    for i in range(size):
        add_edge(parent1[i], parent1[(i + 1) % size])
        add_edge(parent2[i], parent2[(i + 1) % size])

    def generate_child(start):
        current = start
        child = [current]
        while len(child) < size:
            next_node = get_neighbors(current)
            if next_node is not None:
                child.append(next_node)
                edge_map[current].remove(next_node)
                edge_map[next_node].remove(current)
                current = next_node
            else:
                for i in range(size):
                    if i not in child:
                        child.append(i)
                        current = i
                        break
        return child

    child1 = generate_child(parent1[0])
    child2 = generate_child(parent2[0])

    return child1, child2


def order_crossover(parent1, parent2, cost_matrix=None):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child1 = [-1] * size
    child2 = [-1] * size

    child1[start : end + 1] = parent1[start : end + 1]
    child2[start : end + 1] = parent2[start : end + 1]

    fill_child(child1, parent2, end, size)
    fill_child(child2, parent1, end, size)

    return child1, child2


def fill_child(child, parent, end, size):
    current_index = (end + 1) % size
    for i in range(size):
        if parent[i] not in child:
            child[current_index] = parent[i]
            current_index = (current_index + 1) % size


def partially_matched_crossover(parent1, parent2, cost_matrix=None):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child1 = parent1[:]
    child2 = parent2[:]

    for i in range(start, end + 1):
        temp1 = child1[i]
        temp2 = child2[i]

        index1 = child1.index(temp2)
        index2 = child2.index(temp1)

        child1[i], child1[index1] = child1[index1], child1[i]
        child2[i], child2[index2] = child2[index2], child2[i]

    return child1, child2


def position_based_crossover(parent1, parent2, cost_matrix=None):
    size = len(parent1)
    positions = sorted(random.sample(range(size), size // 2))

    child1 = [-1] * size
    child2 = [-1] * size

    for pos in positions:
        child1[pos] = parent1[pos]
        child2[pos] = parent2[pos]

    fill_child_positions(child1, parent2, size)
    fill_child_positions(child2, parent1, size)

    return child1, child2


def fill_child_positions(child, parent, size):
    current_index = 0
    for i in range(size):
        if child[i] == -1:
            while parent[current_index] in child:
                current_index += 1
            child[i] = parent[current_index]


def uniform_order_based_crossover(parent1, parent2, cost_matrix=None, u=0.5):
    size = len(parent1)
    mask = [random.random() < u for _ in range(size)]

    child1 = [-1] * size
    child2 = [-1] * size

    for i in range(size):
        if mask[i]:
            child1[i] = parent1[i]
            child2[i] = parent2[i]

    fill_child_positions(child1, parent2, size)
    fill_child_positions(child2, parent1, size)

    return child1, child2


def non_wrapping_order_crossover(parent1, parent2, cost_matrix=None):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child1 = [-1] * size
    child2 = [-1] * size

    child1[start : end + 1] = parent1[start : end + 1]
    child2[start : end + 1] = parent2[start : end + 1]

    fill_child_non_wrapping(child1, parent2, start, end, size)
    fill_child_non_wrapping(child2, parent1, start, end, size)

    return child1, child2


def fill_child_non_wrapping(child, parent, start, end, size):
    current_index = 0
    for i in range(size):
        if current_index == start:
            current_index = end + 1
        if parent[i] not in child:
            child[current_index] = parent[i]
            current_index += 1


def order_crossover_2(parent1, parent2, cost_matrix=None, u=0.5):
    size = len(parent1)
    mask = [random.random() < u for _ in range(size)]

    child1 = [-1] * size
    child2 = [-1] * size

    for i in range(size):
        if mask[i]:
            child1[i] = parent1[i]
            child2[i] = parent2[i]

    fill_child_positions(child1, parent2, size)
    fill_child_positions(child2, parent1, size)

    return child1, child2


def precedence_preservative_crossover(parent1, parent2, cost_matrix=None):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    k = end - start + 1

    child1 = parent1[:start] + parent2[start : start + k] + parent1[end + 1 :]
    child2 = parent2[:start] + parent1[start : start + k] + parent2[end + 1 :]

    return child1, child2


def uniform_precedence_preservative_crossover(
    parent1, parent2, cost_matrix=None, u=0.5
):
    size = len(parent1)
    mask = [random.random() < u for _ in range(size)]

    child1 = [-1] * size
    child2 = [-1] * size

    for i in range(size):
        if mask[i]:
            child1[i] = parent1[i]
            child2[i] = parent2[i]

    fill_child_positions(child1, parent2, size)
    fill_child_positions(child2, parent1, size)

    return child1, child2


def uniform_partially_matched_crossover(parent1, parent2, cost_matrix=None, u=0.5):
    size = len(parent1)
    mask = [random.random() < u for _ in range(size)]

    child1 = parent1[:]
    child2 = parent2[:]

    for i in range(size):
        if mask[i]:
            temp1 = child1[i]
            temp2 = child2[i]

            index1 = child1.index(temp2)
            index2 = child2.index(temp1)

            child1[i], child1[index1] = child1[index1], child1[i]
            child2[i], child2[index2] = child2[index2], child2[i]

    return child1, child2


def alternating_edges_crossover(parent1, parent2, cost_matrix=None):
    size = len(parent1)
    child = [-1] * size
    child[0] = parent1[0]
    current_index = 0
    used = set()
    used.add(child[0])

    for i in range(1, size):
        if i % 2 == 1:
            next_index = parent2.index(child[current_index])
            next_city = parent1[(next_index + 1) % size]
        else:
            next_index = parent1.index(child[current_index])
            next_city = parent2[(next_index + 1) % size]

        if next_city not in used:
            child[i] = next_city
            used.add(next_city)
            current_index = i
        else:
            for city in parent1:
                if city not in used:
                    child[i] = city
                    used.add(city)
                    current_index = i
                    break

    return child


def heuristic_greedy_crossover(parent1, parent2, cost_matrix):
    size = len(parent1)
    child = [-1] * size
    child[0] = parent1[0]
    used = set()
    used.add(child[0])

    for i in range(1, size):
        candidates = []
        for parent in (parent1, parent2):
            index = parent.index(child[i - 1])
            next_city = parent[(index + 1) % size]
            if next_city not in used:
                candidates.append(next_city)

        if candidates:
            child[i] = min(candidates, key=lambda city: cost_matrix[child[i - 1]][city])
        else:
            for city in parent1:
                if city not in used:
                    child[i] = city
                    break

        used.add(child[i])

    return child


def heuristic_random_crossover(parent1, parent2, cost_matrix=None):
    size = len(parent1)
    child = [-1] * size
    child[0] = parent1[0]
    used = set()
    used.add(child[0])

    for i in range(1, size):
        candidates = []
        for parent in (parent1, parent2):
            index = parent.index(child[i - 1])
            next_city = parent[(index + 1) % size]
            if next_city not in used:
                candidates.append(next_city)

        if candidates:
            child[i] = random.choice(candidates)
        else:
            for city in parent1:
                if city not in used:
                    child[i] = city
                    break

        used.add(child[i])

    return child


def heuristic_probabilistic_crossover(parent1, parent2, cost_matrix, alpha=0.7):
    size = len(parent1)
    child = [-1] * size
    child[0] = parent1[0]
    used = set()
    used.add(child[0])

    for i in range(1, size):
        candidates = []
        for parent in (parent1, parent2):
            index = parent.index(child[i - 1])
            next_city = parent[(index + 1) % size]
            if next_city not in used:
                candidates.append(next_city)

        if candidates:
            probabilities = [
                alpha if cost_matrix[child[i - 1]][city] < 10 else 1 - alpha
                for city in candidates
            ]
            total_prob = sum(probabilities)
            probabilities = [prob / total_prob for prob in probabilities]
            child[i] = random.choices(candidates, weights=probabilities, k=1)[0]
        else:
            for city in parent1:
                if city not in used:
                    child[i] = city
                    break

        used.add(child[i])

    return child
