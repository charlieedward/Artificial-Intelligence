'''
Student: KuoChenHuang
USCID: 8747-1422-96
'''

from collections import deque
import heapq
from heapq import heappush, heappop

# def BFS(dimension, W, H, starting_point, stamina, num_of_lodges, lodges, elevations):
#     solution = dict()
#     movements = ((0, 1), (-1, 0), (0, -1), (1, 0), (-1, 1), (-1, -1), (1, -1), (1, 1))
#     node_travelling = deque()
#     node_travelling.append(starting_point)
#     visited_or_not = [[[False]] * dimension[0] for _ in range(dimension[1])]
#     lodge_index_dict = {key: i for i, key in enumerate(lodges)}
#     # print(lodge_index_dict)
#
#     while len(node_travelling) > 0:
#         current_node = node_travelling.popleft()
#         current_node_elevation = elevations[current_node[1]][current_node[0]]
#         # print('current_node: ', current_node)
#         # print('current_node_elevation: ',current_node_elevation)
#         lodges_copy = list(lodges)
#
#         # if we arrive any lodge
#         if current_node in lodges:
#             print('we find it!')
#
#             my_path = deque()
#             my_path.append(current_node)
#             node_check = current_node
#             while node_check is not starting_point:
#                 # print(visited_or_not)
#                 last_node = visited_or_not[node_check[1]][node_check[0]][1]
#                 my_path.appendleft(last_node)
#                 node_check = last_node
#             lodge_index = lodge_index_dict[current_node]
#             solution[lodge_index] = my_path
#             if lodges_copy is None:
#                 break
#
#             lodges_copy.remove(current_node)
#
#         for move in movements:
#             next_node = tuple(map(lambda i, j: i + j, current_node, move))
#
#             # Check if the next position is within range
#             if 0 <= next_node[0] < dimension[0] and 0 <= next_node[1] < dimension[1]:
#                 # print('next_node: ', next_node)
#                 next_node_elevation = elevations[next_node[1]][next_node[0]]
#
#                 # if next cell is a tree
#                 if next_node_elevation < 0:
#                     if visited_or_not[next_node[1]][next_node[0]][0] is False:
#                         # only if our current E is higher than or equal to tree E, we're allowed to move into the cell
#                         if current_node_elevation >= abs(next_node_elevation):
#                             # print('current_node: ', current_node)
#                             # print('we are adding: ',next_node)
#                             visited_or_not[next_node[1]][next_node[0]] = [True, current_node]
#                             node_travelling.append(next_node)
#                 else:
#                     if visited_or_not[next_node[1]][next_node[0]][0] is False:
#                         # if current + stamina >= next
#                         if abs(current_node_elevation) >= (next_node_elevation) - stamina:
#                             # print('current_node: ', current_node)
#                             # print('we are adding: ', next_node)
#                             visited_or_not[next_node[1]][next_node[0]] = [True, current_node]
#                             node_travelling.append(next_node)
#
#         #print('node_travelling: ',node_travelling)
#         # print(visited_or_not)
#         # print('end of this round: ', current_node)
#
#     # sort by lodge index
#     solution = {k: v for k, v in sorted(list(solution.items()))}
#
#     return solution

def get_neighbors(coord, W, H):
    # return all passable 8-connected neighbors
    x, y = coord
    neighbors = [(x+1, y), (x+1, y+1), (x, y+1), (x-1, y+1), (x-1, y), (x-1, y-1), (x, y-1), (x+1, y-1)]
    return [n for n in neighbors if is_passable(n, W, H)]

def is_passable(coord, W, H):
    x, y = coord
    # check if the coord is passable according to the rules in the problem statement
    if x < 0 or y < 0 or x >= W or y >= H:
        return False
    return True

def BFS(starting_point, W, H, lodges, elevations, stamina):
    # use a dictionary to store the cost to get to each cell
    cost = {starting_point: 0}
    # use a dictionary to store the parent of each cell
    parent = {starting_point: None}
    # use a priority queue to store the cells to explore
    heap = [(0, starting_point)]
    solution = dict()
    lodge_index_dict = {key: i for i, key in enumerate(lodges)}
    for goal in lodges:
        solution[goal] = list()
    while heap:
        # pop the cell with the lowest cost from the priority queue
        current_cost, current = heappop(heap)
        # print('current : ',current)
        if current in lodges:
            # print('current: ', current)
            # print('cost: ', current_cost)
            # if we have reached one of the goals, trace back the path
            path = [current]
            lodge_index = lodge_index_dict[current]
            while parent[path[-1]] != None:
                path.append(parent[path[-1]])
            solution[lodge_index] = list(reversed(path))

        # print('neighbors: ', get_neighbors(current, E))
        for neighbor in get_neighbors(current, W, H):
            # print('neighbor: ', neighbor)
            '''
            if neighbor in cost:
                continue
            '''
            # compute the cost to get to the neighbor
            new_cost = current_cost + 1

            neighbor_E = elevations[neighbor[1]][neighbor[0]]
            current_E = elevations[current[1]][current[0]]

            # case1 : if we're not at a tree
            if current_E >= 0:
                # the cell(not a tree) with the elevation high enough is not allowed
                if neighbor_E >= 0 and neighbor_E > current_E + stamina:
                    continue
                # we can not enter into a higher tree
                if neighbor_E < 0 and abs(neighbor_E) > current_E:
                    continue

            # case2 : if we're at a tree
            elif current_E < 0:
                # we could not move to a higher tree
                if neighbor_E < 0 and abs(neighbor_E) > abs(current_E):
                    continue
                # we could not move to a cell with too high elevation
                if neighbor_E >= 0 and neighbor_E - abs(current_E) > stamina:
                    continue

            if neighbor in cost:
                if new_cost >= cost[neighbor]:
                    continue

            # update the cost and parent of the neighbor
            cost[neighbor] = new_cost
            parent[neighbor] = current
            heappush(heap, (cost[neighbor], neighbor))

    return solution

def UCS(starting_point, W, H, lodges, elevations, stamina):
    # use a dictionary to store the cost to get to each cell
    cost = {starting_point: 0}
    # use a dictionary to store the parent of each cell
    parent = {starting_point: None}
    # use a priority queue to store the cells to explore
    heap = [(0, starting_point)]
    solution = dict()
    lodge_index_dict = {key: i for i, key in enumerate(lodges)}
    for goal in lodges:
        solution[goal] = list()
    while heap:
        # pop the cell with the lowest cost from the priority queue
        current_cost, current = heappop(heap)
        # print('current : ',current)
        if current in lodges:
            print('current: ', current)
            print('cost: ', current_cost)
            # if we have reached one of the goals, trace back the path
            path = [current]
            lodge_index = lodge_index_dict[current]
            while parent[path[-1]] != None:
                path.append(parent[path[-1]])
            solution[lodge_index] = list(reversed(path))

        # print('neighbors: ', get_neighbors(current, E))
        for neighbor in get_neighbors(current, W, H):
            # print('neighbor: ', neighbor)

            # compute the cost to get to the neighbor
            new_cost = current_cost + 10 if abs(neighbor[0] - current[0]) + abs(
                neighbor[1] - current[1]) == 1 else current_cost + 14

            neighbor_E = elevations[neighbor[1]][neighbor[0]]
            current_E = elevations[current[1]][current[0]]


            # case1 : if we're not at a tree
            if current_E >= 0:
                # the cell(not a tree) with the elevation high enough is not allowed
                if neighbor_E >= 0 and neighbor_E > current_E + stamina:
                    continue
                # we can not enter into a higher tree
                if neighbor_E < 0 and abs(neighbor_E) > current_E:
                    continue

            # case2 : if we're at a tree
            else:
                # we could not move to a higher tree
                if neighbor_E < 0 and abs(neighbor_E) > abs(current_E):
                    continue
                # we could not move to a cell with too high elevation
                if neighbor_E >= 0 and neighbor_E - abs(current_E) > stamina:
                    continue

            if neighbor in cost:
                if new_cost >= cost[neighbor]:
                    continue

            # update the cost and parent of the neighbor
            cost[neighbor] = new_cost
            parent[neighbor] = current
            heappush(heap, (cost[neighbor], neighbor))

            # print('neighbor: ', neighbor)
            # print('heap: ', heap)

    return solution

def generate_graph(W,H):
    grid = []
    for x in range(W):
        for y in range(H):
            grid.append((x, y))
    return grid

def reconstruct_path(came_from, current, starting_point, M_T, elevations, stamina):
    total_path = [current]
    x = [elevations[current[1]][current[0]]]
    check_item = ((starting_point, 0), starting_point)
    # print(came_from)
    # print('========')
    # print('check_item: ', (check_item))
    # while starting_point not in total_path or check_item in came_from.items():
    while starting_point not in total_path or check_item in came_from.items():
        # print(x)
        all_possible_prev = sorted([[(curr, M), prev] for (curr, M), prev in came_from.items() if curr == current], key = lambda k: (k[0][1]))
        # print('all_possible_prev: ', all_possible_prev)
        if len(all_possible_prev) > 1:
            # print('more than 2 choices')
            next_node_e = x[-2]
            for pair in all_possible_prev:
                m = pair[0][1]
                prev = pair[1]
                # prev_e = elevations[prev[1]][prev[0]]
                curr_e = abs(elevations[current[1]][current[0]])
                # print('next_node: ', next_node_e)
                if m + stamina >= next_node_e - curr_e:
                    # print('i choose: ', prev)
                    current = prev
                    target = pair
                    # print('pair: ', pair)
                    came_from.pop(pair[0])
                    break

        else:
            target = all_possible_prev[0]
            current = all_possible_prev[0][1]
            # print('all_possible_prev[0]: ', all_possible_prev[0])
            came_from.pop(all_possible_prev[0][0])

        if tuple(target) == check_item:
            # print('target break: ', tuple(target))
            break
        else:
            # print('target: ', tuple(target))
            # print('add: ', current)
            # print(came_from)
            #
            # print('============')
            total_path.append(current)
            x.append(elevations[current[1]][current[0]])


    return list(reversed(total_path))


def heuristic_cost_estimate(a, b):
    """
    Returns the Manhattan distance between two nodes
    """
    x1, y1 = a
    x2, y2 = b
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    # return abs(x1 - x2) + abs(y1 - y2)
    return 14 * min(dx,dy) + 10 * abs(dx-dy)

def dist_between(a, b):
    """
    Returns the Euclidean distance between two nodes
    """
    x1, y1 = a
    x2, y2 = b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def AStar(starting_point, W, H, lodges, elevations, stamina):

    solution = dict()
    lodge_index_dict = {key: i for i, key in enumerate(lodges)}
    # for goal in lodges:
    #     solution[goal] = list()

    for lodge in lodges:
        closed_set = dict()
        # The set of currently discovered nodes that are not evaluated yet.
        # Initially, only the start node is known.
        open_set = {starting_point}
        # For each node, which node it can most efficiently be reached from.
        # If a node can be reached from many nodes, cameFrom will eventually contain the most efficient previous step.
        came_from = dict()
        came_from[(starting_point, 0)] = starting_point
        # the cost of getting from the start node to that node.
        # g_score = {node: [(float('inf'), float('inf'))] for node in generate_graph(W,H)}
        # g_score[starting_point] = [(0, 0)]
        g_score = {node: {} for node in generate_graph(W, H)}
        g_score[starting_point] = {starting_point: 0}
        # For each node, the total cost of getting from the start node to the goal by passing by that node.
        f_score = {node: [(float('inf'), float('inf'))] for node in generate_graph(W,H)}
        # print(f_score)
        # f_score -> [(cost, M)]
        f_score[starting_point] = [(heuristic_cost_estimate(starting_point, lodge), 0)]

        while open_set:
            # print('openset_1: ', open_set)
            # print('gscore: ', [(x, g_score[x]) for x in g_score if len(g_score[x])>0 and g_score[x][0][0] != float('inf') ])
            # print('f_score: ',  [(x, f_score[x]) for x in f_score if len(f_score[x])>0 and f_score[x][0][0] != float('inf') ])
            current = min(open_set, key=lambda x: f_score[x][0][0])
            print('current: ', current)
            print('choose: ', f_score[current][0][0])

            M_T = f_score[current][0][1]

            if current == lodge:
                print('find it')
                print(f_score[current][0][0])
                # print('came_From: ', came_from)
                lodge_index = lodge_index_dict[current]
                solution[lodge_index] = reconstruct_path(came_from, current, starting_point, M_T, elevations, stamina)
                break

            if len(f_score[current]) == 1:
                open_set.remove(current)

            f_score[current].pop(0)

            # if f_score[current][0] != (float('inf'), float('inf')):
            #     f_score[current].pop(0)

            closed_set[current] = M_T

            # get_neighbors only consider the dimension, haven't considered elevation and anything else
            for neighbor in get_neighbors(current, W, H):

                prev_cell = came_from[(current, M_T)]

                prev_E = elevations[prev_cell[1]][prev_cell[0]]
                neighbor_E = elevations[neighbor[1]][neighbor[0]]
                current_E = elevations[current[1]][current[0]]

                M = max(0, abs(prev_E) - abs(current_E)) if (abs(neighbor_E) - abs(current_E)) >= 0 else 0
                # M == M_T
                M_forNext = max(0, abs(current_E) - abs(neighbor_E))
                # case1 : if we're not at a tree
                if current_E >= 0:
                    # the cell(not a tree) with the elevation high enough is not allowed
                    if neighbor_E >= 0 and neighbor_E > current_E + stamina + M:
                        continue
                    # we can not enter into a higher tree
                    if neighbor_E < 0 and abs(neighbor_E) > current_E:
                        continue

                # case2 : if we're at a tree
                elif current_E < 0:
                    # we could not move to a higher tree
                    if neighbor_E < 0 and abs(neighbor_E) > abs(current_E):
                        continue
                    # we could not move to a cell with too high elevation
                    if neighbor_E >= 0 and neighbor_E > abs(current_E) + stamina + M:
                        continue

                if neighbor in closed_set and closed_set[neighbor] >= M_forNext:
                    continue  # Ignore the neighbor which is already evaluated and we're sure the new M is not higher.


                # get the right g_score
                # all_gscore = g_score[current]
                # g = 0
                # for pair in all_gscore:
                #     if pair[1] == M:
                #         g = pair[0]
                #         break
                g = g_score[current][prev_cell]

                # horizontal_g_score = g_score[current] + 10 if abs(neighbor[0] - current[0]) + abs(
                #     neighbor[1] - current[1]) == 1 else g_score[current] + 14

                horizontal_g_score = g + 10 if abs(neighbor[0] - current[0]) + abs(
                    neighbor[1] - current[1]) == 1 else g + 14
                # print('g: ', g)
                # print('horizontal_g_score: ', horizontal_g_score)

                elevation_g_score = 0 if (abs(neighbor_E) - abs(current_E) <= M) else max(0, abs(neighbor_E) - abs(current_E) - M)

                total_g_score = horizontal_g_score + elevation_g_score


                h_score = heuristic_cost_estimate(neighbor, lodge)

                total_score = total_g_score + h_score

                if neighbor not in open_set:
                    if neighbor not in closed_set:
                        # we discover a new node
                        open_set.add(neighbor)
                    # already in closed set
                    # if we get a better case with higher M, add it
                    elif M_forNext > closed_set[neighbor]:
                        open_set.add(neighbor)


                elif total_score >= f_score[neighbor][0][0] and M_forNext <= f_score[neighbor][0][1]:
                    continue  # This is not a better path and cannot help us to discover new nodes.

                # Record it!
                # print('neighbor: ',neighbor)
                came_from[(neighbor, M_forNext)] = current
                # print('g: ', g)
                # print('total_g_score: ', total_g_score)
                # g_score[neighbor] = total_g_score
                g_score[neighbor][current] = total_g_score
                # if g_score[neighbor][0] == (float('inf'), float('inf')):
                #     g_score[neighbor][0] = (total_g_score, M_forNext)
                # else:
                #     for pair in g_score[neighbor]:
                #         if pair[1] == M_forNext and total_g_score < pair[0]:
                #             g_score[neighbor].remove(pair)
                #             # g_score[neighbor].append((total_g_score, M))
                #     g_score[neighbor].append((total_g_score, M_forNext))

                # print('f_score: ', f_score[neighbor])
                if len(f_score[neighbor]) == 0:
                    f_score[neighbor].append((total_score, M_forNext))
                else:
                    if f_score[neighbor][0] == (float('inf'), float('inf')):
                        f_score[neighbor][0] = (total_score, M_forNext)
                    else:
                        f_score[neighbor].append((total_score, M_forNext))
                        f_score[neighbor] = sorted(f_score[neighbor])
            #
            # print('g_score: ', g_score)
            # print('f_score: ', f_score)
    return solution

# READ FILE
file_path = 'testcase/input47.txt'
with open(file_path, 'r') as file:
    # First Line
    algorithm = file.readline().rstrip()
    # Second Line
    #dimension = tuple(file.readline().split())
    dimension = tuple(map(int, file.readline().split()))
    W = dimension[0]
    H = dimension[1]
    # Third Line
    starting_point = tuple(map(int, file.readline().split()))
    # Fourth Line
    stamina = int(file.readline())
    # Fifth Line
    num_of_lodges = int(file.readline())
    # Next N Lines
    lodges = []
    for i in range(num_of_lodges):
        lodges.append(tuple(map(int, file.readline().split())))
    # Next H Lines
    elevations = []
    for line in file:
        elevations.append(tuple(map(int, line.split())))

# print('Algorithm: ',algorithm)
# print('dimension: ',dimension)
# print('W: ', W)
# print('H: ', H)
# print('starting_point: ',starting_point)
# print('stamina: ',stamina)
# print('lodges: ',lodges)
# print('elevations: ',elevations)
# print('elevations[155][110]: ', elevations[110][155])
# print('elevations[156][110]: ', elevations[110][156])
# print('elevations[156][111]: ', elevations[111][156])


if algorithm == 'BFS':
    # output = BFS(dimension, W, H, starting_point, stamina, num_of_lodges, lodges, elevations)
    output = BFS(starting_point, W, H, lodges, elevations, stamina)
if algorithm == 'UCS':
    output = UCS(starting_point, W, H, lodges, elevations, stamina)
if algorithm == 'A*':
    output = AStar(starting_point, W, H, lodges, elevations, stamina)


with open("output_test_47.txt", 'w') as file:

    for lodge_index in range(num_of_lodges):
        try:
            route = output[lodge_index]
            file.write(" ".join(str(a) + ',' + str(b) for a, b in [entry for entry in [r for r in route]]))
            file.write("\n")
        except:
            file.write("FAIL\n")

