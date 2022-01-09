#   Script written by group 6 (Mohammad Reza Moslemifar (Student ID: 963613087),
#                              Ali Abbas Mollaei (Student ID: 963613063),
#                              Mohammad Ali Ghasemi (Student ID: 963613071))
#       for the artificial intelligence and expert systems course of University of Isfahan, class 03 of Autumn 2021
#
#   Local search and pathfinding agent using simulated annealing and A* search algorithms
#
#   The raw code for the classes and utility and pathfinding functions are cloned from the
#       reference book's algorithm implementations repository
#
#   The code was heavily changed and improved upon to suit the given problem for the course


import copy
from queue import Queue

from base_agent import BaseAgent
from game_data import GameData

import functools
import heapq
import numpy as np
import random
import sys
from operator import neg


# ---------------------------------------------Utility functions and classes--------------------------------------------

def sld(here: str, there: str):
    here_x = ord(here[0])
    here_y = ord(here[1])
    there_x = ord(there[0])
    there_y = ord(there[1])
    dist = np.sqrt(pow(there_x - here_x, 2) + pow(there_y - here_y, 2))
    return dist


def is_in(elt, seq):
    """Similar to (elt in seq), but compares with 'is', not '=='."""
    return any(x is elt for x in seq)


def probability(p):
    """Return true with probability p."""
    return p > random.uniform(0.0, 1.0)


def distance(a, b):
    """The distance between two (x, y) points."""
    xA, yA = a
    xB, yB = b
    return np.hypot((xA - xB), (yA - yB))


def memoize(fn, slot=None, maxsize=32):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, use lru_cache for caching the values."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args):
            return fn(*args)

    return memoized_fn


class PriorityQueue:
    """A Queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first.
    If order is 'min', the item with minimum f(x) is
    returned first; if order is 'max', then it is the item with maximum f(x).
    Also supports dict-like lookup."""

    def __init__(self, order='min', f=lambda x: x):
        self.heap = []
        if order == 'min':
            self.f = f
        elif order == 'max':  # now item with max f(x)
            self.f = lambda x: -f(x)  # will be popped first
        else:
            raise ValueError("Order must be either 'min' or 'max'.")

    def append(self, item):
        """Insert item at its correct position."""
        heapq.heappush(self.heap, (self.f(item), item))

    def extend(self, items):
        """Insert each item in items at its correct position."""
        for item in items:
            self.append(item)

    def pop(self):
        """Pop and return the item (with min or max f(x) value)
        depending on the order."""
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception('Trying to pop from empty PriorityQueue.')

    def __len__(self):
        """Return current capacity of PriorityQueue."""
        return len(self.heap)

    def __contains__(self, key):
        """Return True if the key is in PriorityQueue."""
        return any([item == key for _, item in self.heap])

    def __getitem__(self, key):
        """Returns the first value associated with key in PriorityQueue.
        Raises KeyError if key is not present."""
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    def __delitem__(self, key):
        """Delete the first occurrence of key."""
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)


# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------Local search using simulated annealing----------------------------------------
# ----------------------------Problem superclass----------------------------

class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


# ----------------------------Melody problem----------------------------


class MelodyProblem(Problem):
    def __init__(self, init, notes_in_map, problem_game_data: GameData):
        super().__init__(init)
        self.notes_in_map = notes_in_map
        self.game_data = problem_game_data

    def actions(self, state):
        return self.notes_in_map

    def result(self, state, action):
        return state.__add__(action)

    def value(self, state):
        return neg(self.game_data.h(state))


# ----------------------------Simulated annealing----------------------------


def exp_schedule(k=20, lam=0.005, limit=100):
    """One possible schedule function for simulated annealing"""
    return lambda t: (k * np.exp(-lam * t) if t < limit else 0)


def elemental_simulated_annealing(problem, schedule=exp_schedule()):
    current_set = problem.initial
    actions = problem.actions(current_set)
    for i in range(problem.game_data.melody_length):
        if len(current_set) < (i + 1):
            current_element = random.choice(actions)
            current_set.append(current_element)
        for t in range(sys.maxsize):
            T = schedule(t)
            if T == 0:
                break
            replace_element = random.choice(actions)
            replace_set = current_set.copy()
            replace_set[i] = replace_element
            delta_e = problem.value(replace_set) - problem.value(current_set)
            if delta_e > 0 or probability(np.exp(delta_e / T)):
                current_set[i] = replace_element
    return current_set


def full_simulated_annealing(problem, schedule=exp_schedule()):
    current = problem.initial
    actions = problem.actions([])
    for t in range(sys.maxsize):
        T = schedule(t)
        if T == 0:
            return current
        nxt = list()
        for i in range(problem.game_data.melody_length):
            nxt.append(random.choice(actions))
        delta_e = problem.value(nxt) - problem.value(current)
        if delta_e > 0 or probability(np.exp(delta_e / T)):
            current = nxt


# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------------Pathfinding using A* algorithm--------------------------------------------


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


class Graph:
    """A graph connects nodes (vertices) by edges (links). Each edge can also
    have a length associated with it. The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2})
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C. You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added. You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B. 'Lengths' can actually be any object at
    all, and nodes can be any hashable object."""

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)

    def temp_delete(self, key):
        try:
            self.graph_dict.__delitem__(key)
        except KeyError:
            pass
        for i in self.graph_dict:
            try:
                self.graph_dict[i].__delitem__(key)
            except KeyError:
                pass
        return


def undirected_graph(graph_dict=None):
    """Build a Graph where every edge (including future ones) goes both ways."""
    return Graph(graph_dict=graph_dict, directed=False)


class GraphProblem(Problem):
    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or np.inf)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is str:
                return int(distance(locs[node], locs[self.goal]))

            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return np.inf


# find locations of each note in map and
# select best location for each note (the least straight-line distance from here to there)
def find_note_locations(matrix):
    locs = dict()

    chars = []
    for i in matrix:
        for j in i:
            if j != '.' and not chars.__contains__(j):
                chars.append(j)

    for char in chars:
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == char:
                    x = chr(ord('A') + i)
                    y = chr(ord('A') + j)
                    if char in locs:
                        locs[char].append(x + y)
                    else:
                        locs[char] = [x + y]

    return chars, locs


def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None


def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)


# get a list of goal notes and find the path with least cost through them
def find_path_for_goals_list(astar_game_data: GameData, problem_map, init, goals: list):
    notes_in_map, note_locations = find_note_locations(astar_game_data.matrix)

    total_path = list()
    total_moves = list()

    for goal_note in goals:

        min_dist = 1000
        for note_location in note_locations[goal_note]:
            if sld(init, note_location) < min_dist:
                min_dist = sld(init, note_location)
                min_location = note_location

        location = min_location

        new_map = copy.deepcopy(problem_map)
        other_goals = copy.deepcopy(note_locations)
        for n in other_goals:
            if other_goals[n].__contains__(location):
                other_goals[n].remove(location)
            if other_goals[n].__contains__(init):
                other_goals[n].remove(init)
        for n in other_goals:
            for skip in other_goals[n]:
                new_map.temp_delete(skip)

        problem = GraphProblem(init, location, new_map)

        astar_for_goal = astar_search(problem)
        path = astar_for_goal.solution()
        moves = list()

        if len(path) == 0:
            moves.append('nomove')
        else:
            for i in range(len(path)):
                if i == 0:
                    cur_x = init[0]
                    cur_y = init[1]
                    nxt_x = path[i][0]
                    nxt_y = path[i][1]

                    if nxt_x > cur_x:
                        moves.append('E')
                    elif nxt_x < cur_x:
                        moves.append('W')
                    elif nxt_y > cur_y:
                        moves.append('N')
                    else:
                        moves.append('S')

                    cur_x = nxt_x
                    cur_y = nxt_y

                else:
                    nxt_x = path[i][0]
                    nxt_y = path[i][1]

                    if nxt_x > cur_x:
                        moves.append('E')
                    elif nxt_x < cur_x:
                        moves.append('W')
                    elif nxt_y > cur_y:
                        moves.append('N')
                    else:
                        moves.append('S')

                    cur_x = nxt_x
                    cur_y = cur_y

        init = location

        total_path.extend(path)
        total_moves.extend(moves)
    return total_path, total_moves


# ----------------------------------------------------------------------------------------------------------------------


class Agent(BaseAgent):
    done = False
    moves = Queue()

    def do_move(self, game_data: GameData):
        if self.done is False:
            # initialize the state space graph
            game_map_7x7 = undirected_graph(dict(
                AA=dict(BA=1, AB=1),
                AB=dict(BB=1, AC=1),
                AC=dict(BC=1, AD=1),
                AD=dict(BD=1, AE=1),
                AE=dict(BE=1, AF=1),
                AF=dict(BF=1, AG=1),
                AG=dict(BG=1),

                BA=dict(CA=1, BB=1),
                BB=dict(CB=1, BC=1),
                BC=dict(CC=1, BD=1),
                BD=dict(CD=1, BE=1),
                BE=dict(CE=1, BF=1),
                BF=dict(CF=1, BG=1),
                BG=dict(CG=1),

                CA=dict(DA=1, CB=1),
                CB=dict(DB=1, CC=1),
                CC=dict(DC=1, CD=1),
                CD=dict(DD=1, CE=1),
                CE=dict(DE=1, CF=1),
                CF=dict(DF=1, CG=1),
                CG=dict(DG=1),

                DA=dict(EA=1, DB=1),
                DB=dict(EB=1, DC=1),
                DC=dict(EC=1, DD=1),
                DD=dict(ED=1, DE=1),
                DE=dict(EE=1, DF=1),
                DF=dict(EF=1, DG=1),
                DG=dict(EG=1),

                EA=dict(FA=1, EB=1),
                EB=dict(FB=1, EC=1),
                EC=dict(FC=1, ED=1),
                ED=dict(FD=1, EE=1),
                EE=dict(FE=1, EF=1),
                EF=dict(FF=1, EG=1),
                EG=dict(FG=1),

                FA=dict(GA=1, FB=1),
                FB=dict(GB=1, FC=1),
                FC=dict(GC=1, FD=1),
                FD=dict(GD=1, FE=1),
                FE=dict(GE=1, FF=1),
                FF=dict(GF=1, FG=1),
                FG=dict(GG=1),

                GA=dict(GB=1),
                GB=dict(GC=1),
                GC=dict(GD=1),
                GD=dict(GE=1),
                GE=dict(GF=1),
                GF=dict(GG=1)
            ))
            game_map_9x9 = undirected_graph(dict(
                AA=dict(BA=1, AB=1),
                AB=dict(BB=1, AC=1),
                AC=dict(BC=1, AD=1),
                AD=dict(BD=1, AE=1),
                AE=dict(BE=1, AF=1),
                AF=dict(BF=1, AG=1),
                AG=dict(BG=1, AH=1),
                AH=dict(BH=1, AI=1),
                AI=dict(BI=1),

                BA=dict(CA=1, BB=1),
                BB=dict(CB=1, BC=1),
                BC=dict(CC=1, BD=1),
                BD=dict(CD=1, BE=1),
                BE=dict(CE=1, BF=1),
                BF=dict(CF=1, BG=1),
                BG=dict(CG=1, BH=1),
                BH=dict(CH=1, BI=1),
                BI=dict(CI=1),

                CA=dict(DA=1, CB=1),
                CB=dict(DB=1, CC=1),
                CC=dict(DC=1, CD=1),
                CD=dict(DD=1, CE=1),
                CE=dict(DE=1, CF=1),
                CF=dict(DF=1, CG=1),
                CG=dict(DG=1, CH=1),
                CH=dict(DH=1, CI=1),
                CI=dict(DI=1),

                DA=dict(EA=1, DB=1),
                DB=dict(EB=1, DC=1),
                DC=dict(EC=1, DD=1),
                DD=dict(ED=1, DE=1),
                DE=dict(EE=1, DF=1),
                DF=dict(EF=1, DG=1),
                DG=dict(EG=1, BH=1),
                DH=dict(EH=1, BI=1),
                DI=dict(EI=1),

                EA=dict(FA=1, EB=1),
                EB=dict(FB=1, EC=1),
                EC=dict(FC=1, ED=1),
                ED=dict(FD=1, EE=1),
                EE=dict(FE=1, EF=1),
                EF=dict(FF=1, EG=1),
                EG=dict(FG=1, BH=1),
                EH=dict(FH=1, BI=1),
                EI=dict(FI=1),

                FA=dict(GA=1, FB=1),
                FB=dict(GB=1, FC=1),
                FC=dict(GC=1, FD=1),
                FD=dict(GD=1, FE=1),
                FE=dict(GE=1, FF=1),
                FF=dict(GF=1, FG=1),
                FG=dict(GG=1, BH=1),
                FH=dict(GH=1, BI=1),
                FI=dict(GI=1),

                GA=dict(HA=1, GB=1),
                GB=dict(HB=1, GC=1),
                GC=dict(HC=1, GD=1),
                GD=dict(HD=1, GE=1),
                GE=dict(HE=1, GF=1),
                GF=dict(HF=1, GG=1),
                GG=dict(HG=1, BH=1),
                GH=dict(HH=1, BI=1),
                GI=dict(HI=1),

                HA=dict(IA=1, HB=1),
                HB=dict(IB=1, HC=1),
                HC=dict(IC=1, HD=1),
                HD=dict(ID=1, HE=1),
                HE=dict(IE=1, HF=1),
                HF=dict(IF=1, HG=1),
                HG=dict(IG=1, BH=1),
                HH=dict(IH=1, BI=1),
                HI=dict(II=1),

                IA=dict(IB=1),
                IB=dict(IC=1),
                IC=dict(ID=1),
                ID=dict(IE=1),
                IE=dict(IF=1),
                IF=dict(IG=1),
                IG=dict(IH=1),
                IH=dict(II=1)
            ))

            # set node locations (used for computing h)
            game_map_7x7.locations = dict(
                AA=(1, 1), AB=(1, 2), AC=(1, 3), AD=(1, 4), AE=(1, 5), AF=(1, 6), AG=(1, 7),
                BA=(2, 1), BB=(2, 2), BC=(2, 3), BD=(2, 4), BE=(2, 5), BF=(2, 6), BG=(2, 7),
                CA=(3, 1), CB=(3, 2), CC=(3, 3), CD=(3, 4), CE=(3, 5), CF=(3, 6), CG=(3, 7),
                DA=(4, 1), DB=(4, 2), DC=(4, 3), DD=(4, 4), DE=(4, 5), DF=(4, 6), DG=(4, 7),
                EA=(5, 1), EB=(5, 2), EC=(5, 3), ED=(5, 4), EE=(5, 5), EF=(5, 6), EG=(5, 7),
                FA=(6, 1), FB=(6, 2), FC=(6, 3), FD=(6, 4), FE=(6, 5), FF=(6, 6), FG=(6, 7),
                GA=(7, 1), GB=(7, 2), GC=(7, 3), GD=(7, 4), GE=(7, 5), GF=(7, 6), GG=(7, 7)
            )
            game_map_9x9.locations = dict(
                AA=(1, 1), AB=(1, 2), AC=(1, 3), AD=(1, 4), AE=(1, 5), AF=(1, 6), AG=(1, 7), AH=(1, 8), AI=(1, 9),
                BA=(2, 1), BB=(2, 2), BC=(2, 3), BD=(2, 4), BE=(2, 5), BF=(2, 6), BG=(2, 7), BH=(2, 8), BI=(2, 9),
                CA=(3, 1), CB=(3, 2), CC=(3, 3), CD=(3, 4), CE=(3, 5), CF=(3, 6), CG=(3, 7), CH=(3, 8), CI=(3, 9),
                DA=(4, 1), DB=(4, 2), DC=(4, 3), DD=(4, 4), DE=(4, 5), DF=(4, 6), DG=(4, 7), DH=(4, 8), DI=(4, 9),
                EA=(5, 1), EB=(5, 2), EC=(5, 3), ED=(5, 4), EE=(5, 5), EF=(5, 6), EG=(5, 7), EH=(5, 8), EI=(5, 9),
                FA=(6, 1), FB=(6, 2), FC=(6, 3), FD=(6, 4), FE=(6, 5), FF=(6, 6), FG=(6, 7), FH=(6, 8), FI=(6, 9),
                GA=(7, 1), GB=(7, 2), GC=(7, 3), GD=(7, 4), GE=(7, 5), GF=(7, 6), GG=(7, 7), GH=(7, 8), GI=(7, 9),
                HA=(8, 1), HB=(8, 2), HC=(8, 3), HD=(8, 4), HE=(8, 5), HF=(8, 6), HG=(8, 7), HH=(8, 8), HI=(8, 9),
                IA=(9, 1), IB=(9, 2), IC=(9, 3), ID=(9, 4), IE=(9, 5), IF=(9, 6), IG=(9, 7), IH=(9, 8), II=(9, 9)
            )

            notes_in_map, note_locations = find_note_locations(game_data.matrix)
            melody_problem = MelodyProblem(init=[], notes_in_map=notes_in_map, problem_game_data=game_data)
            notes_to_traverse = full_simulated_annealing(melody_problem, schedule=exp_schedule())
            print('Result of simulated annealing: ' + str(notes_to_traverse))

            init = chr(65 + game_data.agent_pos[0]) + chr(65 + game_data.agent_pos[1])

            if game_data.grid_height == 7 and game_data.grid_width == 7:
                game_map = game_map_7x7
            elif game_data.grid_height == 9 and game_data.grid_width == 9:
                game_map = game_map_9x9

            path, moves = find_path_for_goals_list(astar_game_data=game_data, problem_map=game_map,
                                                        init=init, goals=notes_to_traverse)
            for move in moves:
                self.moves.put(move)

            print(path)
            print(moves)
            self.done = True

        return self.moves.get()


if __name__ == "__main__":
    agent = Agent()
    agent.play()

