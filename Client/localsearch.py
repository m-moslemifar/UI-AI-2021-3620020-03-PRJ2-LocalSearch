import sys
from operator import neg
from utils import *


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


# ----------------------------Heuristic function----------------------------


optimalMelody = ['a', 'b', 'c', 'd']
EMPTY_CONST = 1000


def h(state: list):
    state1 = [ord(i) - ord("a") + 1 for i in state]
    if len(state) < len(optimalMelody):
        for i in range(len(optimalMelody) - len(state)):
            state1.append(EMPTY_CONST)
    state2 = [ord(i) - ord("a") + 1 for i in optimalMelody]
    return np.linalg.norm(list(map(operator.sub, state1, state2)))


# ----------------------------Melody problem----------------------------


class MelodyProblem(Problem):
    def __init__(self, init, notesInMap, optimalMelody):
        super().__init__(init)
        self.notesInMap = notesInMap
        self.optimalMelody = optimalMelody

    def actions(self, state):
        return self.notesInMap

    def result(self, state, action):
        return state.__add__(action)

    def value(self, state):
        return neg(h(state))


# ----------------------------Simulated annealing----------------------------


def exp_schedule(k=20, lam=0.005, limit=100):
    """One possible schedule function for simulated annealing"""
    return lambda t: (k * np.exp(-lam * t) if t < limit else 0)


def elemental_simulated_annealing(problem, schedule=exp_schedule()):
    current_set = problem.initial
    actions = problem.actions(current_set)
    for i in range(len(problem.optimalMelody)):
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
        for i in range(len(problem.optimalMelody)):
            nxt.append(random.choice(actions))
        delta_e = problem.value(nxt) - problem.value(current)
        if delta_e > 0 or probability(np.exp(delta_e / T)):
            current = nxt


notesInMap = ['a', 'b', 'c', 'd', 'v', 'w', 'x', 'y', 'z']
test_problem = MelodyProblem(init=[], notesInMap=notesInMap, optimalMelody=optimalMelody)
res_one = elemental_simulated_annealing(problem=test_problem, schedule=exp_schedule())
res_all = full_simulated_annealing(problem=test_problem, schedule=exp_schedule())  # full yields better results
print(res_one)
print(res_all)
