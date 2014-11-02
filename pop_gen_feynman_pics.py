__author__ = 'bryan'

import networkx as nx
import numpy as np
import seaborn as sns
sns.set_context('poster', font_scale=1.25)

def prob_of_path(g, node1, node2):
    '''Calculate the proability of the path the feinman integral way!
    You can see that it agrees with a more simple way, at least in terms of speed.'''
    prob = 0
    allpaths = nx.all_simple_paths(g, node1, node2)
    for path in allpaths:
        prob_path = 1
        start_nodes = path[:-1]
        finish_nodes = path[1:]
        for ni, nf in zip(start_nodes, finish_nodes):
            prob_path *= g[ni][nf]['capacity']

        prob += prob_path
    return prob

class population_state:
    def __init__(self, N, r, t):
        self.N = N
        self.r = r
        self.t = t

        self.g = N - r

    def __hash__(self):
        return hash((self.N, self.r, self.t, self.g))

    def __eq__(self, other):
        if self.__hash__() == other.__hash__():
            return True
        return False

    def __str__(self):
        return str(self.r)

def get_position(pop):
    return (pop.t, pop.r)

def get_prob_increase_neutral(pop):
    frac_red = float(pop.r) / float(pop.N)
    frac_green = float(pop.g) / float(pop.N)
    return frac_red * frac_green

def get_prob_same_neutral(pop):
    frac_red = float(pop.r) / float(pop.N)
    frac_green = float(pop.g) / float(pop.N)
    return frac_red**2 + frac_green**2

def get_prob_decrease_neutral(pop):
    frac_green = float(pop.g) / float(pop.N)
    frac_red = float(pop.r) / float(pop.N)
    return frac_green * frac_red

class population_graph():
    def __init__(self, N=6, max_t=20, Ro=3):
        self.N = N
        self.max_t = max_t
        self.Ro = Ro
        self.g = None
        self.first_node = None
        self.prob_dict = None

        self.get_prob_increase = get_prob_increase_neutral
        self.get_prob_same = get_prob_same_neutral
        self.get_prob_decrease = get_prob_decrease_neutral

    def create_graph(self, N = 6, max_t=20):
        Ro = N/2

        self.g = nx.DiGraph()
        self.first_node = population_state(N, Ro, 0)
        self.g.add_node(self.first_node)

        current_R_range = np.array([Ro, Ro])

        position_list = {}
        position_list[self.first_node] = get_position(self.first_node)

        # Position is determined by the fraction of red you have

        for current_t in range(1, max_t + 1):

            previous_R_range = current_R_range.copy()
            previous_t = current_t - 1

            current_R_range += np.array([-1, 1])

            current_R_range[current_R_range < 0] = 0
            current_R_range[current_R_range > N] = N

            # Add nodes
            list_to_add = range(current_R_range[0], current_R_range[1] + 1)
            for rcur in list_to_add:
                pop_to_add = population_state(N, rcur, current_t)
                self.g.add_node(pop_to_add)
                position_list[pop_to_add] = get_position(pop_to_add)

            # Then add weighted edges
            # Iterate over all previous states

            previous_r = range(previous_R_range[0], previous_R_range[1] + 1)

            for p in previous_r:
                prev_pop = population_state(N, p, previous_t)

                # Deal with up and down states
                if p + 1 <= N:
                    pop_increase = population_state(N, p + 1, current_t)
                    prob_increase = self.get_prob_increase(prev_pop)
                    #g.add_weighted_edges_from([(prev_pop, pop_increase, prob_increase)])
                    self.g.add_edge(prev_pop, pop_increase, capacity=prob_increase)

                if p - 1 >= 0:
                    pop_decrease = population_state(N, p - 1, current_t)
                    prob_decrease = self.get_prob_decrease(prev_pop)
                    #g.add_weighted_edges_from([(prev_pop, pop_decrease, prob_decrease)])
                    self.g.add_edge(prev_pop, pop_decrease, capacity=prob_decrease)

                stay_same_pop = population_state(N, p, current_t)
                stay_same_prob = self.get_prob_same(prev_pop)
                #g.add_weighted_edges_from([(prev_pop, stay_same_pop, stay_same_prob)])
                self.g.add_edge(prev_pop, stay_same_pop, capacity=stay_same_prob)

    def get_prob_dict(self):
        self.prob_dict = {}
        start_node = self.first_node
        self.prob_dict[start_node] = 1

        # It's easiest to just loop over nodes at the next time step, and then calculate the flow into each.

        # It will be easiest to just loop over each time interval

        for cur_t in range(1, self.max_t + 1):
            graph_nodes = np.array(self.g.nodes())
            t_list = np.array([n.t for n in graph_nodes])

            cur_nodes = graph_nodes[t_list == cur_t]
            # Get predecessor
            for n in cur_nodes:
                prev_nodes = self.g.predecessors(n)
                flow_in = 0
                for q in prev_nodes:
                    edge_prob = self.g[q][n]['capacity']
                    flow_in += edge_prob * self.prob_dict[q]

                self.prob_dict[n] = flow_in