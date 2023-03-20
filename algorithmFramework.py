from matplotlib import pyplot as plt

import itertools, time
import networkx as nx

from abc import ABC, abstractmethod

def save_graph(graph: nx.Graph, min_path: list, min_weight: int, title: str) -> None:
    pos = nx.spring_layout(graph)
    plt.figure()
    nx.draw(graph, pos, with_labels=True)
    path_edges = [(min_path[i], min_path[i+1]) for i in range(len(min_path)-1)]
    path_weights = [graph[min_path[i]][min_path[i+1]]['weight'] for i in range(len(min_path)-1)]
    edge_labels = {(min_path[i], min_path[i+1]): str(path_weights[i]) for i in range(len(min_path)-1)}
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='r', width=2, arrows=True)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10, font_color='blue')
    plt.title("Minimum weight path (weight={})".format(min_weight))
    plt.savefig(title + '.png') 

class Algorithm(ABC):
    '''Operations'''
    @abstractmethod
    def run():
        pass

class Resolver(object):
    '''Interface'''
    def __init__(self, graph: nx.Graph, algorithm: Algorithm):
        self._graph = graph
        self._algorithm = algorithm

    @abstractmethod
    def start(self, start, timeout: int = None) -> tuple[list, int, float]:
        return self.algorithm.run(self._graph, start, timeout)

    @property
    def algorithm(self):
        return self._algorithm
    
    @property
    def graph(self):
        return self._graph

    @algorithm.setter
    def algorithm(self, algorithm: Algorithm):
        self._algorithm = algorithm
    
    @graph.setter
    def graph(self, graph: nx.Graph):
        self._graph = graph

class BruteForce(Algorithm):
    def run(self, graph: nx.Graph, start: str, timeout: int = None) -> tuple[list, int, float]:
        start_time = time.perf_counter()
        nodes = list(graph.nodes())
        nodes.remove(start)
        min_weight = float('inf')
        min_path = None
        for perm in itertools.permutations(nodes):
            path_weight = 0
            current_node = start
            for next_node in perm:
                path_weight += graph[current_node][next_node]['weight']
                current_node = next_node
            path_weight += graph[current_node][start]['weight']
            if path_weight < min_weight:
                min_weight = path_weight
                min_path = [start] + list(perm) + [start]
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        save_graph(graph, min_path, min_weight, "bf_minpath")
        return tuple([min_path, min_weight, elapsed_time])

class Voracious(Algorithm):
    def run(self, graph: nx.Graph, start: str, timeout: int = None) -> tuple[list, int, float]:
        start_time = time.perf_counter()
        visited = [start]
        path = [start]
        weight = 0
        
        while len(visited) < len(graph.nodes):
            neighbors = list(graph.neighbors(path[-1]))
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            if not unvisited_neighbors:
                path.pop()
                weight -= graph[path[-1]][path[-2]]['weight']
            else:
                next_node = min(unvisited_neighbors, key=lambda n: graph[path[-1]][n]['weight'])
                visited.append(next_node)
                path.append(next_node)
                weight += graph[path[-2]][next_node]['weight']
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        min_weight = weight + graph[path[-1]][start]['weight']
        min_path = path + [start]
        save_graph(graph, min_path, min_weight, "v_minpath")
        return tuple([min_path, min_weight, elapsed_time])

class DynamicProgramming(Algorithm):
    def run(self, graph: nx.Graph, start: str, timeout: int = None) -> tuple[list, int, float]:
        start_time = time.perf_counter()
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())
        dist = {node: {frozenset(): 0} for node in nodes}
        parent = {node: {frozenset(): None} for node in nodes}

        for k in nodes[1:]:
            dist[k][frozenset([k])] = graph[start][k]['weight']
            parent[k][frozenset([k])] = start

        for subset_size in range(2, n):
            for subset in itertools.combinations(nodes[1:], subset_size):
                subset = frozenset(subset)
                for k in subset:
                    dist[k][subset] = float('inf')
                    for m in subset:
                        if k != m:
                            new_dist = dist[m][subset - {k}] + graph[m][k]['weight']
                            if new_dist < dist[k][subset]:
                                dist[k][subset] = new_dist
                                parent[k][subset] = m

        subset = frozenset(nodes[1:])
        min_dist = float('inf')
        last = None
        for k in nodes[1:]:
            dist_to_start = dist[k][subset] + graph[k][start]['weight']
            if dist_to_start < min_dist:
                min_dist = dist_to_start
                last = k

        path = []
        while last is not None:
            if subset in parent[last]:
                path.append(last)
                new_last = parent[last][subset]
                subset = subset - {last}
                last = new_last
            else:
                raise Exception("No se pudo encontrar una solución óptima")
        path.append(start)
        path.reverse()
        path = path[1:] + [path[0]]

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        save_graph(graph, path, min_dist, "dp_minpath")
        return tuple([path, min_dist, elapsed_time])