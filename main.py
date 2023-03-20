from rich import print
from rich.console import Console
from rich.table import Table
from rich.traceback import install
from algorithmFramework import Resolver, BruteForce, Voracious, DynamicProgramming

import func_timeout, os, random
import networkx as nx
import matplotlib.pyplot as plt
import inquirer

install(show_locals=True)
console = Console(record=True)

QTOWN = [
    inquirer.Text('ciudad', 
                  message="Ingrese el nombre del archivo de la ciudad",
                  validate=lambda _, x: os.path.exists(x),
                  ),
]

def solutions_report_table(solutions: list[tuple[list, int, float]], time_limit: int) -> Table:
    table = Table(title="Reporte de soluciones")
    table.add_column("Algoritmo", justify="center", style="cyan")
    table.add_column("Ruta", justify="center", style="magenta")
    table.add_column("Costo", justify="center", style="green")
    table.add_column("Tiempo (s)", justify="center")
    for solution in solutions:
        if solution[2] == None:
            table.add_row(solution[0], str(solution[1]), "No se pudo resolver", "Tiempo límite excedido")
        else:
            table.add_row(solution[0], str(solution[1]), str(solution[2]), str(solution[3]))
    return table

def print_graph(graph: nx.Graph) -> None:
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels={(u, v): d['weight'] for u, v, d in graph.edges(data=True)})
    nx.draw_networkx_labels(graph, pos)
    plt.axis('off')
    plt.show()

def generate_random_graph(n_nodes: int) -> nx.Graph:
    nodes = [chr(65+i) for i in range(n_nodes)]
    edges = [(n1, n2) for n1 in nodes for n2 in nodes if n1 != n2]
    weights = [random.randint(1, 10) for _ in range(len(edges))]
    graph = nx.Graph()
    for i, edge in enumerate(edges):
        graph.add_edge(*edge, weight=weights[i])
    return graph

def file_to_graph(town: str) -> nx.Graph:
    new_graph = nx.Graph()
    with open(town, 'r') as f:
        town_set = set()
        expected_size = int(f.readline().strip())
        for line in f:
            ciudad1, ciudad2, costo = line.strip().split()
            print(ciudad1, ciudad2, costo)
            town_set.add(ciudad1)
            town_set.add(ciudad2)
            if len(town_set) > expected_size:
                raise Exception("El grafo tiene más nodos de los esperados")
            costo = int(costo)
            new_graph.add_edge(ciudad1, ciudad2, weight=costo)

    return new_graph

def test_limit(resolver: Resolver, start_node: str, timeout: int) -> tuple[list, int, float]:
    try:
        min_path, weight, execution_time = func_timeout.func_timeout(timeout, resolver.start, args=(start_node,))
        console.print(f"Solución encontrada para el problema", style="bold green")
        return min_path, weight, execution_time
    except func_timeout.FunctionTimedOut:
        console.print(f"No se pudo encontrar una solución óptima para el problema en {timeout} segundos", style="bold red")
        return ['FAIL'], None, None

def main():
    # town_answer = inquirer.prompt(QTOWN)
    # grafo = Grafo(town_answer['ciudad'])

    # town_graph = file_to_graph('ciudad.gf')

    town_graph = generate_random_graph(10)
    time_limit = 1

    resolver = Resolver(town_graph, BruteForce())
    bf_min_path, bf_min_cost, bf_time = test_limit(resolver, 'A', time_limit)
    resolver.algorithm = Voracious()
    vor_min_path, vor_min_cost, vor_time = test_limit(resolver, 'A', time_limit)
    resolver.algorithm = DynamicProgramming()
    dp_min_path, dp_min_cost, dp_time = test_limit(resolver, 'A', time_limit)

    solutions = [
        ('Brute Force', bf_min_path, bf_min_cost, bf_time),
        ('Voraz', vor_min_path, vor_min_cost, vor_time),
        ('Programación Dinámica', dp_min_path, dp_min_cost, dp_time)
    ]

    report_table = solutions_report_table(solutions, time_limit)
    print(report_table)

if __name__ == "__main__":
    main()