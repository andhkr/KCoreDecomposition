import random
import time


def generate_BA_graph(n, m0=5, edges_per_node=5, filename="BA_graph.txt"):
    start = time.time()
    edges = set()

    degree_list = []  # preferential attachment list
    for i in range(m0):
        for j in range(i + 1, m0):
            edges.add((i, j))
            degree_list.extend([i, j])

    for new_node in range(m0, n):
        targets = set()
        while len(targets) < edges_per_node:
            existing = random.choice(degree_list)
            if existing != new_node:
                targets.add(existing)

        for t in targets:
            edges.add((new_node, t))
            degree_list.append(t)
            degree_list.append(new_node)

        if new_node % (n // 20) == 0:
            print(f"\rProgress: {100 * new_node // n}%", end="")

    print("\rProgress: 100%")

    with open(filename, "w") as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")

    print(
        f"Generated BA graph '{filename}' with {n} nodes, {len(edges)} edges in {time.time() - start:.2f}s"
    )


if __name__ == "__main__":
    print("Barabasi-Albert scale-free graph generator")
    n = int(input("Enter number of nodes: "))
    m0 = int(input("Enter initial fully connected nodes (m0): "))
    edges_per_node = int(input("Enter edges per new node: "))

    # Estimate number of edges
    num_edges = m0 * (m0 - 1) // 2 + (n - m0) * edges_per_node
    filename = f"testGraph_{n}_{num_edges}.txt"
    generate_BA_graph(n, m0, edges_per_node, filename)

