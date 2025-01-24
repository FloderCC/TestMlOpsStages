import argparse
from experiment import run_experiment

nodes = [
    ["1", "1"], # 1: Flux 3, Waiting R1

    ["1", "8"], # 1: Flux 2, Waiting R1

    ["1", "16"], # 1: Flux 3, Waiting R1

    ["4", "1"], # 1: Flux 2, Waiting R1
    ["4", "8"], # 1: Flux 2, RUNNING R1

    ["4", "16"], # 1: Flux 3, RUNNING R2
    ["8", "1"], # 1: Flux 3, Waiting R1

    ["8", "8"], # 1: Flux 2, RUNNING R2

    ["8", "16"] # 1: Flux 3, DONE R2
]

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with specified node.")
    parser.add_argument("current_node", type=int, help="Index of the current node to run the experiment on.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    current_node_index = args.current_node - 1

    print(f"Running experiment on node {current_node_index + 1} with {nodes[current_node_index][0]} CPUs and {nodes[current_node_index][1]} RAM.")

    # Executes the experiment
    run_experiment(f"results_node_{current_node_index + 1}", nodes[current_node_index][0], nodes[current_node_index][1], f"Node {current_node_index + 1}")