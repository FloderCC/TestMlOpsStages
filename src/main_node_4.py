from experiment import run_experiment

"""
| ID  |    hostname    |      server     |   ip_address   |    vCPU    |     RAM     |
|-----|----------------|-----------------|----------------|------------|-------------|
| 653 |  vm-hCPU-hRAM  |       flux      |  10.255.32.61  |      8     |     16GB    |
| 654 |  vm-lCPU-hRAM  |       flux      |  10.255.32.58  |      2     |     16GB    |
| 655 |  vm-hCPU-lRAM  |       flux      |  10.255.32.54  |      8     |     1GB     |
| 656 |  vm-lCPU-lRAM  |       flux      |  10.255.32.81  |      2     |     1GB     |
"""

node_cpus = 2
node_ram = "1"
node_name = "Node 4"

# Executes the experiment
run_experiment("results_node_4", node_cpus, node_ram, node_name)