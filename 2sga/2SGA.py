# -*- coding: utf-8 -*-
import random  # Importing the random module for shuffling and selecting random values
import numpy as np  # Importing NumPy for numerical calculations
import matplotlib.pyplot as plt  # Importing Matplotlib for visualization
from deap import base, creator, tools # Importing DEAP for genetic algorithms
from collections import defaultdict  # Importing defaultdict for storing schedules efficiently
import logging  # Importing logging for debugging
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import time
from datetime import datetime
import json  


# ---------------------------
# Configuring logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Setting up logging format
logger = logging.getLogger(__name__)  # Creating a logger instance
input_path = "C:\\Users\\Dan\\Desktop\\An2\\Sem2+Disertatie\\Disertatie\\CodExplicatii\\SetDeDate\\test.fjs"
# ---------------------------
# Function to parse input file
# ---------------------------
def parse_input(file_path):
    print("Parsing input file...")
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    n_jobs, n_machines = map(int, lines[0].split())
    jobs = []

    op_to_global = {}      # (job, op) -> global id
    global_to_op = {}      # global id -> (job, op)
    global_op_id = 0       # counter for global operation ID

    print(f"Number of jobs: {n_jobs}, Number of machines: {n_machines}")

    for j, line in enumerate(lines[1:n_jobs+1]):
        parts = list(map(int, line.split()))
        num_operations = parts[0]
        operations = []
        i = 1

        for o in range(num_operations):
            num_machines = parts[i]
            machines = []
            i += 1

            for _ in range(num_machines):
                machine = parts[i] - 1
                time = parts[i + 1]
                machines.append((machine, time))
                i += 2
            operations.append(machines)

            # ADD: mapping between (job, op) and global id
            op_to_global[(j, o)] = global_op_id
            global_to_op[global_op_id] = (j, o)
            global_op_id += 1

        jobs.append(operations)

    precedence = {}
    for line in lines[n_jobs+1:]:
        if ':' in line:
            op, preds = line.split(':')
            precedence[int(op)-1] = [p-1 for p in map(int, preds.split())]

    print("Finished parsing input file.")
    return jobs, n_machines, precedence, op_to_global, global_to_op



def test_mapping(file_path):
    jobs, n_machines, precedence, op_to_global, global_to_op = parse_input(file_path)

    print("\n--- Mapping (job, op) -> global ID ---")
    for (job, op), global_id in op_to_global.items():
        print(f"(Job {job+1}, Operation {op+1}) -> Global Operation {global_id+1}")

    print("\n--- Mapping global ID -> (job, op) ---")
    for global_id, (job, op) in global_to_op.items():
        print(f"Global Operation {global_id+1} -> (Job {job+1}, Operation {op+1})")

# ---------------------------
# Genetic Algorithm Operators
# ---------------------------
def init_stage1_individual(icls, jobs):
    """Initialize Stage 1 individuals (job-operation sequence, no machines yet)."""
    individual = [(j, o) for j, ops in enumerate(jobs) for o in range(len(ops))]
    random.shuffle(individual)
    return icls(individual)

def init_stage2_individual(icls, stage1_individual, jobs):
    """Initialize Stage 2 individuals by assigning machines."""
    individual = []
    for job, op in stage1_individual:
        valid_machines = [m for m, _ in jobs[job][op]]
        if not valid_machines:
            raise ValueError(f"Job {job} Op {op} nu are mașini valide!")
        individual.append((job, op, random.choice(valid_machines)))
    return icls(individual)


    
def validate_individual(individual, expected_ops):
    """Verifică validitatea unui individ (fără duplicate și toate operațiile prezente)"""
    unique_ops = set((job, op) for job, op, *rest in individual)
    if len(unique_ops) != expected_ops:
        raise ValueError(f"Individual invalid: {len(unique_ops)} operații unice vs {expected_ops} așteptate")

def cx_ordered(ind1, ind2):
    """Ordered Crossover (OX) adaptat pentru indivizi care sunt liste de tuple (ex: (job, op))."""
    size = min(len(ind1), len(ind2))
    a, b = sorted(random.sample(range(size), 2))

    # Segmentul de păstrat din primul părinte
    temp1 = ind1[a:b]
    temp2 = [item for item in ind2 if item not in temp1]

    child1 = temp2[:a] + temp1 + temp2[a:]

    temp1 = ind2[a:b]
    temp2 = [item for item in ind1 if item not in temp1]

    child2 = temp2[:a] + temp1 + temp2[a:]

    ind1[:] = child1
    ind2[:] = child2
    return ind1, ind2


def mut_swap(ind):
    """Mutation by swapping two random operations."""
    i, j = random.sample(range(len(ind)), 2)
    ind[i], ind[j] = ind[j], ind[i]
    return ind,

# def cx_sequence(ind1, ind2):
#     """Crossover by swapping a sequence of operations."""
#     p1, p2 = sorted(random.sample(range(len(ind1)), 2))
#     temp1, temp2 = ind1[p1:p2], ind2[p1:p2]
#     new1 = [g for g in ind1 if g not in temp2] + temp2
#     new2 = [g for g in ind2 if g not in temp1] + temp1
#     return creator.Individual(new1), creator.Individual(new2)

# def mut_swap(ind):
#     """Mutation by swapping two random operations."""
#     i, j = random.sample(range(len(ind)), 2)
#     ind[i], ind[j] = ind[j], ind[i]
#     return ind,


def assignment_crossover(ind1, ind2):
    """
    Assignment Crossover Operator (ACO) for Stage 2 individuals in FAJSP.
    Each gene is a tuple: (job, op, machine).
    This function:
        - Keeps the job-op sequence fixed (from parents),
        - Swaps machine assignments between ind1 and ind2 at a random slice.
    """
    size = len(ind1)
    p1, p2 = sorted(random.sample(range(size), 2))  # crossover points

    # Extract machine assignments
    machines1 = [gene[2] for gene in ind1]
    machines2 = [gene[2] for gene in ind2]

    # Swap the machine assignments in the selected slice
    for i in range(p1, p2):
        machines1[i], machines2[i] = machines2[i], machines1[i]

    # Reconstruct the individuals with original (job, op) but new machine assignments
    ind1[:] = [(ind1[i][0], ind1[i][1], machines1[i]) for i in range(size)]
    ind2[:] = [(ind2[i][0], ind2[i][1], machines2[i]) for i in range(size)]

    return ind1, ind2



def mutation_stage2(individual, jobs, prob_aam=0.5):
    """
    Combined Assignment Altering Mutation (AAM) and Operation Swapping Mutation (OSM).
    With probability prob_aam, apply AAM, otherwise OSM.
    """
    ind = individual.copy()

    if random.random() < prob_aam:
        # AAM: Change machine assignment of one random operation
        idx = random.randrange(len(ind))
        job, op, current_machine = ind[idx]
        valid_machines = [m for m, _ in jobs[job][op] if m != current_machine]
        if valid_machines:
            new_machine = random.choice(valid_machines)
            ind[idx] = (job, op, new_machine)
    else:
        # OSM: Swap two random operations (entire gene triplet swap)
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]

    return ind,




# ---------------------------
# Function to Compute Schedule
# ---------------------------
def decode_with_allowed_set(individual, jobs, precedence, stage, op_to_global, global_to_op,
                            setup_time_same_machine=0, setup_time_diff_machine=0, lag_time=0,
                            release_date=0, batch_size=1):
    """
    Decodes a chromosome into a schedule based on Defersha's 2SGA logic with full compliance.
    Diagnostic print statements are included to detect deadlocks or infinite loops.
    """

    expected_ops = sum(len(ops) for ops in jobs)
    try:
        validate_individual(individual, expected_ops)
    except ValueError as e:
        logger.error(f"Invalid individual: {str(e)}")
        return float('inf'), None

    n_operations = sum(len(ops) for ops in jobs)
    machine_times = defaultdict(list)
    op_times = {}
    scheduled_operations = set()

    chromosome = individual.copy()
    max_iterations = 5 * n_operations
    iteration = 0

    # Debug: check for duplicate genes
    if len(set(chromosome)) != len(chromosome):
        print("[WARNING] Chromosome contains duplicate genes!")
        print(f"Chromosome: {chromosome}")

    while len(scheduled_operations) < n_operations:
        iteration += 1
        print(f"[Stage {stage}] Iteration {iteration}: scheduled {len(scheduled_operations)} / {n_operations}")

        if iteration > max_iterations:
            print(f"[FATAL] Infinite loop detected after {iteration} iterations.")
            return float('inf'), None

        # Step 1: Build Allowed Set
        allowed_set = []
        for gene in chromosome:
            job, op = gene[:2]
            if (job, op) in scheduled_operations:
                continue
            global_id = op_to_global[(job, op)]
            preds = precedence.get(global_id, [])

            # Ensure that operations within the same job are executed in order:
            # For any operation that is not the first (op > 0), allow scheduling
            # only if the immediate preceding operation (op - 1) has already been scheduled.
            # This guarantees the sequential execution of job operations.
            if op > 0 and (job, op - 1) not in scheduled_operations:
                continue

            if all(global_to_op[pred] in scheduled_operations for pred in preds):
                allowed_set.append(gene)


        if not allowed_set:
            print("[ERROR] Deadlock detected: allowed set is empty.")
            print(f"Remaining chromosome: {chromosome}")
            print(f"Scheduled so far: {scheduled_operations}")
            return float('inf'), None

        # Step 2: Pick next schedulable operation
        for gene in chromosome:
            if gene in allowed_set:
                selected_gene = gene
                break

        job, op = selected_gene[:2]

        # Step 3: Determine machine and schedule
        if stage == 1:
            best_machine, best_start, best_end = None, None, float('inf')
            for m, t in jobs[job][op]:
                r_m = len(machine_times[m]) + 1
                start = calculate_start_time(
                    job, op, m, r_m, machine_times, op_times, precedence,
                    op_to_global, global_to_op,
                    setup_time_same_machine, setup_time_diff_machine, lag_time, release_date
                )
                end = start + batch_size * t
                if end < best_end:
                    best_machine, best_start, best_end = m, start, end
            machine, start, end = best_machine, best_start, best_end
        else:
            machine = selected_gene[2]
            durations = {m: t for m, t in jobs[job][op]}
            if machine not in durations:
                print(f"[ERROR] Invalid machine {machine} for J{job+1} O{op+1}")
                return float('inf'), None
            t = durations[machine]
            r_m = len(machine_times[machine]) + 1
            start = calculate_start_time(
                job, op, machine, r_m, machine_times, op_times, precedence,
                op_to_global, global_to_op,
                setup_time_same_machine, setup_time_diff_machine, lag_time, release_date
            )
            end = start + batch_size * t

        # Step 4: Save operation
        machine_times[machine].append((start, end, job, op))
        op_times[(job, op)] = (start, end)
        scheduled_operations.add((job, op))
        chromosome.remove(selected_gene)



    makespan = max(end for tasks in machine_times.values() for (_, end, _, _) in tasks)
    print(f"[Stage {stage}] Finished scheduling. Makespan: {makespan}")
    return makespan, machine_times



def calculate_start_time(job, op, machine, r_m, machine_times, op_times, precedence,
                         op_to_global, global_to_op,
                         setup_same, setup_diff, lag, release):
    """
    Calculates the start time for an operation
    """

    global_id = op_to_global[(job, op)]
    preds = precedence.get(global_id, [])

    # End times of all predecessors (inter-job and intra-job)
    preds_end = [op_times[global_to_op[p]][1] for p in preds if global_to_op[p] in op_times]
    intra_job_pred_end = op_times.get((job, op - 1), (0, 0))[1] if op > 0 else 0
    latest_pred_end = max(preds_end + [intra_job_pred_end], default=0)

    last_machine_end = machine_times[machine][-1][1] if machine_times[machine] else 0

    # Case 1: first op of job, first on machine
    if r_m == 1 and op == 0:
        return max(release, latest_pred_end + setup_same + lag)

    # Case 2: first op of job, not first on machine
    if r_m == 1 and op > 0:
        return max(release + setup_same + latest_pred_end,
                   latest_pred_end + setup_diff + lag)

    # Case 3: not first on machine, first in job
    if r_m > 1 and op == 0:
        return last_machine_end + setup_same

    # Case 4: general case
    if r_m > 1 and op > 0:
        return max(last_machine_end + setup_same,
                   latest_pred_end + setup_diff + lag)

    # Fallback
    return last_machine_end


# ---------------------------
# Gantt Chart Visualization
# ---------------------------
def plot_gantt(schedule):  # Function to visualize Gantt chart
    print("Generating Gantt Chart...")
   # Extract machine list and sort them
    machines = sorted(schedule.keys())
    machine_ids = {m: i for i, m in enumerate(machines)}  # Assign indices to machines

    # Create figure
    plt.figure(figsize=(14, 6))
    colors = plt.cm.get_cmap('tab20', 20).colors  # Get a color palette with 20 unique colors

    # Iterate through schedule and plot each operation as a horizontal bar
    for machine, tasks in schedule.items():
        for start, end, job, op in tasks:
            label = f"J{job+1}-O{op+1}"  # Label format: J1-O1, J2-O3, etc.

            plt.barh(
                machine_ids[machine],  # Y-position (machine index)
                width=end - start,  # Duration (end - start time)
                left=start,  # Start time of the job
                color=colors[job % 20],  # Assign color based on job ID
                edgecolor='black',
                label=label if (machine, start) not in locals() else "_nolegend_"  # Avoid duplicate legends
            )

            # Add text label inside the bar
            plt.text(
                start + (end - start) / 2,  # Position label at center of bar
                machine_ids[machine],  # Align with machine row
                label,
                va='center',
                ha='center',
                fontsize=8,
                color='white' if (end - start) > 3 else 'black',  # Contrast text color
                fontweight='bold'
            )

    # Configure plot appearance
    plt.yticks(range(len(machines)), [f"Machine {m+1}" for m in machines])  # Set machine labels
    plt.xlabel('Time Units')
    plt.ylabel('Machines')
    plt.title('Optimal Schedule - Genetic Algorithm')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    print("Gantt Chart generated.")

print("Job Scheduling Algorithm is ready to execute.")

def print_schedule(schedule, stage_name):
    """Prints a formatted schedule table for a given scheduling stage."""
    print(f"\n[SCHEDULE] Generating detailed timeline for {stage_name}")
    print("Job  Op   Machine Start   End     Duration")
    schedule_list = []

    # Extract and format schedule data
    for machine, tasks in schedule.items():
        for start, end, job, op in tasks:
            schedule_list.append((job+1, op+1, machine+1, start, end, end-start))

    # Sort by start time
    schedule_list.sort(key=lambda x: x[3])

    # Print the formatted table
    for job, op, machine, start, end, duration in schedule_list:
        print(f"J{job:<3} O{op:<3} M{machine:<3} {start:<6} {end:<6} {duration:<6}")


# ---------------------------
# Validate Decoded Schedule
# ---------------------------
def validate_decoded_schedule(machine_times, jobs, verbose=True):
    """
    Validates the decoded schedule from `machine_times`.
    Ensures:
      1. Operations within a job follow correct order (precedence).
      2. No machine processes two jobs at the same time (no overlaps).
    """
    op_start_end_machine = dict()
    for machine, tasks in machine_times.items():
        for start, end, job, op in tasks:
            op_start_end_machine[(job, op)] = (start, end, machine)

    for job_id, job_ops in enumerate(jobs):
        for op_id in range(1, len(job_ops)):
            prev_op = (job_id, op_id - 1)
            curr_op = (job_id, op_id)
            if prev_op not in op_start_end_machine or curr_op not in op_start_end_machine:
                if verbose:
                    print(f"[INVALID] Missing operation: {prev_op} or {curr_op}")
                return False
            prev_end = op_start_end_machine[prev_op][1]
            curr_start = op_start_end_machine[curr_op][0]
            if curr_start < prev_end:
                if verbose:
                    print(f"[INVALID] Job {job_id} Op {op_id} starts at {curr_start} before Op {op_id-1} ends at {prev_end}")
                return False

    machine_intervals = defaultdict(list)
    for (job, op), (start, end, machine) in op_start_end_machine.items():
        machine_intervals[machine].append((start, end, (job, op)))

    for machine, intervals in machine_intervals.items():
        intervals.sort(key=lambda x: x[0])
        for i in range(1, len(intervals)):
            prev_end = intervals[i - 1][1]
            curr_start = intervals[i][0]
            if curr_start < prev_end:
                if verbose:
                    print(f"[INVALID] Overlap on machine {machine} between {intervals[i - 1][2]} and {intervals[i][2]}")
                return False

    if verbose:
        print("[VALID] Schedule passed all checks.")
    return True

# ---------------------------
# Main Algorithm
# ---------------------------
# Complete run_one_instance() with JSON-compatible output and parameter passing
def run_one_instance(args):
    run_index, POP_SIZE, STAGE1_GEN, STAGE2_GEN, CX_PROB, MUT_PROB, TOURN_SIZE, ELITE_SIZE = args


    try:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
    except RuntimeError:
        pass

    random.seed(run_index)

    # Load data
    jobs, n_machines, precedence, op_to_global, global_to_op = parse_input(input_path)

    # Stage 1
    pop1 = [init_stage1_individual(creator.Individual, jobs) for _ in range(POP_SIZE)]
    toolbox1 = base.Toolbox()
    toolbox1.register("evaluate", lambda ind: (decode_with_allowed_set(ind, jobs, precedence, 1, op_to_global, global_to_op)[0],))
    toolbox1.register("select", tools.selTournament, tournsize=TOURN_SIZE)
    toolbox1.register("mate", cx_ordered)
    toolbox1.register("mutate", mut_swap)

    fitnesses = list(map(toolbox1.evaluate, pop1))
    for ind, fit in zip(pop1, fitnesses):
        ind.fitness.values = fit

    for _ in range(STAGE1_GEN):
    # Elitism cu fallback dacă populația e prea mică
        elite = tools.selBest(pop1, min(ELITE_SIZE, len(pop1)))
        
        # Restul indivizilor (excludem elitele)
        offspring = toolbox1.select(pop1, len(pop1) - len(elite))
        offspring = list(map(toolbox1.clone, offspring))

        # Crossover și mutație
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox1.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox1.mutate(mutant)
                del mutant.fitness.values

        # Evaluăm indivizii noi
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox1.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        # Recompunem populația
        best_stage1 = tools.selBest(pop1, 1)[0]



    # Stage 2
    k_percent = 0.3
    top_k = int(POP_SIZE * k_percent)
    selected_stage1 = tools.selBest(pop1, top_k)
    pop2 = [init_stage2_individual(creator.Individual, ind, jobs) for ind in selected_stage1]

    toolbox2 = base.Toolbox()
    toolbox2.register("evaluate", lambda ind: (decode_with_allowed_set(ind, jobs, precedence, 2, op_to_global, global_to_op)[0],))
    toolbox2.register("select", tools.selTournament, tournsize=TOURN_SIZE)
    toolbox2.register("mate", assignment_crossover)
    toolbox2.register("mutate", mutation_stage2, jobs=jobs)

    fitnesses = list(map(toolbox2.evaluate, pop2))
    for ind, fit in zip(pop2, fitnesses):
        ind.fitness.values = fit

    convergence = []
    for _ in range(STAGE2_GEN):
        elite2 = tools.selBest(pop2, min(ELITE_SIZE, len(pop2)))
        offspring = toolbox2.select(pop2, len(pop2) - len(elite2))
        offspring = list(map(toolbox2.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox2.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox2.mutate(mutant)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox2.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        best_stage2 = tools.selBest(pop2, 1)[0]



        best = tools.selBest(pop2, 1)[0]
        makespan, _ = decode_with_allowed_set(best, jobs, precedence, 2, op_to_global, global_to_op)
        convergence.append(makespan)

    final_best = tools.selBest(pop2, 1)[0]
    final_makespan, _ = decode_with_allowed_set(final_best, jobs, precedence, 2, op_to_global, global_to_op)

    # Convert best individuals to serializable format
    best_stage1_serialized = [(job, op) for job, op in best_stage1]
    best_stage2_serialized = [(job, op, machine) for job, op, machine in best_stage2]


    # Return JSON-compatible result
    return {
    "run": run_index + 1,
    "final_makespan": final_makespan,
    "convergence": convergence,
    "best_individual": best_stage2_serialized,
    "best_stage1": best_stage1_serialized,
    "parameters": {
        "population_size": POP_SIZE,
        "stage1_generations": STAGE1_GEN,
        "stage2_generations": STAGE2_GEN,
        "crossover_prob": CX_PROB,
        "mutation_prob": MUT_PROB,
        "tournament_size": TOURN_SIZE
    },
    "timestamp": datetime.now().isoformat()
}




def main():
    NUM_RUNS = 30
    max_workers = 12

    # Parametrii algoritmului
    POP_SIZE = 50
    STAGE1_GEN = 50
    STAGE2_GEN = 70
    CX_PROB = 0.85
    MUT_PROB = 0.15
    TOURN_SIZE = max(2, int(0.5 * POP_SIZE))
    ELITE_SIZE = 2 


    start_time = time.time()

    # Pregătim argumentele pentru fiecare run
    args = [(i, POP_SIZE, STAGE1_GEN, STAGE2_GEN, CX_PROB, MUT_PROB, TOURN_SIZE, ELITE_SIZE) for i in range(NUM_RUNS)]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_one_instance, args))

    makespans = [res["final_makespan"] for res in results]
    all_convergences = [res["convergence"] for res in results]
    best_individuals = [res["best_individual"] for res in results]


    best_run_idx = np.argmin(makespans)
    best_individual = best_individuals[best_run_idx]

    # Decode best individual
    jobs, n_machines, precedence, op_to_global, global_to_op = parse_input(input_path)
    _, schedule = decode_with_allowed_set(best_individual, jobs, precedence, 2, op_to_global, global_to_op)

    print_schedule(schedule, "Best Schedule (Gantt)")
    plot_gantt(schedule)

    print("\nFinal Results:")
    print("Makespans:", makespans)
    print(f"Min: {min(makespans)}, Avg: {np.mean(makespans):.2f}, Max: {max(makespans)}")

    plt.figure(figsize=(12, 6))
    for i, conv in enumerate(all_convergences):
        plt.plot(conv, label=f'Run {i+1}', alpha=0.5)
    plt.title('Evolution of Makespan on Runs')
    plt.xlabel('Generation')
    plt.ylabel('Makespan')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    is_valid = validate_decoded_schedule(schedule, jobs, verbose=True)
    print("Is the final schedule valid?", is_valid)

    # Salvare JSON cu toți indicatorii
    all_runs_data = []
    for i in range(NUM_RUNS):
        all_runs_data.append({
            "run": i + 1,
            "final_makespan": makespans[i],
            "convergence": all_convergences[i],
            "parameters": {
                "population_size": POP_SIZE,
                "stage1_generations": STAGE1_GEN,
                "stage2_generations": STAGE2_GEN,
                "crossover_prob": CX_PROB,
                "mutation_prob": MUT_PROB,
                "tournament_size": TOURN_SIZE
            },
            "timestamp": datetime.now().isoformat()
        })

    output_path = "C:\\Users\\Dan\\Desktop\\An2\\Sem2+Disertatie\\Disertatie\\CodExplicatii\\2sga\\results\\2sga_all_runs_test.json"
    with open(output_path, "w") as f:
        json.dump(all_runs_data, f, indent=2)

    print(f"\nAll run results saved to: {output_path}")
    print(f"\nExecution time: {time.time() - start_time:.2f} seconds")



if __name__ == '__main__':
    # start_time = time.time()

    main()

