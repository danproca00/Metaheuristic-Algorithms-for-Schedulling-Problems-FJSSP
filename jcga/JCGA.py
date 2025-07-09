# JCGA (with 2D-encoding TDE, Allowed Sets Decoding, POX Crossover, Reverse Mutation)

import numpy as np
import random
from deap import base, creator, tools
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import json
from datetime import datetime
import os
import itertools
import pandas as pd
import time



# --------------------------
# 1. Parsing the .fjs file including constraints and predecessors
# --------------------------

# This function reads a .fjs benchmark file and extracts:
# - Jobs and their operations (with machines and processing times)
# - Precedence constraints between jobs (tight job constraints)
def parse_fjs_file(file_path):
    jobs, constraints, operations = {}, {}, {}
    
    # Read non-empty lines from file
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    header = list(map(int, lines[0].split()))  # First line contains job/machine counts
    current_line, job_id = 1, 1

    # ----------------------------
    # Parse operations for each job
    # ----------------------------
    while current_line < len(lines) and not lines[current_line].startswith("#") and ':' not in lines[current_line]:
        parts = list(map(int, lines[current_line].split()))
        ops, ptr = {}, 1

        # Each job has parts[0] operations
        for op_num in range(1, parts[0] + 1):
            num_options = parts[ptr]  # Number of alternative machines
            machines = parts[ptr+1:ptr+1+2*num_options:2]
            times = parts[ptr+2:ptr+1+2*num_options:2]
            ops[str(op_num)] = {'machines': machines, 'times': times}
            ptr += 2 * num_options + 1  # Move to next operation
        jobs[f'J{job_id}'] = {'operations': ops}
        operations[f'J{job_id}'] = list(ops.keys())  # e.g., ['1', '2']
        job_id += 1
        current_line += 1

    # ----------------------------
    # Parse tight job constraints: "8 : 5 6 7" => J8 must follow J5, J6, J7
    # ----------------------------
    for line in lines[current_line:]:
        if ':' in line and not line.startswith('#'):
            try:
                job_part, preds_part = line.split(':')
                job = f'J{job_part.strip()}'
                preds = [f'J{p.strip()}' for p in preds_part.strip().split()]
                constraints[job] = preds
            except ValueError:
                print(f"[WARNING] Could not parse constraint line: {line}")

    # Debug output: list constraints
    print("Parsed constraints:")
    for job, preds in constraints.items():
        print(f"{job} <- {preds}")

    return jobs, constraints, operations



# --------------------------
# 2. Calculating Constraint Levels
# --------------------------

# This function assigns a "constraint level" (also called dependency depth) to each job.
# The level indicates how deep a job is in the precedence graph.
# Level 1 = no predecessors, Level 2 = depends on Level 1, etc.
def calculate_constraint_levels(jobs, constraints):
    levels = {}  # Dictionary to store level for each job

    # Recursive helper to compute level of one job
    def get_level(job):
        if job not in constraints or not constraints[job]:  # No predecessors → level 1
            return 1
        # Max level of all predecessors + 1
        return max(get_level(pred) for pred in constraints[job]) + 1

    # Compute levels for all jobs
    for job in jobs:
        levels[job] = get_level(job)

    return levels  # Dict: {J1: 1, J2: 1, J3: 2, ...}


# --------------------------
# 3. Building the 2D-encoding (TDE)
# --------------------------

# This function creates the OS-segment of the chromosome using Two-Dimensional Encoding (TDE).
# It organizes operations by constraint level (row) and sequence (column).
def build_tde(levels, operations):
    level_jobs = defaultdict(list)  # A list of jobs per level

    # Step 1: Group job operations by their constraint level
    for job, level in levels.items():
        level_jobs[level].extend([job] * len(operations[job]))  # One entry per operation of the job

    # Determine the shape of the TDE matrix (rows = levels, cols = max number of jobs per level)
    max_level = max(level_jobs.keys())
    max_cols = max(len(v) for v in level_jobs.values())

    # Step 2: Initialize an empty 2D array (matrix) for the encoding
    tde = [[None for _ in range(max_cols)] for _ in range(max_level)]

    # Step 3: Fill the 2D matrix row by row
    for level in sorted(level_jobs.keys()):
        jobs_list = level_jobs[level]
        random.shuffle(jobs_list)  # Randomize job order within same level (diversity in OS-segment)
        for col, job in enumerate(jobs_list):
            tde[level - 1][col] = job  # Place job in correct row (level - 1)

    return tde

# --------------------------
# 4. Initializing DEAP Framework
# --------------------------

# Define a minimization fitness (DEAP expects maximization by default, so we negate it)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Define the Individual class as a list with a fitness attribute
# Each individual has two components: [machine_assignment_genes, TDE_matrix]
creator.create("Individual", list, fitness=creator.FitnessMin)

# This function initializes a single individual (chromosome) in the population
def init_individual(icls, jobs, constraints, operations):
    machine_genes = []  # 1D vector for machine assignment (MA-segment)
    
    # For each operation of each job, randomly assign a machine
    for job_id in sorted(jobs.keys()):
        for op_id in jobs[job_id]['operations']:
            machines = jobs[job_id]['operations'][op_id]['machines']
            machine_genes.append(random.choice(machines))  # random machine per operation

    # Calculate constraint levels (depth of precedence tree)
    levels = calculate_constraint_levels(jobs, constraints)

    # Build a 2D matrix encoding the order of operations respecting job precedence
    tde = build_tde(levels, operations)

    return icls([machine_genes, tde])  # return the two-part chromosome

# This function sets up the DEAP toolbox (genetic operators)
def configure_deap(jobs, constraints, operations):
    toolbox = base.Toolbox()

    # Register individual generator
    toolbox.register("individual", init_individual, creator.Individual, jobs, constraints, operations)

    # Register population generator
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register evaluation function (fitness = makespan)
    toolbox.register("evaluate", evaluate_fitness, jobs=jobs, constraints=constraints, operations=operations)

    # Register POX crossover (for TDE only)
    toolbox.register("mate", crossover_pox)

    # Register reverse mutation (for TDE only)
    toolbox.register("mutate", mutate_reverse)

    # Register selection operator (tournament of size 3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


# --------------------------
# 5. Decoding and Fitness
# --------------------------

# This function performs the decoding of the chromosome into a real schedule
# It uses the Allowed Scheduling Job Sets (ASJS) decoding method proposed in the JCGA paper
# => TYPE OF DECODING: ALLOWED SCHEDULING JOB SET (ASJS) BASED DECODING
# The decoding respects job precedence constraints and schedules only feasible operations
def decode_with_allowed_sets(individual, jobs, constraints, operations):
    # Unpack the individual: machine assignment genes and operation sequence matrix (TDE)
    machine_genes, tde = individual[0], individual[1]

    # Calculate the total number of operations in the problem
    total_ops = sum(len(operations[job]) for job in operations)

    # Pointer to current index in machine assignment gene list
    machine_ptr = 0

    # Track the progress of each job: operation index and latest completion time
    job_progress = {job: {'op_ptr': 0, 'completion': 0} for job in jobs}

    # Track availability time of each machine (when it's free to be used again)
    machine_times = defaultdict(int)

    # Final list of scheduled operations (each with Job, Operation, Machine, Start, End)
    schedule = []

    # Get how many operations each job has
    job_op_counts = {job: len(operations[job]) for job in operations}

    # Create a dictionary mapping each job to its set of predecessor jobs
    predecessors = {job: set(constraints.get(job, [])) for job in jobs}

    # Initialize the allowed scheduling set: jobs from top row of each column in TDE
    allowed_set = set()
    for col_idx in range(len(tde[0])):  # For each column
        for row_idx in range(len(tde)):  # From top to bottom
            if tde[row_idx][col_idx] is not None:
                allowed_set.add(tde[row_idx][col_idx])  # Add the first valid job
                break

    # Initialize number of scheduled operations
    scheduled_ops = 0

    # Flatten the TDE matrix to use as a fallback job sequence
    sequence = [job for row in tde for job in row if job is not None]

    # Limit the decoding loop to prevent infinite loops
    MAX_ATTEMPTS = 10000
    attempts = 0

    # Main loop: continue until all operations are scheduled
    while scheduled_ops < total_ops:
        # Count the number of iterations
        attempts += 1

        # If too many iterations without progress, abort with error
        if attempts > MAX_ATTEMPTS:
            raise Exception("[ERROR] Maximum decoding attempts exceeded. Possible infinite loop due to invalid TDE or constraints.")

        # Filter allowed jobs that are ready to schedule
        current_allowed_set = {
            job for job in allowed_set
            if job_progress[job]['op_ptr'] < job_op_counts[job]  # Has remaining operations
            and all(job_progress[p]['op_ptr'] >= job_op_counts[p] for p in predecessors[job])  # Predecessors completed
        }

        # Fallback: if no job can be scheduled, search the flattened TDE sequence
        if not current_allowed_set:
            for candidate_job in sequence:
                if job_progress[candidate_job]['op_ptr'] < job_op_counts[candidate_job]:
                    # Check if all predecessors are completed
                    preds_done = all(job_progress[p]['op_ptr'] >= job_op_counts[p] for p in predecessors[candidate_job])
                    if preds_done:
                        current_allowed_set.add(candidate_job)
                        break

        # If no job is schedulable even after fallback, raise error
        if not current_allowed_set:
            raise Exception("[ERROR] No schedulable job found. Decoding deadlock even after fallback. Check TDE and constraints.")

        # Choose the first job from the sequence that is in the allowed set
        for candidate_job in sequence:
            if candidate_job in current_allowed_set:
                current_job = candidate_job
                break

        # Determine the operation index (1-based as string)
        current_op = str(job_progress[current_job]['op_ptr'] + 1)

        # Get available machines and times for this operation
        machines = jobs[current_job]['operations'][current_op]['machines']
        times = jobs[current_job]['operations'][current_op]['times']

        # Use machine gene to select machine (with modulo for safety)
        selected_idx = machine_genes[machine_ptr] % len(machines)
        machine = machines[selected_idx]
        proc_time = times[selected_idx]

        # Determine when the operation can start (based on machine and job readiness)
        start = max(job_progress[current_job]['completion'], machine_times[machine])
        end = start + proc_time  # Compute operation end time

        # Add the scheduled operation to the list
        schedule.append({
            'Job': current_job,
            'Operation': current_op,
            'Machine': machine,
            'Start': start,
            'End': end
        })

        # Update machine availability and job progress after scheduling
        machine_times[machine] = end
        job_progress[current_job]['completion'] = end
        job_progress[current_job]['op_ptr'] += 1
        machine_ptr += 1
        scheduled_ops += 1

        # Remove the job from allowed set temporarily
        allowed_set.discard(current_job)

        # Add next job appearance in TDE to allowed set (if any)
        for row_idx in range(len(tde)):
            for col_idx in range(len(tde[0])):
                if tde[row_idx][col_idx] == current_job:
                    # Add job below (next level) if present
                    if row_idx + 1 < len(tde) and tde[row_idx + 1][col_idx] is not None:
                        allowed_set.add(tde[row_idx + 1][col_idx])
                    # Add job to the right in same level if present
                    if col_idx + 1 < len(tde[0]) and tde[row_idx][col_idx + 1] is not None:
                        allowed_set.add(tde[row_idx][col_idx + 1])
                    break  # Exit inner loop after first match
            else:
                continue  # Continue outer loop if inner loop was not broken
            break  # Break outer loop if inner loop was broken

    # Return the complete valid schedule
    return schedule



# FITNESS FUNCTION = MAKESPAN
# Evaluates how good a schedule is by calculating its total duration
def evaluate_fitness(individual, jobs, constraints, operations):
    schedule = decode_with_allowed_sets(individual, jobs, constraints, operations)
    if not schedule:
        return (float('inf'),)  # If schedule failed
    makespan = max(event['End'] for event in schedule)  # Max completion time
    return (makespan,)

# --------------------------
# 6. Crossover and Mutation
# --------------------------

# POX Crossover (Precedence Operation Crossover)
# => TYPE: POX (Precedence-based Crossover for TDE)
# Swaps partial sequences between two parents, per constraint level
def crossover_pox(ind1, ind2):
    machine_genes1, tde1 = ind1[0], ind1[1]
    machine_genes2, tde2 = ind2[0], ind2[1]

    levels = len(tde1)
    max_cols = max(len(row) for row in tde1)

    # Create new TDE matrices (offspring) initialized with None
    new_tde1 = [[None for _ in range(max_cols)] for _ in range(levels)]
    new_tde2 = [[None for _ in range(max_cols)] for _ in range(levels)]

    # For each constraint level (row)
    for level in range(levels):
        # Extract non-None jobs from the current row (sequence)
        seq1 = [job for job in tde1[level] if job is not None]
        seq2 = [job for job in tde2[level] if job is not None]

        if len(seq1) < 2:
            new_seq1, new_seq2 = seq1[:], seq2[:]  # Nothing to crossover
        else:
            # Choose a random subset of jobs from parent 1
            subset_size = random.randint(1, len(seq1) - 1)
            subset = set(random.sample(seq1, subset_size))

            # Construct new sequences by combining subsets
            new_seq1 = [job for job in seq1 if job in subset] + [job for job in seq2 if job not in subset]
            new_seq2 = [job for job in seq2 if job in subset] + [job for job in seq1 if job not in subset]

        # Place the new sequences back into offspring TDE matrices
        for idx, job in enumerate(new_seq1):
            new_tde1[level][idx] = job
        for idx, job in enumerate(new_seq2):
            new_tde2[level][idx] = job

    # Update individuals' TDE segment (OS only, MA stays the same)
    ind1[1] = new_tde1
    ind2[1] = new_tde2
    return ind1, ind2

# Reverse Mutation (segment-based reversal within rows of TDE)
# => TYPE: REVERSE-BASED MUTATION (TDE-level)
def mutate_reverse(individual):
    machine_genes, tde = individual[0], individual[1]
    levels = len(tde)
    max_cols = max(len(row) for row in tde)

    mutated_flat = []  # Flattened list of jobs after mutation

    for level in range(levels):
        # Extract valid sequence
        seq = [job for job in tde[level] if job is not None]
        if len(seq) < 2:
            mutated_flat.extend(seq)
            continue

        # Select a random segment and reverse it
        s1, s2 = sorted(random.sample(range(len(seq)), 2))
        segment = seq[s1:s2+1]
        segment.reverse()
        mutated_seq = seq[:s1] + segment + seq[s2+1:]
        mutated_flat.extend(mutated_seq)

    # Reconstruct TDE from mutated_flat
    new_tde = [[None for _ in range(max_cols)] for _ in range(levels)]
    idx = 0
    for r in range(levels):
        for c in range(max_cols):
            if tde[r][c] is not None and idx < len(mutated_flat):
                new_tde[r][c] = mutated_flat[idx]
                idx += 1

    individual[1] = new_tde
    return individual,


# --------------------------
# 7. Gantt Chart Visualization
# --------------------------
# Gantt Chart visualization of the final schedule
# Shows start and end times of each operation per machine
def generate_gantt(schedule):
    machines = sorted({op['Machine'] for op in schedule})  # All machines used
    machine_ids = {m:i for i, m in enumerate(machines)}  # Mapping: machine → y-position

    plt.figure(figsize=(14, 6))
    colors = plt.cm.tab20.colors  # Predefined color palette

    for event in schedule:
        label = f"J{event['Job'][1:]}-{event['Operation']}"  # Label: job-op
        plt.barh(
            machine_ids[event['Machine']],  # Y-position = machine
            width=event['End'] - event['Start'],  # Duration
            left=event['Start'],  # Start time on x-axis
            color=colors[int(event['Job'][1:]) % 20],  # Color by job ID
            edgecolor='black',
            label=label
        )
        # Add text inside each bar for clarity
        plt.text(
            event['Start'] + 0.5,
            machine_ids[event['Machine']],
            label,
            va='center',
            ha='left',
            fontsize=8
        )

    plt.yticks(range(len(machines)), machines)
    plt.xlabel('Time Units')
    plt.ylabel('Machines')
    plt.title('Optimal Schedule - Genetic Algorithm')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --------------------------
# 8. Validate Schedule (Order & Conflicts)
# --------------------------
def validate_decoded_schedule(schedule, jobs, verbose=False):
    """Validates the decoded schedule to ensure correct job order and no machine overlaps."""
    job_ops = defaultdict(list)
    for event in schedule:
        job = event['Job']
        op = int(event['Operation'])
        start = event['Start']
        end = event['End']
        job_ops[job].append((op, start, end))

    for job, ops in job_ops.items():
        sorted_ops = sorted(ops, key=lambda x: x[0])
        for i in range(1, len(sorted_ops)):
            prev_end = sorted_ops[i-1][2]
            curr_start = sorted_ops[i][1]
            if curr_start < prev_end:
                if verbose:
                    print(f"[INVALID] Job {job} - Op {sorted_ops[i][0]} starts at {curr_start} before prev ends at {prev_end}")
                return False

    machine_intervals = defaultdict(list)
    for event in schedule:
        machine = event['Machine']
        machine_intervals[machine].append((event['Start'], event['End'], event['Job'], event['Operation']))

    for machine, intervals in machine_intervals.items():
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        for i in range(1, len(sorted_intervals)):
            if sorted_intervals[i][0] < sorted_intervals[i-1][1]:
                if verbose:
                    print(f"[INVALID] Machine {machine} overlap between J{sorted_intervals[i-1][2][1:]}-O{sorted_intervals[i-1][3]} and J{sorted_intervals[i][2][1:]}-O{sorted_intervals[i][3]}")
                return False
    return True

# --------------------------
# 9. Save in JSON file
# --------------------------
def save_json_result(algorithm, dataset, run_id, final_makespan, convergence,
                     fitness_stats, parameters, best_schedule, notes="", output_dir="results"):
    """
    Save a complete result of one run of a metaheuristic algorithm to a JSON file.
    
    :param algorithm: Name of the algorithm (e.g. "JCGA")
    :param dataset: Dataset name (e.g. "Mk5")
    :param run_id: Integer ID of the run (e.g. 1, 2, ...)
    :param final_makespan: Final makespan value (int or float)
    :param convergence: List of makespan values per generation
    :param fitness_stats: Dict with 'min', 'max', 'avg' fitness values
    :param parameters: Dict with algorithm parameters (e.g. population, generations, etc.)
    :param best_schedule: List of dicts (each with Job, Operation, Machine, Start, End)
    :param notes: Optional string with notes about this run
    :param output_dir: Directory to save results in (default: "results")
    """
    # Create directory if not exists
    dataset_dir = os.path.join(output_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    # Assemble result dictionary
    result = {
        "algorithm": algorithm,
        "dataset": dataset,
        "run": run_id,
        "final_makespan": final_makespan,
        "convergence": convergence,
        "fitness_stats": fitness_stats,
        "parameters": parameters,
        "timestamp": datetime.now().isoformat(),
        "best_schedule": best_schedule,
        "notes": notes
    }

    # Save to JSON file
    filename = f"{algorithm.lower()}_run{run_id}.json"
    filepath = os.path.join(dataset_dir, filename)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Result saved to {filepath}")

# -----------------------
# 10. Taguchi Orthogonal Array L16 (4 factors, 4 levels)
# -----------------------

# Generate all 16 combinations for Taguchi's L16 orthogonal array
def get_l16_orthogonal_array():

    # Manual L16 orthogonal array with 4 columns and 16 rows (values in 0-3)
    oa = [
        [0, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 2, 2, 2],
        [0, 3, 3, 3],
        [1, 0, 1, 2],
        [1, 1, 0, 3],
        [1, 2, 3, 0],
        [1, 3, 2, 1],
        [2, 0, 2, 3],
        [2, 1, 3, 2],
        [2, 2, 0, 1],
        [2, 3, 1, 0],
        [3, 0, 3, 1],
        [3, 1, 2, 0],
        [3, 2, 1, 3],
        [3, 3, 0, 2]
    ]


    levels = {
        'POP_SIZE': [50, 100, 150, 200],      # Population size: 4 levels
        'MAX_GEN': [60, 80, 100, 120],        # Number of generations: 4 levels
        'CX_PROB': [0.50, 0.65, 0.80, 0.95],  # Crossover rate: 4 levels
        'MUT_PROB': [0.05, 0.10, 0.15, 0.20]  # Mutation rate: 4 levels
    }
    
    # Map index values to actual parameter values
    configs = []
    for row in oa:
        config = tuple(levels[key][row[i]] for i, key in enumerate(levels))
        configs.append(config)

    return configs




# Run JCGA once with specific parameters and return the final makespan
def run_jcga_with_params(jobs, constraints, operations, pop_size, max_gen, cx_prob, mut_prob):
    # Set up the DEAP genetic toolbox
    toolbox = configure_deap(jobs, constraints, operations)

    # Generate initial population
    population = toolbox.population(n=pop_size)

    # Evaluate initial fitness
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Main evolutionary loop
    for gen in range(max_gen):
        # Select parents
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Re-evaluate individuals with invalidated fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace old population with offspring
        population[:] = offspring

    # Return makespan of the best solution
    best = tools.selBest(population, k=1)[0]
    return best.fitness.values[0]



# Run all 16 configurations (5 runs each) and compute average makespans
def run_taguchi_experiment(fjs_path):
    # Parse .fjs file
    jobs, constraints, operations = parse_fjs_file(fjs_path)

    # Generate 16 configurations using L16 array
    configs = get_l16_orthogonal_array()
    results = []

    # Evaluate each configuration
    for i, (pop_size, max_gen, cx_prob, mut_prob) in enumerate(configs, 1):
        print(f"\n=== Taguchi Config {i}/16 ===")
        print(f"POP={pop_size}, GEN={max_gen}, CX={cx_prob}, MUT={mut_prob}")
        makespans = []

        # Repeat each configuration 5 times
        for run in range(5):
            print(f"Run {run+1}/5...")
            try:
                mksp = run_jcga_with_params(
                    jobs, constraints, operations,
                    pop_size, max_gen, cx_prob, mut_prob
                )
                makespans.append(mksp)
            except Exception as e:
                # If failure, count it as infinity (penalize)
                print(f"[ERROR] Run failed: {e}")
                makespans.append(float('inf'))

        # Compute average makespan over 5 runs
        avg_mksp = sum(makespans) / len(makespans)

        # Save result
        results.append({
            'POP_SIZE': pop_size,
            'MAX_GEN': max_gen,
            'CX_PROB': cx_prob,
            'MUT_PROB': mut_prob,
            'AVG_MAKESPAN': avg_mksp
        })

    # Convert results to DataFrame for sorting and export
    df = pd.DataFrame(results)
    df = df.sort_values(by='AVG_MAKESPAN')

    # Show all results sorted
    print("\n========== Taguchi Experiment Results ==========")
    print(df.to_string(index=False))

    # Display best configuration
    best = df.iloc[0]
    print("\nBest Configuration Found:")
    print(best)

    best_params = {
    'POP_SIZE': int(best['POP_SIZE']),
    'MAX_GEN': int(best['MAX_GEN']),
    'CX_PROB': float(best['CX_PROB']),
    'MUT_PROB': float(best['MUT_PROB'])
}
    return best_params, jobs, constraints, operations

if __name__ == "__main__":

    start_time = time.time()

     # STEP 1: Run Taguchi to get best parameter configuration
    fjs_path = r"C:\Users\Dan\Desktop\An2\Sem2+Disertatie\Disertatie\CodExplicatii\SetDeDate\test.fjs"
    best_params, jobs, constraints, operations = run_taguchi_experiment(fjs_path)


    # STEP 2: Use those parameters to run JCGA 30 times
    NUM_RUNS = 30
    ELITE_SIZE = 2  # remeber the best 2 individuals
    POP_SIZE = best_params['POP_SIZE']
    MAX_GEN = best_params['MAX_GEN']
    CX_PROB = best_params['CX_PROB']
    MUT_PROB = best_params['MUT_PROB']

    all_convergences = []
    all_runs_data = []


    # STEP 2: Set up DEAP
    toolbox = configure_deap(jobs, constraints, operations)
    best_overall = {'makespan': float('inf'), 'schedule': None, 'run': 0}

    # STEP 3: Multiple runs
    for run in range(NUM_RUNS):
        convergence = []
        print(f"\n{'='*60}\n RUN {run+1}/{NUM_RUNS} ".center(60, '#') + f"\n{'='*60}")

        # Initialize population
        population = toolbox.population(n=POP_SIZE)
        print(f"\n[INIT] Created population of {POP_SIZE} individuals")

        # Evaluate initial fitness
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Generational loop
        for gen in range(MAX_GEN):
            print(f"\n[GEN {gen+1}] Population statistics:")

            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CX_PROB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUT_PROB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid)
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            elite = tools.selBest(population, ELITE_SIZE)
            population[:] = elite + offspring[:-ELITE_SIZE]


            fits = [ind.fitness.values[0] for ind in population]
            print(f"  Min makespan: {min(fits):.1f}")
            print(f"  Avg makespan: {np.mean(fits):.1f}")
            print(f"  Max makespan: {max(fits):.1f}")
            convergence.append(min(fits))

        # Select best solution from this run
        current_best = tools.selBest(population, 1)[0]
        current_makespan = current_best.fitness.values[0]
        current_schedule = decode_with_allowed_sets(current_best, jobs, constraints, operations)

        # VALIDATE schedule correctness
        if not validate_decoded_schedule(current_schedule, jobs, verbose=True):
            print(f"[WARNING] Invalid schedule detected at run {run+1}")
        else:
            print("[VALID] Schedule passed validation check.")


        print(f"\n[RUN SUMMARY]")
        print(f"  Best makespan: {current_makespan}")
        print(f"  Previous best: {best_overall['makespan']}")

        if current_makespan < best_overall['makespan']:
            print("  NEW GLOBAL BEST FOUND!")
            best_overall = {'makespan': current_makespan, 'schedule': current_schedule, 'run': run+1}

        all_convergences.append(convergence)

        # Save full data for this run
        fits = [ind.fitness.values[0] for ind in population]
        
        run_summary = {
            "run": run + 1,
            "final_makespan": float(current_makespan),
            "convergence": convergence,
            "parameters": {
                "population_size": POP_SIZE,
                "generations": MAX_GEN,
                "crossover_prob": CX_PROB,
                "mutation_prob": MUT_PROB,
            },
            "timestamp": datetime.now().isoformat()
        }

        all_runs_data.append(run_summary)

    # STEP 4: Save all results to JSON
    output_path = r"C:\Users\Dan\Desktop\An2\Sem2+Disertatie\Disertatie\CodExplicatii\jcga\Results\jcga_all_runs_test.json"
    with open(output_path, "w") as f:
        json.dump({
            "all_runs": all_runs_data,
            "average_makespan": np.mean([r['final_makespan'] for r in all_runs_data]),
            "best_run": min(all_runs_data, key=lambda r: r['final_makespan'])
        }, f, indent=2)

    print(f"\n All runs saved to: {output_path}")

    # STEP 5: Final best solution
    print(f"\n{' FINAL RESULTS: '}")
    print(f"Best makespan achieved: {best_overall['makespan']}")
    print(f"Found in run {best_overall['run']}")
    print("\nBest schedule details:")
    for event in best_overall['schedule']:
        print(f"J{event['Job'][1:]}-{event['Operation']} on M{event['Machine']}: Start={event['Start']:3d} End={event['End']:3d}")

    # Gantt chart for best run
    print("\nGenerating Gantt chart...")
    generate_gantt(best_overall['schedule'])

    # Plot convergence for all runs
    plt.figure(figsize=(12, 6))
    for i, conv in enumerate(all_convergences):
        plt.plot(conv, label=f'Run {i+1}', alpha=0.5)
    plt.title('Convergence of Makespan per Run')
    plt.xlabel('Generation')
    plt.ylabel('Makespan')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
