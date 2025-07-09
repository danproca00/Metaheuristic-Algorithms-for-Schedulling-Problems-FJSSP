# ===========================================
# 1. Import necessary libraries
# ===========================================
import json  # For JSON file operations
import random  # For random number generation
import numpy as np  # For numerical computations
from deap import base, creator, tools, algorithms  # DEAP framework for evolutionary algorithms
import matplotlib.pyplot as plt  # For plotting and visualization
from collections import defaultdict, Counter  # For dictionary with default values
import pandas as pd  # For data manipulation
import time

# ===========================================
# 2. DEAP type definitions
# ===========================================
# Create fitness type for minimization (makespan)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Create individual type as list with fitness attribute
creator.create("Individual", list, fitness=creator.FitnessMin)

# ===========================================
# 3. FJS problem parser
# ===========================================
def parse_fjs(filename):
    """
    Parses Flexible Job Shop problem file
    Args:
        filename: path to FJS file
    Returns:
        jobs: list of jobs with operations
        num_machines: total number of machines
    """
    jobs = []  # Initialize empty jobs list
    with open(filename, 'r') as f:  # Open file in read mode
        # Read header with job and machine counts
        header = list(map(int, f.readline().strip().split()))
        num_jobs, num_machines = header[0], header[1]  # Extract counts
        
        for _ in range(num_jobs):  # Process each job
            line = list(map(int, f.readline().strip().split()))
            num_ops = line[0]  # Number of operations for this job
            operations = []  # Store operations for current job
            pos = 1  # Current position in line
            
            for _ in range(num_ops):  # Process each operation
                num_machines_op = line[pos]  # Available machines for operation
                machines = []  # Machine IDs
                times = []  # Processing times
                for _ in range(num_machines_op):
                    machine = line[pos + 1]  # Machine ID
                    time = line[pos + 2]  # Operation duration
                    machines.append(machine)
                    times.append(time)
                    pos += 2  # Move to next machine-time pair
                pos += 1  # Skip num_machines_op value
                operations.append({'machines': machines, 'times': times})  # Add operation
            
            jobs.append(operations)  # Add job to jobs list
    return jobs, num_machines  # Return parsed data

# ===========================================
# 4. DEAP Toolbox configuration
# ===========================================
# Initialize problem data
jobs, num_machines = parse_fjs("C:/Users/Dan/Desktop/An2/Sem2+Disertatie/Disertatie/CodExplicatii/SetDeDate/test.fjs")

# def create_individual(jobs):
#     """Creates individual with structure ((job_id, op_id), machine)"""
#     individual = []  # Initialize empty individual
#     for job_id, job in enumerate(jobs):  # Process each job
#         for op_id, op in enumerate(job):  # Process each operation
#             # Choose valid machine from available options
#             chosen_machine = random.choice(op['machines'])
#             individual.append(((job_id, op_id), chosen_machine))  # Add to individual
    
#     # Shuffle operations for diversity
#     random.shuffle(individual)
#     return creator.Individual(individual)  # Return as DEAP individual

# # Configure DEAP toolbox
toolbox = base.Toolbox()  # Create toolbox instance
# toolbox.register("individual", tools.initIterate, creator.Individual, lambda: create_individual(jobs))  # Individual creation
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Population creation

# ===========================================
# 5. Evaluation function (makespan)
# ===========================================
# def evaluate(self, individual):
#     machine_times = [0] * (self.num_machines + 1)  # Track end time for each machine
#     job_times = defaultdict(int)  # Track end time for each job
#     machine_working_times = [0] * (self.num_machines + 1)  # Track working time per machine
    
#     # Process each operation in individual
#     for ((job_id, op_id), machine) in individual:
#         op = self.jobs[job_id][op_id]  # Get operation data
#         machine_idx = op['machines'].index(machine)  # Find machine index
#         time = op['times'][machine_idx]  # Get processing time
        
#         # Calculate start time (max of machine availability and job completion)
#         start = max(machine_times[machine], job_times[job_id])
#         end = start + time  # Calculate end time
#         machine_times[machine] = end  # Update machine end time
#         job_times[job_id] = end  # Update job end time
        
#         # Add working time for machine
#         machine_working_times[machine] += time
    
#     makespan = max(machine_times)  # Find overall makespan
    
#     # Calculate idle time: (makespan * num_machines) - total_working_time
#     total_working_time = sum(machine_working_times)
#     idle_time = (makespan * self.num_machines) - total_working_time
    
#     # Save idle_time as individual attribute
#     individual.idle_time = idle_time
    
#     return (makespan,)  # Return makespan as tuple

# # Register evaluation function
# toolbox.register("evaluate", evaluate, jobs=jobs, num_machines=num_machines)

# ===========================================
# 6. Genetic operators
# ===========================================
def custom_crossover(ind1, ind2):
    """Custom crossover adapted for tuple structures"""
    list1 = list(ind1)  # Convert to list
    list2 = list(ind2)  # Convert to list
    
    size = min(len(list1), len(list2))  # Get minimum size
    # Choose two cut points
    cx1, cx2 = sorted(random.sample(range(size), 2))
    
    hole1 = list1[cx1:cx2]  # Segment from first parent
    hole2 = list2[cx1:cx2]  # Segment from second parent
    
    # Build children preserving order
    temp1 = [op for op in list2 if op not in hole1]
    child1 = temp1[:cx1] + hole1 + temp1[cx1:]
    
    temp2 = [op for op in list1 if op not in hole2]
    child2 = temp2[:cx1] + hole2 + temp2[cx1:]
    
    return creator.Individual(child1), creator.Individual(child2)  # Return children

def custom_mutation(ind):
    """Mutation by reversing subsequence"""
    mutant = list(ind)  # Convert to list
    start, end = sorted(random.sample(range(len(mutant)), 2))  # Choose segment
    mutant[start:end] = reversed(mutant[start:end])  # Reverse segment
    return (creator.Individual(mutant),)  # Return mutated individual

# Register operators
toolbox.register("mate", custom_crossover)  # Crossover
toolbox.register("mutate", custom_mutation)  # Mutation
toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection

# ===========================================
# 7. Three-stage Adaptive Memetic Algorithm
# ===========================================
class TAMA:
    def __init__(self, jobs_data, num_machines, 
                 pop_size=100, max_gen=500, 
                 cxpb=0.8, mutpb=0.3, 
                 epsilon=0.2, alpha=0.1, gamma=0.9):
        
        # Algorithm parameters
        self.jobs = jobs_data
        self.num_machines = num_machines
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.cxpb = cxpb  # Crossover probability
        self.mutpb = mutpb  # Mutation probability
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = defaultdict(lambda: np.zeros(30))  # Q-table for RL
        
        # Define action space (30 possible local search actions)
        self.actions = [
            *[{'type': 'swap_pq', 'params': {'p': p, 'q': q}} for p in [1,2,3] for q in [1,2,3]],  # Swap segments
            {'type': 'swap_adjacent', 'params': {}},  # Swap adjacent operations
            *[{'type': 'insert_p', 'params': {'p': p}} for p in [1,2,3]],  # Insert segment
            *[{'type': 'reverse_p', 'params': {'p': p}} for p in [2,3,4,5,6,7]],  # Reverse segment
            {'type': 'adjust_machine', 'params': {}},  # Change machine assignment
            *[{'type': 'segment_swap_p', 'params': {'p': p}} for p in [1,2,3,4,5,6]],  # Swap segments between jobs
            *[{'type': 'segment_insert_p', 'params': {'p': p}} for p in [1,2,3]],  # Insert segment between jobs
            {'type': 'copy_machine', 'params': {}}  # Copy machine assignment
        ]

        self.action_log = [] # Log of action indices used across generations
        self._setup_deap()  # Initialize DEAP components
        self.prev_best_makespan = float('inf')  # Track previous best makespan
        self.prev_avg_idle = float('inf')  # Track previous average idle time

    def _setup_deap(self):
        """Configure DEAP toolbox for this instance"""
        self.toolbox = base.Toolbox()  # Create toolbox
        self.toolbox.register("individual", self.create_individual)  # Individual creation
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)  # Population creation
        self.toolbox.register("evaluate", self.evaluate)  # Evaluation function
        self.toolbox.register("select", tools.selTournament, tournsize=3)  # Selection method

    def create_individual(self):
        """Create individual with structure ((job_id, op_id), machine)"""
        individual = []  # Initialize empty individual
        for job_id, job in enumerate(self.jobs):  # Process each job
            for op_id, op in enumerate(job):  # Process each operation
                chosen_machine = random.choice(op['machines'])  # Random machine selection
                individual.append(((job_id, op_id), chosen_machine))  # Add operation
        random.shuffle(individual)  # Shuffle for diversity
        return creator.Individual(individual)  # Return DEAP individual

    def evaluate(self, individual):
        """Evaluate individual's makespan and idle time"""
        try:
            # Handle invalid individuals
            if individual is None or len(individual) == 0:
                if not hasattr(individual, 'idle_time'):
                    individual.idle_time = float('inf')
                return (float('inf'),)  # Return infinite makespan
            
            machine_times = [0] * (self.num_machines + 1)  # Machine end times
            job_times = defaultdict(int)  # Job end times
            machine_working_times = [0] * (self.num_machines + 1)  # Machine working times
            
            # Process each operation
            for ((job_id, op_id), machine) in individual:
                op = self.jobs[job_id][op_id]  # Get operation
                
                # Validate machine selection
                if machine not in op['machines']:
                    machine = random.choice(op['machines'])  # Fix invalid choice
                
                machine_idx = op['machines'].index(machine)  # Get machine index
                time = op['times'][machine_idx]  # Get processing time
                
                # Calculate start time
                start = max(machine_times[machine], job_times[job_id])
                end = start + time  # Calculate end time
                
                # Update times
                machine_times[machine] = end
                job_times[job_id] = end
                machine_working_times[machine] += time
            
            makespan = max(machine_times)  # Calculate makespan
            total_working_time = sum(machine_working_times)  # Total machine work time
            idle_time = (makespan * self.num_machines) - total_working_time  # Calculate idle time
            
            # Set idle_time attribute
            individual.idle_time = idle_time
            
            return (makespan,)  # Return fitness
        
        except Exception as e:
            print(f"Evaluation error: {str(e)}")  # Error handling
            if not hasattr(individual, 'idle_time'):
                individual.idle_time = float('inf')
            return (float('inf'),)  # Return infinite makespan

    def _apply_action(self, individual, action_idx):
        """Apply RL-selected local search action"""
        try:
            action_info = self.actions[action_idx]  # Get action details
            action_type = action_info['type']  # Action type
            params = action_info['params']  # Action parameters
            new_ind = list(individual.copy())  # Create copy
            
            def get_random_position(length, p):
                """Get random valid position in sequence"""
                return random.randint(0, max(0, length - p))
            
            def validate_segment(segment, p):
                """Validate segment length"""
                return len(segment) >= p
            
            # Internal operations (within same job)
            if action_type == 'swap_pq':
                p = params['p']  # First segment size
                q = params['q']  # Second segment size
                if len(new_ind) >= p + q:  # Check if possible
                    pos = get_random_position(len(new_ind), p+q)  # Random position
                    # Swap segments
                    new_ind[pos:pos+p+q] = new_ind[pos+p:pos+p+q] + new_ind[pos:pos+p]
                    
            elif action_type == 'swap_adjacent':
                if len(new_ind) >= 2:  # Need at least 2 operations
                    pos = random.randint(0, len(new_ind) - 2)  # Random position
                    # Swap adjacent operations
                    new_ind[pos], new_ind[pos+1] = new_ind[pos+1], new_ind[pos]
                    
            elif action_type == 'insert_p':
                p = params['p']  # Segment size
                if len(new_ind) >= p:  # Check if possible
                    src = get_random_position(len(new_ind), p)  # Source position
                    block = new_ind[src:src+p]  # Segment to move
                    dest = get_random_position(len(new_ind) - p, 0)  # Destination position
                    # Remove from source
                    new_ind = new_ind[:src] + new_ind[src+p:]
                    # Insert at destination
                    new_ind[dest:dest] = block
                    
            elif action_type == 'reverse_p':
                p = params['p']  # Segment size
                if len(new_ind) >= p:  # Check if possible
                    pos = get_random_position(len(new_ind), p)  # Random position
                    # Reverse segment
                    new_ind[pos:pos+p] = reversed(new_ind[pos:pos+p])
                    
            elif action_type == 'adjust_machine':
                if new_ind:  # Non-empty individual
                    pos = random.randint(0, len(new_ind)-1)  # Random position
                    job_id, op_id = new_ind[pos][0]  # Operation ID
                    # Choose new machine
                    new_machine = random.choice(self.jobs[job_id][op_id]['machines'])
                    # Update assignment
                    new_ind[pos] = ((job_id, op_id), new_machine)
            
            # External operations (between jobs)
            elif action_type == 'segment_swap_p':
                segments = self._get_segments(new_ind)  # Get job segments
                if len(segments) >= 2:  # Need at least 2 segments
                    seg1, seg2 = random.sample(segments, 2)  # Random segments
                    p = params['p']  # Segment size
                    # Validate segments
                    if validate_segment(seg1, p) and validate_segment(seg2, p):
                        pos1 = get_random_position(len(seg1), p)  # Position in first segment
                        pos2 = get_random_position(len(seg2), p)  # Position in second segment
                        # Swap segments
                        seg1[pos1:pos1+p], seg2[pos2:pos2+p] = seg2[pos2:pos2+p], seg1[pos1:pos1+p]
                        new_ind = [item for seg in segments for item in seg]  # Recombine
                        
            elif action_type == 'segment_insert_p':
                segments = self._get_segments(new_ind)  # Get job segments
                if len(segments) >= 2:  # Need at least 2 segments
                    src_seg, dest_seg = random.sample(segments, 2)  # Random segments
                    p = params['p']  # Segment size
                    if validate_segment(src_seg, p):  # Validate source
                        block = src_seg[:p]  # Segment to move
                        dest_pos = get_random_position(len(dest_seg), 0)  # Destination position
                        # Insert in destination
                        dest_seg[dest_pos:dest_pos] = block
                        del src_seg[:p]  # Remove from source
                        new_ind = [item for seg in segments for item in seg]  # Recombine
            
            elif action_type == 'copy_machine':
                if len(new_ind) >= 2:  # Need at least 2 operations
                    # Choose two random positions
                    src_pos, dest_pos = random.sample(range(len(new_ind)), 2)
                    dest_job_id, dest_op_id = new_ind[dest_pos][0]  # Destination operation
                    src_machine = new_ind[src_pos][1]  # Source machine
                    
                    # Copy machine if valid
                    if src_machine in self.jobs[dest_job_id][dest_op_id]['machines']:
                        new_ind[dest_pos] = (new_ind[dest_pos][0], src_machine)
                    else:  # Choose random machine if invalid
                        new_machine = random.choice(self.jobs[dest_job_id][dest_op_id]['machines'])
                        new_ind[dest_pos] = (new_ind[dest_pos][0], new_machine)
            
            return creator.Individual(new_ind)  # Return modified individual
        
        except Exception as e:
            print(f"Error applying action {action_type}: {str(e)}")  # Error handling
            return individual  # Return original on error

    def _get_segments(self, individual):
        """Segment individual by job ID"""
        segments = []  # Initialize segments list
        current_segment = []  # Current job segment
        current_job = None  # Track current job
        
        # Group operations by job
        for ((job_id, op_id), machine) in individual:
            if job_id != current_job:  # New job encountered
                if current_segment:  # Add previous segment if exists
                    segments.append(current_segment)
                current_segment = []  # Start new segment
                current_job = job_id  # Update current job
            current_segment.append(((job_id, op_id), machine))  # Add operation
        
        if current_segment:  # Add last segment
            segments.append(current_segment)
        return segments

    def _get_state(self, population):
        """Quantize average fitness to create discrete state"""
        try:
            avg_fitness = np.mean([ind.fitness.values[0] for ind in population])  # Average makespan
            return int(avg_fitness // 10 * 10)  # Quantize to nearest 10
        except:
            return 0  # Default state on error

    def run(self):
        """Run the evolutionary algorithm"""
        pop = self.toolbox.population(n=self.pop_size)  # Initialize population
        
        # Evaluate initial population
        for ind in pop:
            if not ind.fitness.valid:  # Check if needs evaluation
                ind.fitness.values = self.toolbox.evaluate(ind)
        
        # Setup Hall of Fame and statistics
        hof = tools.HallOfFame(1)  # Track best individual
        
        # Configure statistics
        stats = tools.Statistics()  # Statistics collector
        stats.register("min", lambda pop: np.min([ind.fitness.values[0] for ind in pop]))  # Minimum makespan
        stats.register("avg", lambda pop: np.mean([ind.fitness.values[0] for ind in pop]))  # Average makespan
        stats.register("idle", self._collect_idle_time)  # Average idle time
        
        logbook = tools.Logbook()  # Data logger
        logbook.header = ["gen", "min", "avg", "idle"]  # Log headers
        
        # Initialize previous metrics
        self.prev_best_makespan = min(ind.fitness.values[0] for ind in pop)
        self.prev_avg_idle = self._collect_idle_time(pop)
        
        ELITE_SIZE = 2  # Number of elites to preserve
        
        for gen in range(self.max_gen):  # Main generational loop
            print(f"\n Generation {gen+1}/{self.max_gen}")
            
            # Get current state
            state = self._get_state(pop)
            
            # Choose action using ε-greedy
            if random.random() < self.epsilon:  # Exploration
                action_idx = random.randint(0, 29)
                print(f" Exploration: selected random action {action_idx}")
            else:  # Exploitation
                action_idx = np.argmax(self.q_table[state])
                print(f" Exploitation: selected best action {action_idx}")
            
            # 👉 Log the action index used in this generation
            self.action_log.append(action_idx)


            # Apply action to each individual
            offspring = []
            for ind in pop:
                new_ind = self._apply_action(ind, action_idx)  # Apply local search
                
                # Check whether the individual has a 'fitness' attribute.
                # This might not be the case if the individual was created or cloned improperly.
                # If it doesn't exist, it needs to be created before evaluation.
                # Even if 'fitness' exists, it might be *invalid* (i.e., .valid == False).
                # This happens if the individual was recently modified (e.g., by mutation or crossover),
                # because DEAP automatically marks the fitness as invalid using .invalidate().
                # An invalid fitness means the individual needs to be re-evaluated to get fresh objective values.
                # Therefore, we check both: missing fitness or invalid fitness.
                if not hasattr(new_ind, 'fitness') or not new_ind.fitness.valid:
                    # Evaluate the individual using the evaluation function from the toolbox.
                    # The result (usually a tuple like (makespan,)) is assigned to fitness.values.
                    new_ind.fitness.values = self.toolbox.evaluate(new_ind)

                # After evaluation, we append the individual to the offspring list
                # so it can participate in the next generation.
                offspring.append(new_ind)

            
            # Calculate current metrics
            current_best_makespan = min(ind.fitness.values[0] for ind in offspring)
            current_avg_idle = self._collect_idle_time(offspring)
            # Calculate reward based on improvements
            reward = self._calculate_reward(current_best_makespan, current_avg_idle, gen)
            print(f" Reward: {reward:.2f} | Best makespan: {current_best_makespan:.2f} | Avg idle: {current_avg_idle:.2f}")
            
            # Update Q-table
            next_state = self._get_state(offspring)
            old_q = self.q_table[state][action_idx]  # Current Q-value
            # Best Q-value for next state
            max_next_q = np.max(self.q_table[next_state]) if next_state in self.q_table else 0
            # Update Q-value using Bellman equation
            new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
            self.q_table[state][action_idx] = new_q
            
            # Update previous metrics
            self.prev_best_makespan = current_best_makespan
            self.prev_avg_idle = current_avg_idle
            
            # Elitism: preserve best individuals
            elite = tools.selBest(pop, ELITE_SIZE)  # Select elite
            # Select remaining from combined population
            rest = self.toolbox.select(pop + offspring, self.pop_size - ELITE_SIZE)
            pop = elite + rest  # New population
            print(f" Elite preserved with makespan: {elite[0].fitness.values[0]:.2f}")
            
            # Update Hall of Fame
            hof.update(pop)
            
            # Record statistics
            record = stats.compile(pop)
            logbook.record(gen=gen, **record)
        
        return pop, hof, logbook  # Return final results

    def _collect_idle_time(self, population):
        """Collect average idle time from population"""
        idle_times = []  # Initialize list
        for ind in population:  # Process each individual
            if hasattr(ind, 'idle_time'):  # Check if attribute exists
                idle_times.append(ind.idle_time)
            else:
                idle_times.append(float('inf'))  # Default if missing
        return np.mean(idle_times) if idle_times else float('inf')  # Return average

    def _calculate_reward(self, current_makespan, current_idle, gen):
        """Calculate reward based on improvement metrics"""
        if gen == 0:  # First generation has no improvement
            return 0
        
        # Calculate improvements
        makespan_improvement = self.prev_best_makespan - current_makespan
        idle_improvement = self.prev_avg_idle - current_idle
        
        # Normalize improvements
        norm_makespan = makespan_improvement / max(1, self.prev_best_makespan)
        norm_idle = idle_improvement / max(1, self.prev_avg_idle)
        
        # Composite reward with weights
        reward = 0.7 * norm_makespan + 0.3 * norm_idle
        
        return reward * 100  # Scale reward
    
# ===========================================
# 7.1 Decoding with operation order constraint
# ===========================================
def find_earliest_slot(machine_schedule, duration, ready_time):
    """
    Finds earliest available time slot on machine that doesn't conflict 
    with scheduled operations and starts no earlier than ready_time.
    """
    if not machine_schedule:  # No operations scheduled
        return ready_time

    sorted_ops = sorted(machine_schedule, key=lambda x: x[0])  # Sort by start time

    # Case 1: Before first operation
    if ready_time + duration <= sorted_ops[0][0]:
        return ready_time

    # Case 2: Between existing operations
    for i in range(len(sorted_ops) - 1):
        prev_end = sorted_ops[i][1]  # Previous operation end
        next_start = sorted_ops[i + 1][0]  # Next operation start
        gap = next_start - prev_end  # Available gap
        if gap >= duration and ready_time <= prev_end:
            return max(ready_time, prev_end)  # Start after previous ends

    # Case 3: After last operation
    last_end = sorted_ops[-1][1]  # Last operation end
    return max(ready_time, last_end)  # Start after last ends

def decode_with_allowed_set(individual, jobs, num_machines, debug=True):
    """ Decodes individual into valid schedule respecting operation order and machine availability """
    schedule = {}  # Final schedule: {(job_id, op_id): (start, end, machine)}
    machine_ready = [0] * (num_machines + 1)  # Machine availability times
    job_ready = [0] * len(jobs)  # Job completion times

    # Sort by job and operation ID to maintain order
    individual.sort(key=lambda x: x[0])

    # Process each operation in order
    for (job_id, op_id), machine in individual:
        op_data = jobs[job_id][op_id]  # Operation data
        machine_idx = op_data['machines'].index(machine)  # Machine index
        duration = op_data['times'][machine_idx]  # Processing time
        # Calculate start time (max of machine and job availability)
        start_time = max(machine_ready[machine], job_ready[job_id])
        end_time = start_time + duration  # Calculate end time

        # Record schedule
        schedule[(job_id, op_id)] = (start_time, end_time, machine)
        # Update availability times
        machine_ready[machine] = end_time
        job_ready[job_id] = end_time

        if debug:  # Debug output
            print(f"[SCHEDULE] J{job_id}-O{op_id} on M{machine}: {start_time} → {end_time} (dur {duration})")

    return schedule  # Return schedule

# ===========================================
# 7.3 Calculate idle time for schedule
# ===========================================
def calculate_idle_time(schedule, num_machines):
    """
    Calculates total idle time for all machines.
    Args:
        schedule: dict {(job_id, op_id): (start, end, machine)}
        num_machines: number of machines
    Returns:
        idle_time: total idle time across all machines
    """
    from collections import defaultdict
    machine_intervals = defaultdict(list)  # Machine: [(start, end)]
    
    # Group operations by machine
    for (job_id, op_id), (start, end, machine) in schedule.items():
        machine_intervals[machine].append((start, end))

    idle_total = 0  # Total idle time
    for m in range(1, num_machines + 1):  # Process each machine
        intervals = sorted(machine_intervals[m], key=lambda x: x[0])  # Sort by start time
        for i in range(1, len(intervals)):  # Check gaps between operations
            idle = intervals[i][0] - intervals[i - 1][1]  # Gap duration
            if idle > 0:  # Only positive gaps
                idle_total += idle  # Accumulate idle time

    return idle_total  # Return total

# ===========================================
# 7.2 Validate decoded schedule
# ===========================================
def validate_decoded_schedule(schedule, jobs, verbose=False):
    """Validates schedule: operation order and no machine conflicts"""
    # 1. Check operation order within jobs
    for job_id, job in enumerate(jobs):
        prev_end = 0  # Track previous operation end
        for op_id in range(len(job)):
            start = schedule[(job_id, op_id)][0]  # Current start time
            if start < prev_end:  # Operation starts before previous ends
                if verbose:
                    print(f"[INVALID] Job {job_id} op {op_id} starts before op {op_id-1} finishes.")
                return False
            prev_end = schedule[(job_id, op_id)][1]  # Update end time

    # 2. Check machine conflicts
    machine_intervals = defaultdict(list)  # Machine: [(start, end, op)]
    for (job_id, op_id), (start, end, machine) in schedule.items():
        machine_intervals[machine].append((start, end, (job_id, op_id)))
    
    for machine, intervals in machine_intervals.items():
        sorted_ints = sorted(intervals, key=lambda x: x[0])  # Sort by start time
        for i in range(1, len(sorted_ints)):
            if sorted_ints[i][0] < sorted_ints[i-1][1]:  # Overlap detected
                if verbose:
                    print(f"[INVALID] Overlap on machine {machine} between {sorted_ints[i-1][2]} and {sorted_ints[i][2]}")
                return False
    return True  # Valid schedule

# ===========================================
# 7.4 Visualize Gantt Chart with Jx-Oy labels
# ===========================================
def draw_gantt_chart(schedule, num_machines, title="Gantt Chart"):
    """
    Displays Gantt chart for given schedule.
    Args:
        schedule: dict {(job_id, op_id): (start, end, machine)}
        num_machines: total number of machines
        title: chart title
    """
    fig, ax = plt.subplots(figsize=(10, 6))  # Create figure
    
    # Color map for distinct job-operation colors
    colors = plt.cm.get_cmap("tab20", len(schedule))
    machine_rows = [[] for _ in range(num_machines + 1)]  # Track machine bars

    # Compute makespan (maximum end time)
    max_time = max(end for (_, (start, end, _)) in schedule.items())

    # Plot each operation
    for idx, ((job_id, op_id), (start, end, machine)) in enumerate(schedule.items()):
        duration = end - start  # Operation duration
        # Plot bar on machine row
        ax.barh(machine, duration, left=start, height=0.4, color=colors(idx))
        # Add label at bar center
        ax.text(start + duration / 2, machine, f"J{job_id}-O{op_id}", 
                va='center', ha='center', fontsize=8, color='white', fontweight='bold')

    # Configure axes
    ax.set_xlim(0, max_time + 5)  # Limit X-axis just beyond makespan
    ax.set_yticks(range(1, num_machines + 1))  # Machine ticks
    ax.set_yticklabels([f"M{m}" for m in range(1, num_machines + 1)])  # Machine labels
    ax.set_xlabel("Time")  # X-axis label
    ax.set_ylabel("Machines")  # Y-axis label
    ax.set_title(title)  # Chart title
    ax.grid(True, linestyle="--", alpha=0.4)  # Grid lines
    plt.tight_layout()  # Adjust layout
    plt.show()  # Display chart

def analyze_action_usage(action_log, actions):
    
    action_counts = Counter(action_log)
    sorted_actions = sorted(action_counts.items())
    action_ids = [idx for idx, _ in sorted_actions]
    action_freqs = [count for _, count in sorted_actions]

    print("\n=== Action Usage Summary ===")
    for idx, count in sorted_actions:
        print(f"Action {idx:02d} ({actions[idx]['type']}, params={actions[idx]['params']}): used {count} times")

    # graph to see the most used actions per run, commented for testing
    # plt.figure(figsize=(12, 5))
    # plt.bar(action_ids, action_freqs)
    # plt.xlabel("Action Index")
    # plt.ylabel("Usage Count")
    # plt.title("Frequency of RL Actions Used")
    # plt.xticks(action_ids)
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.show()

# ===========================================
# 8. Experiment execution and visualization
# ===========================================
def main():
    # 1. Read data and define constraints
    #jobs, num_machines = parse_fjs("C:/Users/Dan/Desktop/An2/Sem2+Disertatie/Disertatie/SetDeDate/Mk05.fjs")
    
    # Precedence constraints (only within jobs)
    op_precedence = {}

    all_results = []  # Store all run results
    idle_times = []  # Store idle times
    rewards = []  # Store rewards
    makespans = []  # Store makespans

    # Run 30 experiments
    for run in range(30):
        print(f"\n=== Starting run {run+1}/30 ===")
        # Initialize algorithm
        tama = TAMA(jobs_data=jobs, num_machines=num_machines, 
                   pop_size=300, max_gen=200) 
        
        # Run algorithm
        pop, hof, logbook = tama.run()
        best_ind = hof[0]  # Best individual
        makespan = best_ind.fitness.values[0]  # Best makespan
        
        analyze_action_usage(tama.action_log, tama.actions)

        # Decode and validate schedule
        schedule = decode_with_allowed_set(best_ind, jobs, num_machines, debug=True)
        is_valid = validate_decoded_schedule(schedule, jobs, verbose=True)
        
        # Calculate idle time
        idle_time = calculate_idle_time(schedule, num_machines)
        
        # Calculate final reward
        final_stats = logbook[-1]  # Last generation stats
        reward = tama._calculate_reward(
            final_stats["min"],  # Final best makespan
            final_stats["idle"],  # Final idle time
            tama.max_gen-1  # Last generation
        )
        
        # Store run results
        run_result = {
            'run': run+1,
            'makespan': makespan,
            'idle_time': idle_time,
            'reward': reward,
            'valid': is_valid,
            'log': [dict(entry) for entry in logbook]  # Convert logbook to dict
        }
        all_results.append(run_result)
        idle_times.append(idle_time)
        rewards.append(reward)
        makespans.append(makespan)
        
        # Display summary
        print(f"  Makespan: {makespan}")
        print(f"  Idle Time: {idle_time}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Valid Schedule: {is_valid}")

    # Global statistics
    avg_makespan = np.mean(makespans)
    avg_idle = np.mean(idle_times)
    avg_reward = np.mean(rewards)
    
    # Find best run (lowest makespan)
    best_run_idx = np.argmin(makespans)
    best_run = all_results[best_run_idx]
    
    print("\n=== Final Results ===")
    print(f" - Average makespan: {avg_makespan:.2f}")
    print(f" - Average idle time: {avg_idle:.2f}")
    print(f" - Average reward: {avg_reward:.2f}")
    print(f" - Best run: #{best_run_idx+1}")
    print(f" - Makespan: {best_run['makespan']}")
    print(f" - Idle Time: {best_run['idle_time']}")
    print(f" - Reward: {best_run['reward']:.2f}")

    # Visualization of evolution
    if best_run['log']:
        plt.figure(figsize=(12, 8))  # Create figure
        
        # Makespan plot
        plt.subplot(2, 2, 1)
        plt.plot([entry['gen'] for entry in best_run['log']], 
                 [entry['min'] for entry in best_run['log']], 'b-')
        plt.xlabel('Generation')
        plt.ylabel('Makespan')
        plt.title('Makespan Evolution')
        plt.grid(True)
        
        # Idle time plot
        plt.subplot(2, 2, 2)
        plt.plot([entry['gen'] for entry in best_run['log']], 
                 [entry['idle'] for entry in best_run['log']], 'r-')
        plt.xlabel('Generation')
        plt.ylabel('Idle Time')
        plt.title('Idle Time Evolution')
        plt.grid(True)
        
        # Reward plot
        plt.subplot(2, 2, 3)
        # Calculate rewards for each generation
        rewards_plot = [tama._calculate_reward(
            best_run['log'][i]['min'],  # Makespan at generation i
            best_run['log'][i]['idle'],  # Idle time at generation i
            i  # Generation index
        ) for i in range(len(best_run['log']))]
        plt.plot([entry['gen'] for entry in best_run['log']], rewards_plot, 'g-')
        plt.xlabel('Generation')
        plt.ylabel('Reward')
        plt.title('Reward Evolution')
        plt.grid(True)
        
        # Normalized metrics comparison
        plt.subplot(2, 2, 4)
        makespans_log = [entry['min'] for entry in best_run['log']]  # Makespan values
        idle_times_log = [entry['idle'] for entry in best_run['log']]  # Idle time values
        # Normalize metrics to [0,1] range
        norm_makespan = (makespans_log - np.min(makespans_log)) / (np.max(makespans_log) - np.min(makespans_log) + 1e-10)
        norm_idle = (idle_times_log - np.min(idle_times_log)) / (np.max(idle_times_log) - np.min(idle_times_log) + 1e-10)
        plt.plot([entry['gen'] for entry in best_run['log']], norm_makespan, 'b-', label='Makespan')
        plt.plot([entry['gen'] for entry in best_run['log']], norm_idle, 'r-', label='Idle Time')
        plt.xlabel('Generation')
        plt.ylabel('Normalized Metrics')
        plt.title('Normalized Metrics Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()  # Adjust layout
        plt.savefig('evolution_metrics.png')  # Save figure
        plt.show()  # Display figure

    # Gantt chart for best solution
    print("\nGenerating Gantt Chart for best solution...")
    best_schedule = decode_with_allowed_set(hof[0], jobs, num_machines, debug=True)
    draw_gantt_chart(best_schedule, num_machines, title=f"Best Solution (Makespan: {best_run['makespan']})")
    
    # Save results
    with open('C:\\Users\\Dan\\Desktop\\An2\\Sem2+Disertatie\\Disertatie\\CodExplicatii\\tama\\results\\final_results_tama_test.json', 'w') as f:
        json.dump({
            'all_runs': all_results,  # All run results
            'average_makespan': avg_makespan,  # Average makespan
            'average_idle_time': avg_idle,  # Average idle time
            'average_reward': avg_reward,  # Average reward
            'best_run': best_run  # Best run details
        }, f, indent=2)  # Write JSON with indentation
    
    print(" Experiment complete! Results saved to final_results.json")

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    start_time = time.time()
    main()  # Execute main function