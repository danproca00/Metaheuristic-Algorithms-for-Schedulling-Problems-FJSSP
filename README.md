
README - Metaheuristic Algorithms for Scheduling Problems

Overview
This project addresses the Flexible Job Shop Scheduling Problem (FJSP) using various metaheuristic algorithms. It includes implementations for three major algorithms:
- 2SGA (Two-Stage Genetic Algorithm)
- JCGA (Job-Constrained Genetic Algorithm with 2D Encoding)
- TAMA (Three-stage Adaptive Memetic Algorithm)
- Plus a Statistics module for comparing the performance of the algorithms.

Files and Their Purpose:
1. 2SGA.py:
Purpose: Implements the Two-Stage Genetic Algorithm (2SGA) to solve FJSP.
Key Features:
- Stage 1: Optimizes operation sequence.
- Stage 2: Assigns operations to machines.
- Includes Gantt chart visualization and multiple crossover/mutation strategies.
Input: .fjs dataset file (flexible job shop format).
Output: Final schedule, makespan, Gantt chart.
How to Run:
1. Set the input_path variable to the full path of the .fjs file (e.g., "10a.fjs").
2. Run the file directly: python 2SGA.py

2. JCGA.py:
Purpose: Implements the Job-Constrained Genetic Algorithm with 2D Two-Dimensional Encoding (TDE) and Allowed Scheduling Job Set (ASJS) decoding.
Key Features:
- Uses precedence constraints.
- 2D chromosome encoding (machine genes + job operation matrix).
- POX crossover and reverse mutation.
- Includes Gantt chart visualization and schedule validation.
Input: .fjs dataset file.
Output: Optimal schedule with makespan and Gantt chart.
How to Run:
1. Ensure you have a .fjs benchmark file (e.g., "10a.fjs").
2. Import and use the main functions:
   from JCGA import parse_fjs_file, configure_deap
   jobs, constraints, operations = parse_fjs_file("10a.fjs")
   toolbox = configure_deap(jobs, constraints, operations)
3. Manually evolve the population and evaluate fitness, or extend with a main loop.

3. TAMA.py:
Purpose: Implements the Three-Stage Adaptive Memetic Algorithm using Reinforcement Learning (Q-Learning) for local search control.
Key Features:
- Integrates local search operators adaptively.
- Supports action logging, reward-based learning.
- Balances exploration and exploitation.
Input: .fjs dataset file.
Output: Final makespan, idle time, local search stats.
How to Run:
1. Make sure the file path in parse_fjs() matches your .fjs input.
2. Initialize and run:
   from TAMA import TAMA, parse_fjs
   jobs, num_machines = parse_fjs("10a.fjs")
   algo = TAMA(jobs, num_machines)
   algo.run()

4. Statistics.py:
Purpose: Aggregates and analyzes the results of multiple runs of all algorithms.
Key Features:
- Computes min, mean, median makespan across benchmarks.
- Generates boxplots, critical difference diagrams.
- Performs Iman-Davenport and Nemenyi statistical tests.
- Exports CSV summaries.
Input: JSON result files of the format:
- 2sga_all_runs_XX.json
- jcga_all_runs_XX.json
- final_results_tama_XX.json
Output:
- makespan_summary.csv
- avg_ranks_*.csv
- Gantt and statistical visualizations
How to Run:
1. Set the correct results_path pointing to your JSON files.
2. Run the script:
   python Statistics.py

Requirements:
Install the following Python packages: pip install deap numpy pandas matplotlib seaborn scikit-posthocs statsmodels networkx

Recommended Run Order:
1. Prepare .fjs dataset files (e.g., "10a.fjs").
2. Run the algorithms individually:
   - 2SGA.py or use its functions
   - JCGA.py as a module
   - TAMA.py via the class method
3. Save each algorithm's run results as JSON (as expected by Statistics.py).
4. Run Statistics.py to evaluate and compare the performance of the algorithms.


Notes:
- Each algorithm file is designed to be standalone but may require minimal adaptation (e.g., file paths).
- All algorithms assume well-formatted FJSP .fjs benchmark files.
- The .fjs benchmark files are located in the "SetDeDate" folder
- You can customize hyperparameters such as population size, crossover/mutation probabilities, etc., within each file.
