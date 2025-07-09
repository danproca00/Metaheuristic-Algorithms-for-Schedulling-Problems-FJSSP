import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
from scipy.stats import f as f_dist, wilcoxon
from statsmodels.stats.libqsturng import qsturng

# === CONFIGURATION ===
# Combine full list of benchmarks: mk01 to mk10 and 01a to 10a
benchmarks = [f"mk{str(i).zfill(2)}" for i in range(1, 11)] + [f"{str(i).zfill(2)}a" for i in range(1, 11)]

algorithms = {
    "2SGA": "2sga_all_runs_{}.json",
    "TA-MA": "final_results_tama_{}.json",
    "JCGA": "jcga_all_runs_{}.json"
}
results_path = "C:\\Users\\Dan\\Desktop\\An2\\Sem2+Disertatie\\Disertatie\\CodExplicatii\\statistici\\Rezultate_JSON"
output_csv = os.path.join(results_path, "makespan_summary.csv")

# === Function to extract makespans from a JSON file ===
def extract_makespans(file_path):
    with open(file_path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "all_runs" in data:
        runs = data["all_runs"]
    elif isinstance(data, list):
        runs = data
    else:
        raise ValueError(f"Unexpected JSON structure in file: {file_path}")
    for key in ["makespan", "final_makespan"]:
        if key in runs[0]:
            return [r[key] for r in runs if r.get("valid", True)]
    raise ValueError(f"No recognized makespan key in file: {file_path}")

# === Collect statistics ===
tables = {"min": {}, "mean": {}, "median": {}}
all_makespans = {algo: {} for algo in algorithms}

for bench in benchmarks:
    for algo, pattern in algorithms.items():
        path = os.path.join(results_path, pattern.format(bench))
        makespans = extract_makespans(path)
        all_makespans[algo][bench] = makespans
        for metric in tables:
            if bench not in tables[metric]:
                tables[metric][bench] = {}
        tables["min"][bench][algo] = np.min(makespans)
        tables["mean"][bench][algo] = np.mean(makespans)
        tables["median"][bench][algo] = np.median(makespans)

# === Export makespan summary ===
all_flat = []
for bench in benchmarks:
    row = {"Benchmark": bench}
    for metric in ["min", "mean", "median"]:
        for algo in algorithms:
            row[f"{algo}_{metric}"] = tables[metric][bench][algo]
    all_flat.append(row)
summary_df = pd.DataFrame(all_flat)
summary_df.to_csv(output_csv, index=False)

# === Convert dictionaries to DataFrames ===
dfs = {metric: pd.DataFrame.from_dict(tables[metric], orient="index")[list(algorithms)] for metric in tables}

# === Iman-Davenport test ===
def iman_davenport(df):
    ranks = df.rank(axis=1)
    avg_ranks = ranks.mean(axis=0)
    k = len(df.columns)
    N = len(df)
    chi2_f = (12 * N) / (k * (k + 1)) * np.sum((avg_ranks - (k + 1) / 2) ** 2)
    F = ((N - 1) * chi2_f) / (N * (k - 1) - chi2_f)
    p = 1 - f_dist.cdf(F, k - 1, (k - 1) * (N - 1))
    return F, p, avg_ranks

# === Critical Difference Diagram ===
def plot_cd(avg_ranks, num_benchmarks, title="Critical Difference Diagram", save_path=None):
    k = len(avg_ranks)
    cd = 2.343 * np.sqrt(k * (k + 1) / (6.0 * num_benchmarks))
    sorted_ranks = avg_ranks.sort_values()
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_xlim(0, max(sorted_ranks) + 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    y = 0.5
    for alg, rank in sorted_ranks.items():
        ax.plot([rank, rank], [y - 0.05, y + 0.05], color='black')
        ax.text(rank, y + 0.1, alg, ha='center', fontsize=10)
    best = sorted_ranks.iloc[0]
    ax.plot([best, best + cd], [y - 0.2, y - 0.2], color='black', lw=2)
    ax.text(best + cd / 2, y - 0.3, f"CD = {cd:.2f}", ha='center', fontsize=10)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# === Statistical Analysis and Boxplots ===
import pandas as pd

boxplot_data = []
for metric, df in dfs.items():
    F, p, avg_ranks = iman_davenport(df)
    nemenyi = sp.posthoc_nemenyi_friedman(df.values)
    nemenyi.columns = df.columns
    nemenyi.index = df.columns
    avg_ranks.to_csv(os.path.join(results_path, f"avg_ranks_{metric}.csv"))
    nemenyi.to_csv(os.path.join(results_path, f"nemenyi_matrix_{metric}.csv"))
    pd.DataFrame({"F": [F], "p": [p]}).to_csv(os.path.join(results_path, f"iman_davenport_{metric}.csv"), index=False)
    plot_cd(avg_ranks, len(df), f"CD Diagram ({metric.title()} Makespan)", os.path.join(results_path, f"cd_diagram_{metric}.png"))
    df_long = df.reset_index().melt(id_vars='index', var_name='Algorithm', value_name='Makespan')
    df_long["Metric"] = metric
    boxplot_data.append(df_long)

# Combine for final boxplot display with Wilcoxon p-values
print("Makespan Summary with Metrics:")
print(summary_df)


# === Generate and save boxplots ===
import itertools
from scipy.stats import wilcoxon

def add_stat_annotation(ax, pairs, pvalues, y_offset=1.05):
    ymax = ax.get_ylim()[1]
    for i, ((a1, a2), pval) in enumerate(zip(pairs, pvalues)):
        x1, x2 = sorted([algorithms_list.index(a1), algorithms_list.index(a2)])
        y, h, col = ymax * y_offset + i * 0.05 * ymax, 0.02 * ymax, 'k'
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
        label = f"p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"
        ax.text((x1 + x2) * .5, y + h + 0.01 * ymax, label, ha='center', va='bottom', color=col)


# === Generate and save boxplots with Wilcoxon p-values ===
final_box_df = pd.concat(boxplot_data)
algorithms_list = list(algorithms)  # Ensure order is consistent: 2SGA first

for metric in final_box_df["Metric"].unique():
    df_metric = final_box_df[final_box_df["Metric"] == metric]

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df_metric, x="Algorithm", y="Makespan", order=algorithms_list, palette="Set2")
    sns.stripplot(data=df_metric, x="Algorithm", y="Makespan", order=algorithms_list,
                jitter=True, dodge=True, alpha=0.5, color='black', size=4)
    plt.title(f"Boxplot + Wilcoxon p-values — {metric.title()} Makespan")
    plt.grid(True)
    plt.tight_layout()


    # === Wilcoxon test for each pair ===
    pairs = []
    pvalues = []
    for a1, a2 in itertools.combinations(algorithms_list, 2):
        x = df_metric[df_metric["Algorithm"] == a1]["Makespan"].values
        y = df_metric[df_metric["Algorithm"] == a2]["Makespan"].values
        try:
            stat, p = wilcoxon(x, y, zero_method='wilcox', alternative='two-sided')
        except ValueError:
            p = 1.0  # fallback if inputs are constant or too small
        pairs.append((a1, a2))
        pvalues.append(p)

    add_stat_annotation(ax, pairs, pvalues)

    # Save and show
    plot_path = os.path.join(results_path, f"boxplot_{metric}_with_pvalues.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"Saved annotated boxplot with Wilcoxon p-values to: {plot_path}")

# === Plot average makespan per benchmark and save ===
# Convert to long format
df_long = pd.DataFrame()
for metric in ["min", "mean", "median"]:
    temp_df = summary_df[["Benchmark", f"TA-MA_{metric}", f"2SGA_{metric}", f"JCGA_{metric}"]].copy()

    temp_df = temp_df.melt(id_vars="Benchmark", var_name="Algorithm", value_name="Makespan")
    temp_df["Algorithm"] = temp_df["Algorithm"].str.replace(f"_{metric}", "", regex=False)
    temp_df["Metric"] = metric
    df_long = pd.concat([df_long, temp_df], ignore_index=True)

# === DEFINE BENCHMARK GROUPS ===
benchmark_groups = {
    "Low & Medium Complexity (mk01–mk07)": [f"mk{str(i).zfill(2)}" for i in range(1, 8)],
    "High Complexity (mk08–mk10)": [f"mk{str(i).zfill(2)}" for i in range(8, 11)],
    "Industrial (01a–10a)": [f"{str(i).zfill(2)}a" for i in range(1, 11)]
}

# === PLOTTING ===
for title, benchmarks in benchmark_groups.items():
    df_group = df_long[df_long["Benchmark"].isin(benchmarks)]

    plt.figure(figsize=(14, 6))
    
    # Creează graficul fără bare de eroare (ci doar bare simple)
    ax = sns.barplot(
        data=df_group,
        x="Benchmark",
        y="Makespan",
        hue="Metric",
        palette="pastel",
        dodge=True,
        errorbar=None  # <<< elimină complet barele de eroare
    )

    # Afișează valorile deasupra barelor
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', label_type='edge', padding=3)

    plt.title(f"Makespan Comparison – {title}")
    plt.ylabel("Makespan")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()



print("mean == median:", dfs["mean"].equals(dfs["median"]))
print("mean == min:", dfs["mean"].equals(dfs["min"]))
print("median == min:", dfs["median"].equals(dfs["min"]))

for metric, df in dfs.items():
    F, p, avg_ranks = iman_davenport(df)
    print(f"\n=== {metric.upper()} ===")
    print("F =", F)
    print("p =", p)
    print("Average ranks:")
    print(avg_ranks)
