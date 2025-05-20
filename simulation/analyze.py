import os
import numpy as np
import matplotlib.pyplot as plt

def read_data(basedir):
    dir_name=[]
    for item in med_trace_list:
        dir_name.append("/med_"+str(item))
    for item in tail_trace_list:
        dir_name.append("/tail_"+str(item))
                       
    keepalive = []
    time = []
    predicted = []
    actual = []
                        
    for item in dir_name:
        filename=basedir+item+"/keepalive_cost_list.txt"
        with open(filename) as f:
            l_list=f.read().splitlines()
            l_list=[float(i) for i in l_list]
        keepalive.append(l_list)
        
        filename=basedir+item+"/time_list.txt"
        with open(filename) as f:
            l_list=f.read().splitlines()
            l_list=[float(i) for i in l_list]
        time.append(l_list)
        
        filename=basedir+item+"/predicted_list.txt"
        with open(filename) as f:
            l_list=f.read().splitlines()
            l_list=[float(i) for i in l_list]
        predicted.append(l_list)
        
        filename=basedir+item+"/true_list.txt"
        with open(filename) as f:
            l_list=f.read().splitlines()
            l_list=[float(i) for i in l_list]
        actual.append(l_list)
        
    return(keepalive, time, predicted, actual)

def stats(numbers):
    stats = {
        "sum": sum(numbers),
        "avg": sum(numbers)/len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "count": len(numbers),
    }
    
    percentiles = {
        "50th": np.percentile(numbers, 50),
        "90th": np.percentile(numbers, 90),
        "95th": np.percentile(numbers, 95),
    }
    
    for k, v in stats.items():
        print(f"{k}: {v}")
    for k, v in percentiles.items():
        print(f"{k} percentile: {v}")
    return stats, percentiles
    
def plot_metrics(metrics_dict, trace_id, title, ylabel):
    techniques = list(metrics_dict.keys())
    values = [metrics_dict[tech] for tech in techniques]

    plt.figure(figsize=(8, 5))
    plt.bar(techniques, values, color=["skyblue", "orange", "green"])
    plt.title(f"{title} for Trace {trace_id}")
    plt.ylabel(ylabel)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_metrics_grouped(metric_dict_per_tech, title, ylabel):
    """
    metric_dict_per_tech: dict of {tech: [metric_per_trace_i]}
    trace_ids: list of trace IDs (e.g., [249, 9])
    """
    techniques = list(metric_dict_per_tech.keys())
    num_traces = len(combined_list)
    bar_width = 0.2
    x = np.arange(num_traces)  # trace indices

    plt.figure(figsize=(10, 5))
    
    for i, tech in enumerate(techniques):
        offsets = x + (i - len(techniques) / 2) * bar_width + bar_width / 2
        plt.bar(offsets, metric_dict_per_tech[tech], width=bar_width, label=tech)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x, trace_legend_list)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
def plot_prediction(predicted_dict, trace_i, actual, chunk_size=1000):
    total_len = len(actual)
    for tech, pl in predicted_dict.items():
        if len(pl[trace_i]) != total_len:
            raise ValueError(f"Mismatch predicted {len(pl[trace_i])}, actual {total_len}")
    num_chunks = (total_len + chunk_size - 1) // chunk_size
    
    # Plotting
    for i in range(num_chunks):
        print(f"Plotting chunk {i}")
        start = i * chunk_size
        end = min(start + chunk_size, total_len)
        x = np.arange(start, end)
        
        plt.figure(figsize=(12, 5))
        plt.plot(x, actual[start:end], label='Actual Trace')
        for tech, pl in predicted_dict.items():
            plt.plot(x, pl[trace_i][start:end], label=tech)
        plt.title(f'Predicted vs Actual Trace Chunk {i+1}/{num_chunks} (Index {start}–{end})')
        # plt.xlabel('Time Step')
        # plt.ylabel('Value')
        plt.legend()
        # plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), f"../results/simulation/prewarm/chunk_{i+1}.png"))
        # input("Press Enter to continue to next chunk...")
    
if __name__ == "__main__":
    med_trace_list = [249] 
    tail_trace_list = [9]
    combined_list = med_trace_list + tail_trace_list
    trace_legend_list = [f"med_{tid}" for tid in med_trace_list] + [f"tail_{tid}" for tid in tail_trace_list]
    print(trace_legend_list)
    
    technique_list = ["keepalive", "oracle", "fft"]
    keepalive_dict = {}
    time_dict = {}
    predicted_dict = {}
    actual = []
    
    for tech in technique_list:        
        keepalive, time, predicted, cur_actual = read_data(os.path.join(os.path.dirname(__file__), f"../results/simulation/prewarm/{tech}"))
        keepalive_dict[tech] = keepalive
        time_dict[tech] = time
        predicted_dict[tech] = predicted
        if not actual:
            actual = cur_actual
        else:
            if not np.allclose(actual, cur_actual):
                raise ValueError("Actual trace not close")
    
    keepalive_avg_all = {tech: [] for tech in technique_list}
    time_avg_all = {tech: [] for tech in technique_list}
    for i, trace in enumerate(combined_list):
        print(" ")
        print(f"Trace: {combined_list[i]}, idx {i}")
        
        for j, tech in enumerate(technique_list):
            print(" ")
            print(f"Technique: {tech}")
            
            print(f"STAT: Keepalive cost")
            stat_ka, per_ka = stats(keepalive_dict[tech][i])
            print(f"STAT: Service time")
            stat_time, per_time = stats(time_dict[tech][i])
            
            keepalive_avg_all[tech].append(stat_ka["avg"])
            time_avg_all[tech].append(stat_time["avg"])
        
        # plot_prediction(predicted_dict, i, actual[i])
        
    plot_metrics_grouped(keepalive_avg_all, "Average Keepalive Cost", "Keepalive Cost")
    plot_metrics_grouped(time_avg_all, "Average Service Time", "Service Time")