import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import os
# from utils import analyze_cprofile

# model_name_list = ["bloom-560m", "bert-large-uncased"]
model_name_list = ["bloom-560m", 
                   "flan-t5-small", "flan-t5-base", "flan-t5-large", 
                   "bert-base-uncased", "bert-large-uncased"]
# runtime_config = ["base"]
full_runtime_config = ["base", "opt"]
methods = ["Load Pretrained", "Load State Dict", "Torch Load"]

save_directory = os.path.join(os.path.expanduser('~'), "Paper-prototype/Serverless-LLM-serving/figures")

def fetch_data_from_file(runtime_config=full_runtime_config):
    comparison_dir = os.path.join(os.path.dirname(__file__), f"../results/comparison")
    model_dataframes = {}
    event_order = []  # Initialize an empty list to store the order of events
    for model in model_name_list:
        for runtime in runtime_config:
            data_fname = os.path.join(comparison_dir, f"{runtime}/init-{model}")
            if runtime == "base":
                data_fname += "-mar.csv"
            else:
                data_fname += ".csv"
            df = pd.read_csv(data_fname).set_index('Event')
            df_filtered = df.drop(['Total', 'Total App'])
            model_dataframes[f"{model}-{runtime}"] = df_filtered['Duration']
            
            # If event_order is empty, populate it with the order of events from the first file read
            if not event_order:
                event_order = df_filtered.index.tolist()
                
    return model_dataframes, event_order

def get_data_fk(df, fk, fv, k):
    # Substring matching to find the correct column name
    column_name = [col for col in df.columns if k in col][0]  # Assumes at least one match exists
    print(f"Found matching column: {column_name}")
    matched_rows = df.loc[df[fk] == fv, column_name]
    assert len(matched_rows) == 1, f"Expected only one matching row, Got {matched_rows}"
    v = matched_rows.iloc[0]
    print(f"For {fk} = {fv}, {column_name} = {v}")
    return v

def draw_cprofile():
    # analyze_cprofile()
    func_color_map = {
        "TensorBase.copy()": "#FF9999",
        "TensorBase.uniform()": "#66B3FF",
        "str.startswith()": "#99FF99",
        "TensorBase.normal()": "#FFCC99",
        "load_tensor()": "#FF6666",
        "TensorBase.set()": "#CCCCFF",
        "Unpickler.load()": "#66FF66",
        "Other": "#999999"
    }
    legend_order = [
        "TensorBase.copy()",
        "TensorBase.uniform()",
        "str.startswith()",
        "TensorBase.normal()",
        "load_tensor()",
        "Other"
    ]
    legend_labels = set()  # Track which labels have been added to avoid duplicates
    
    cprofile_dir = os.path.join(os.path.dirname(__file__), f"../results/load_profile")
    fig, ax = plt.subplots(figsize=(10, 7))
    for method in methods:
        print(f"Method: {method}")
        cprofile_path = os.path.join(cprofile_dir, f'{method.replace(" ", "").lower()}.csv')
        df = pd.read_csv(cprofile_path)
        df = df.drop(6)
        # Aggregate the fourth and fifth rows' "Time" and "Percent" into the "Other" row
        other_time = df.iloc[3:5]['Time'].sum() + df[df['Func'] == 'Other']['Time'].iloc[0]
        other_percent = df.iloc[3:5]['Percent'].sum() + df[df['Func'] == 'Other']['Percent'].iloc[0]
        # Update the "Other" row
        df.loc[df['Func'] == 'Other', 'Time'] = other_time
        df.loc[df['Func'] == 'Other', 'Percent'] = other_percent
        # Drop the fourth and fifth rows
        df = df.drop(df.index[3:5]).reset_index(drop=True)
        if method == "Torch Load":
            # Combine 2nd and 3rd functions into "Other"
            other_time = df.iloc[1:3]['Time'].sum() + df[df['Func'] == 'Other']['Time'].iloc[0]
            df.loc[df['Func'] == 'Other', 'Time'] = other_time
            df = df.drop(df.index[1:3]).reset_index(drop=True)
            method = 'Deserialization'
            
        print(df)
            
        bottom=np.zeros(1)
        for i, time in enumerate(df['Time']):
            func = df['Func'][i]
            # print(legend_labels)
            label = func if func not in legend_labels else ""
            # print(label)
            if label:
                legend_labels.add(func)
            ax.bar(method, time, width=0.5, bottom=bottom, color=func_color_map.get(func, "#000000"), label=label)
            bottom += time
    # Manually create legend handles and labels based on the desired order
    handles = [plt.Rectangle((0,0),1,1, color=func_color_map[func]) for func in legend_order]
    labels = [func for func in legend_order]
    # Adding labels, title, and custom x-axis tick labels
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Time Distribution by Method and Function')
    ax.legend(handles, labels, title='Function')
    plt.savefig(os.path.join(save_directory, "motivation_cprofile.png"))
    plt.show()

def draw_motivation():
    model_dataframes, event_order = fetch_data_from_file(["base"])
    # Prepare combined data for stacked bar chart
    combined_df = pd.DataFrame(model_dataframes)
    # Reorder DataFrame columns based on the event_order
    combined_df = combined_df.loc[event_order]
    # Transpose the DataFrame for plotting
    combined_df_transposed = combined_df.T

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    combined_df_transposed.plot(kind='bar', stacked=True, ax=ax, colormap=cm.viridis)
    plt.ylabel('Duration (seconds)')
    plt.title('Combined Event Durations for Each Model Group')
    plt.xticks(rotation=45)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "motivation_latency_composition.png"))
    plt.show()

def draw_comparison():
    # Read datasets from files
    model_dataframes, event_order = fetch_data_from_file()
    print(model_dataframes)

    # Prepare combined data for stacked bar chart
    combined_df = pd.DataFrame(model_dataframes)
    # Reorder DataFrame columns based on the event_order
    combined_df = combined_df.loc[event_order]
    # Transpose the DataFrame for plotting
    combined_df_transposed = combined_df.T

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    combined_df_transposed.plot(kind='bar', stacked=True, ax=ax, colormap=cm.viridis)
    plt.ylabel('Duration (seconds)')
    plt.title('Combined Event Durations for Each Model Group')
    plt.xticks(rotation=45)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "misc_model_comparison.png"))
    plt.show()
    
def draw_sagemaker():
    sagemaker_path = os.path.join(os.path.dirname(__file__), f"../results/sagemaker/init-summary.csv")
    df = pd.read_csv(sagemaker_path)
    df['NetworkLatency'] = df['E2ELatency'] - df['ModelLatency'] - df['OverheadLatency']
    fig, ax = plt.subplots(figsize=(12, 8))
    print(df)
    bottom = np.zeros(len(df))
    for latency_t in ['ModelLatency', 'OverheadLatency', 'NetworkLatency']:
        ax.bar(df['Model Name'], df[latency_t], bottom=bottom, label=latency_t)
        bottom += df[latency_t]
    # Adding labels and title
    ax.set_ylabel('Latency (s)')
    ax.set_title('End-to-End Latency Breakdown by Model')
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "misc_sagemaker.png"))
    plt.show()
    
def draw_evaluation_base():
    sagemaker_path = os.path.join(os.path.dirname(__file__), f"../results/sagemaker/init-summary.csv")
    sagemaker_df = pd.read_csv(sagemaker_path)
    sagemaker_df['Total'] = sagemaker_df['ModelLatency'] + sagemaker_df['OverheadLatency']
    
    base_list = []
    opt_list = []
    sagemaker_list = []
    for model in model_name_list:
        base_init_path = os.path.join(os.path.dirname(__file__), f"../results/comparison", f"base/init-{model}-mar.csv")
        opt_init_path = os.path.join(os.path.dirname(__file__), f"../results/comparison", f"opt/init-{model}.csv")
        
        base_df = pd.read_csv(base_init_path)
        opt_df = pd.read_csv(opt_init_path)
        base_list.append(base_df.loc[base_df['Event'] == 'Total', 'Duration'].iloc[0])
        opt_list.append(opt_df.loc[opt_df['Event'] == 'Total App', 'Duration'].iloc[0])
        
        matched_rows = sagemaker_df.loc[sagemaker_df['Model Name'] == model]
        if not matched_rows.empty:
            sagemaker_list.append(matched_rows['Total'].values[0])
        else:
            sagemaker_list.append(0)
    print(base_list)
    print(opt_list)
    print(sagemaker_list)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 7))

    # Define bar width
    bar_width = 0.25

    # Set position of bar on X axis
    r1 = np.arange(len(base_list))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Make the plot
    ax.bar(r1, opt_list, color='b', width=bar_width, edgecolor='grey', label='FaServe')
    ax.bar(r2, base_list, color='r', width=bar_width, edgecolor='grey', label='KServe')
    ax.bar(r3, sagemaker_list, color='g', width=bar_width, edgecolor='grey', label='SageMaker')

    # Add xticks on the middle of the group bars
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Duration (seconds)', fontweight='bold')
    ax.set_title('Evaluation Comparison by Model')
    ax.set_xticks([r + bar_width for r in range(len(base_list))])
    ax.set_xticklabels(model_name_list)

    # Create legend & Show graphic
    ax.legend()
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(save_directory, "evaluation_comparison_base.png"))
    plt.show()
    
def draw_inference():
    # Base inference vs load
    inf_path = os.path.join(os.path.dirname(__file__), f"../results/inference/opt.csv")
    inf_df = pd.read_csv(inf_path, sep='\t')
    
    inf_list = []
    load_list = []
    
    for model in model_name_list:
        base_load_path = os.path.join(os.path.dirname(__file__), f"../results/comparison/base/init-{model}-mar.csv")
        base_load_df = pd.read_csv(base_load_path)
        print(base_load_df)
        
        inf_list.append(round(get_data_fk(inf_df, 'Model', model, 'Inference') / 1000, 3))  # ms to s
        load_list.append(get_data_fk(base_load_df, 'Event', 'Total', 'Duration'))
    
    print(inf_list)
    print(load_list)
    
    fig, ax1 = plt.subplots(figsize=(10, 7))
    
    # Calculate the ratio of load latency to inference latency for each model
    latency_ratio = [load / inference for load, inference in zip(load_list, inf_list)]

    # Define bar width
    bar_width = 0.35

    # Set position of bar on X axis
    r1 = np.arange(len(model_name_list))
    r2 = [x + bar_width for x in r1]

    # Bar plot for inference and load latency
    ax1.bar(r1, inf_list, color='b', width=bar_width, edgecolor='grey', label='Inference Latency (s)')
    ax1.bar(r2, load_list, color='r', width=bar_width, edgecolor='grey', label='Load Latency (s)')

    # Setting the y-axis to log scale for latency values
    ax1.set_yscale('log')
    ax1.set_xlabel('Model', fontweight='bold')
    ax1.set_ylabel('Latency (seconds)', fontweight='bold')

    # Second y-axis for the ratio
    ax2 = ax1.twinx()
    ax2.plot(model_name_list, latency_ratio, color='g', label='Load/Inference Latency Ratio', marker='o')
    ax2.set_ylabel('Ratio', fontweight='bold')
    ax2.set_ylim(0)  # Starts the secondary y-axis from 0
    # ax2.set_yscale('log')
    # # Adjust the y-axis range to make the line appear uniformly high
    # # This is done by setting the lower limit closer to the smallest value and the upper limit beyond the largest value
    # min_ratio, max_ratio = min(latency_ratio), max(latency_ratio)
    # ax2.set_ylim(bottom=0.8 * min_ratio, top=1.1 * max_ratio)  # Example adjustment

    
    # Setting the title, adjusting the x-axis, and adding legend
    ax1.set_title('Inference and Load Latency Comparison by Model')
    ax1.set_xticks([r + bar_width/2 for r in r1])
    ax1.set_xticklabels(model_name_list)

    # Combining legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "motivation_inference_ratio.png"))
    plt.show()
    
def draw_resource():
    res_list = []
    cpus = [1, 2, 4, 8, 16]
    for cpu in cpus:
        resource_path = os.path.join(os.path.dirname(__file__), f"../results/resource/init-flan-t5-large{cpu}cpu.csv")
        df = pd.read_csv(resource_path)
        print(df)
        res_list.append(get_data_fk(df, 'Event', 'Worker Response', 'Duration'))
    print(res_list)
    
    # Plotting the relationship between number of CPUs and load latency
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(cpus, res_list, marker='o', linestyle='-', color='b', label='Load Latency (s)')

    # Adding labels and title
    ax.set_xlabel('Number of CPUs', fontweight='bold')
    ax.set_ylabel('Load Latency (seconds)', fontweight='bold')
    ax.set_title('Load Latency vs. Number of CPUs')

    # # Adding a grid for better readability
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Setting the x-axis to logarithmic scale to better represent the CPU configurations
    ax.set_xscale('log')
    ax.set_xticks(cpus)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())  # Show actual CPU numbers, not scientific notation

    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "motivation_resource_effect.png"))
    plt.show()

def draw_chosen_trace():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), f"../results/trace/chosen_data.csv")).to_numpy()
    df = df[:, 4:]
    plt.figure(figsize=(20, 8))

    # Generating data for 1440 minutes (24 hours)
    minutes = np.arange(1, 1441)

    plt.plot(minutes, df[0], label='Sporadic', alpha=0.7)
    plt.plot(minutes, df[1], label='Bursty', alpha=0.7)
    plt.plot(minutes, df[2], label='Periodic', alpha=0.7)

    plt.title('Workload Patterns Over 24 Hours')
    plt.xlabel('Minute of Day')
    plt.ylabel('Activity Level')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_directory, "evaluation_chosen_trace.png"))
    plt.show()
    
def draw_evaluation_trace_test():
    # Scatter graph
    for model in ['flan-t5-base']:
        for trace_label in ['Sporadic', 'Bursty', 'Periodic']:
            plt.figure(figsize=(10, 5))
            for runtime in ['opt', 'sagemaker']:
                if runtime == 'sagemaker':
                    df = pd.read_csv(os.path.join(os.path.dirname(__file__), f"../results/trace/{runtime}-{model}-{trace_label}.csv"))
                else:
                    df = pd.read_csv(os.path.join(os.path.dirname(__file__), f"../results/trace/{runtime}-{model}-{trace_label}-1m.csv"))
                plt.scatter(df['Timestamp'].to_numpy(), df['E2ELatency'].to_numpy(), alpha=0.5, s=20, label=f"{runtime}-{trace_label}")
            plt.title(f'{trace_label} E2E Latency Over Time')
            plt.xlabel('Timestamp (minute)')
            plt.ylabel('E2E Latency (seconds)')
            plt.legend()
            plt.grid(True)
            plt.show()
    
    # CDF
    for model in ['flan-t5-base']:
        for trace_label in ['Sporadic', 'Bursty', 'Periodic']:
            plt.figure(figsize=(10, 5))
            for runtime in ['opt', 'sagemaker']:
                if runtime == 'sagemaker':
                    df = pd.read_csv(os.path.join(os.path.dirname(__file__), f"../results/trace/{runtime}-{model}-{trace_label}.csv"))
                else:
                    df = pd.read_csv(os.path.join(os.path.dirname(__file__), f"../results/trace/{runtime}-{model}-{trace_label}-1m.csv"))
                # Sorting data for CDF
                sorted_latencies = np.sort(df['E2ELatency'])
                yvals = np.arange(len(sorted_latencies))/float(len(sorted_latencies)-1)
                plt.plot(sorted_latencies, yvals, label=runtime)
            plt.title(f'{trace_label} CDF of E2E Latencies')
            plt.xlabel('E2E Latency (seconds)')
            plt.ylabel('CDF')
            plt.legend()
            plt.grid(True)
            plt.show()
    
    # Hist graph
    plt.figure(figsize=(10, 5))
    for model in ['flan-t5-base']:
        for trace_label in ['Sporadic', 'Bursty', 'Periodic']:
            for runtime in ['opt', 'sagemaker']:
                if runtime == 'sagemaker':
                    df = pd.read_csv(os.path.join(os.path.dirname(__file__), f"../results/trace/{runtime}-{model}-{trace_label}.csv"))
                else:
                    df = pd.read_csv(os.path.join(os.path.dirname(__file__), f"../results/trace/{runtime}-{model}-{trace_label}-1m.csv"))
                plt.hist(df['E2ELatency'], label=f"{runtime}-{trace_label}")
    plt.title('E2E Latency Over Time')
    plt.xlabel('E2E Latency (seconds)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # draw_motivation()
    # draw_comparison()
    # draw_cprofile()
    # draw_sagemaker()
    # draw_evaluation_base()
    # draw_inference()
    # draw_resource()
    # draw_chosen_trace()
    draw_evaluation_trace_test()