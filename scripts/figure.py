import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import os
# from utils import analyze_cprofile
from matplotlib.ticker import FixedLocator
import matplotlib
import seaborn as sns

# model_name_list = ["bloom-560m", "bert-large-uncased"]
model_name_list = ["bert-base-uncased", "bert-large-uncased", 
                   "flan-t5-small", "flan-t5-base", "flan-t5-large", "bloom-560m"]
# runtime_config = ["base"]
full_runtime_config = ["base", "opt"]
methods = ["Load Pretrained", "Load State Dict"]
rename_mapping = {
    'Storage Init': 'Storage Init',
    'Unzip Model Archive': 'Model Archive Decompression',
    'Setup Model Dependency': 'Setup Model Dependency',
    'Worker Load Model': 'Load Model',
    'Other': 'Others'
}
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
    # Get colors from the viridis colormap
    viridis = plt.get_cmap('viridis')
    color_indices = np.linspace(0.1, 0.7, 7)  # Generate seven points from 0 to 1
    print(color_indices)
    viridis_colors = [viridis(2 * i) for i in color_indices]  # Map these points to colors in the colormap
    
    func_color_map = {
        "TensorBase.copy()": viridis_colors[0],
        "TensorBase.uniform()": viridis_colors[1],
        "str.startswith()": viridis_colors[2],
        "TensorBase.normal()": viridis_colors[3],
        # "load_tensor()": viridis_colors[4],
        # "TensorBase.set()": viridis_colors[5],
        # "Unpickler.load()": viridis_colors[6],
        "Other": "#999999"  # Grey for 'Other'
    }
    func_hatch_map = {
        "TensorBase.copy()": "//",  # Diagonal hatching
        "TensorBase.uniform()": "xx",  # Back diagonal
        "str.startswith()": "--",  # Horizontal lines
        "TensorBase.normal()": "++",  # Crossed
        # "load_tensor()": "xx",  # Crossed diagonal
        # "TensorBase.set()": "..",  # Dotted
        # "Unpickler.load()": "||",  # Vertical lines
        "Other": "**"  # Stars
    }
    legend_order = [
        "TensorBase.copy()",
        "TensorBase.uniform()",
        "str.startswith()",
        "TensorBase.normal()",
        # "load_tensor()",
        "Other"
    ]
    legend_labels = set()  # Track which labels have been added to avoid duplicates
    
    cprofile_dir = os.path.join(os.path.dirname(__file__), f"../results/load_profile")
    fig, ax = plt.subplots(figsize=(7, 2))
    method_mapping = {"Load State Dict": "load_state_dict()", "Load Pretrained": "from_pretrained()"}
    ypos = range(len(methods))
    
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
            
        left=np.zeros(1)
        # bottom=np.zeros(1)
        for i, time in enumerate(df['Time']):
            func = df['Func'][i]
            # print(legend_labels)
            label = func if func not in legend_labels else ""
            # print(label)
            if label:
                legend_labels.add(func)
            ax.barh(method, time, height=0.3, left=left, color=func_color_map.get(func, "#000000"), hatch=func_hatch_map[func], label=label)
            # ax.bar(method, time, width=0.5, bottom=bottom, color=func_color_map.get(func, "#000000"), label=label)
            left += time
            # bottom += time
    
    ax.set_yticks(ypos)
    ax.set_yticklabels([method_mapping[m] for m in methods])
    # ax.yaxis.set_major_locator(FixedLocator([0, -0.5]))
    # Manually create legend handles and labels based on the desired order
    handles = [plt.Rectangle((0,0),1,1, color=func_color_map[func], hatch=func_hatch_map[func]) for func in legend_order]
    labels = [func for func in legend_order]
    # Adding labels, title, and custom x-axis tick labels
    ax.set_xlabel('Duration (seconds)')
    # ax.set_title('Time Distribution by Method and Function')
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=3, frameon=False)
    # plt.tight_layout(rect=[0, 0, 1, 0.7])
    # plt.savefig(os.path.join(save_directory, "motivation_cprofile.pdf"), dpi=600)
    # TODO: fix hatch for pdf, it is a pdf issue
    plt.savefig(os.path.join(save_directory, "motivation_cprofile.pdf"), bbox_inches='tight', dpi=600, backend='pdf')
    plt.show()

def draw_motivation():
    comparison_dir = os.path.join(os.path.dirname(__file__), f"../results/comparison")
    model_dataframes = {}
    event_order = []  # Initialize an empty list to store the order of events
    runtime = "base"
    for model in model_name_list:
        data_fname = os.path.join(comparison_dir, f"{runtime}/init-{model}")
        data_fname += "-mar.csv"
        df = pd.read_csv(data_fname).set_index('Event')
        df_filtered = df.drop(['Total', 'Total App'])
        model_dataframes[f"{model}".capitalize()] = df_filtered['Duration']
        
        # If event_order is empty, populate it with the order of events from the first file read
        if not event_order:
            event_order = df_filtered.index.tolist()
                
    # Prepare combined data for stacked bar chart
    combined_df = pd.DataFrame(model_dataframes)
    # Define the stages to keep
    stages_to_keep = ['Storage Init', 'Unzip Model Archive', 'Setup Model Dependency', 'Worker Load Model']
    other_stages = [stage for stage in event_order if stage not in stages_to_keep]
    combined_df.loc['Other'] = combined_df.loc[other_stages].sum()
    # Reorder DataFrame columns based on the event_order
    combined_df = combined_df.drop(other_stages, errors='ignore')
    event_order = stages_to_keep + ['Other']
    combined_df = combined_df.loc[event_order]    
    combined_df.rename(index=rename_mapping, inplace=True)
    # Transpose the DataFrame for plotting
    combined_df_transposed = combined_df.T
    combined_df_transposed = combined_df_transposed.iloc[::-1]
    
    print(combined_df_transposed)
    print(combined_df_transposed.index)
    
    # Calculate the percentage of each stage
    percent_df = combined_df_transposed.div(combined_df_transposed.sum(axis=1), axis=0) * 100
    print("\nPercentage DataFrame:\n", percent_df)

    # Print the percentage of each stage for every model
    for model in percent_df.index:
        print(f"\nPercentages for {model}:")
        for stage, percentage in percent_df.loc[model].items():
            print(f"{stage}: {percentage:.2f}%")

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 3))
    combined_df_transposed.plot(kind='barh', stacked=True, ax=ax, colormap=cm.viridis)
    plt.xlabel('Duration (seconds)')
    # plt.title('Combined Event Durations for Each Model Group')
    # plt.xticks(rotation=45)
    # plt.yticks(rotation=45)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "motivation_latency_composition.pdf"), bbox_inches='tight', dpi=600)
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
    plt.savefig(os.path.join(save_directory, "misc_model_comparison.pdf"))
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
    plt.savefig(os.path.join(save_directory, "misc_sagemaker.pdf"))
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
    plt.savefig(os.path.join(save_directory, "evaluation_comparison_base.pdf"))
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
    plt.savefig(os.path.join(save_directory, "motivation_inference_ratio.pdf"))
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
    plt.savefig(os.path.join(save_directory, "motivation_resource_effect.pdf"))
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
    plt.savefig(os.path.join(save_directory, "evaluation_chosen_trace.pdf"))
    plt.show()
    
def draw_evaluation_trace_test():
    runtimes_trace = ['base', 'baseplus', 'opt', 'sagemaker']
    models = ['flan-t5-base']
    trace_labels = ['Sporadic', 'Bursty', 'Periodic']
    runtime_order = ['opt', 'base', 'baseplus', 'sagemaker']  # Desired order
    runtime_names = {
        'opt': 'FaLLServe',
        'base':'KServe',
        'sagemaker': 'Sagemaker',
        'baseplus': 'KServe+'
    }
    markers = {
        'opt': 'o',
        'base':'s',
        'baseplus': 'x',
        'sagemaker': 'v'
    }
    # Preparing data
    data = {}
    for model in models:
        for trace_label in trace_labels:
            for runtime in runtimes_trace:
                file_suffix = "" if runtime == 'sagemaker' else "-1m"
                file_path = f"../results/trace/{runtime}-{model}-{trace_label}{file_suffix}.csv"
                full_path = os.path.join(os.path.dirname(__file__), file_path)
                df = pd.read_csv(full_path)
                data[(model, trace_label, runtime)] = df
    
    def count_and_analyze_non_200_status_codes(data):
        analysis_results = {}
        for key, df in data.items():
            total_requests = df.shape[0]
            non_200_count = df[df['StatusCode'] != 200].shape[0]
            if total_requests > 0:  # Prevent division by zero
                non_200_percentage = (non_200_count / total_requests) * 100
            else:
                non_200_percentage = 0
            analysis_results[key] = (non_200_count, non_200_percentage, total_requests)
        return analysis_results
    def replace_non_200_with_max_latency(data):
        for key, df in data.items():
            if 'opt' in key:
                continue
            # Find the overall maximum latency
            max_overall_latency = df['E2ELatency'].max()
            print(f"Key {key}: Max Overall Latency = {max_overall_latency}")
            
            # Check if there are any non-200 entries first
            if not df[df['StatusCode'] != 200].empty:
                max_non_200_latency = df[df['StatusCode'] != 200]['E2ELatency'].max()
                print(f"Key {key}: Max Non-200 Latency = {max_non_200_latency}")

                # Replace non-200 latencies with the overall maximum found
                df.loc[df['StatusCode'] != 200, 'E2ELatency'] = max_overall_latency
    def print_percentile_latencies(data):
        percentiles = [50, 90, 99]  # Define the percentiles you want to calculate
        for key, df in data.items():
            # Calculate the specified percentiles for the 'E2ELatency' column
            p_values = np.percentile(df['E2ELatency'], percentiles)
            
            print(f"Key {key}:")
            # print(df['E2ELatency'].describe())
            for percentile, value in zip(percentiles, p_values):
                print(f"  P{percentile} Latency: {value:.2f} seconds")
    def plot_percentile_comparison(data):
        percentiles = [50, 90, 99]
        percentile_data = []

        trace_data = {}
        for key, df in data.items():
            model, trace_label, runtime = key
            # Calculate percentiles
            p_values = np.percentile(df['E2ELatency'], percentiles)
            # Organize data for plotting
            if trace_label not in trace_data:
                trace_data[trace_label] = []
            trace_data[trace_label].append({
                'Runtime': runtime,
                'P50': p_values[0],
                'P90': p_values[1],
                'P99': p_values[2]
            })

        # Plot each trace label's data
        for trace_label, entries in trace_data.items():
            runtime_labels = [entry['Runtime'] for entry in entries]
            latencies = [entry['P90'] for entry in entries]
            lower_errors = [entry['P90'] - entry['P50'] for entry in entries]
            upper_errors = [entry['P99'] - entry['P90'] for entry in entries]

            fig, ax = plt.subplots()
            ax.bar(runtime_labels, latencies, yerr=[lower_errors, upper_errors], capsize=5)
            ax.set_title(f'90th Percentile Latencies with Error Bars for {trace_label}')
            ax.set_xlabel('Runtime')
            ax.set_ylabel('Latency (seconds)')
            plt.show()
    def plot_aggregated_scatter(data, models, trace_labels, runtimes_trace, interval=10, aggregation_func=np.mean):
        # Set up a larger figure to hold all subplots
        fig, axes = plt.subplots(nrows=1, ncols=len(trace_labels), figsize=(18, 5), sharey=True)
        # Labels for each subplot as requested
        subplot_titles = ['(a) Sporadic', '(b) Periodic', '(c) Bursty']
        desired_order = ['FaLLServe', 'KServe', 'KServe+', 'Sagemaker']
        
        for index, trace_label in enumerate(trace_labels):
            for model in models:
                # plt.figure(figsize=(8, 4))
                
                for runtime in runtimes_trace:
                    df = data[(model, trace_label, runtime)]
                    
                    # Ensure 'Timestamp' is in a datetime format if not already
                    df['Interval'] = (df['Timestamp'] // interval) * interval
                    
                    # Resample and aggregate
                    df_aggregated = df.groupby('Interval')['E2ELatency'].agg(aggregation_func).reset_index()

                    axes[index].scatter(df_aggregated['Interval'], df_aggregated['E2ELatency'], alpha=0.5, s=20, label=runtime_names[runtime], marker=markers[runtime])
                
                # plt.title(f'{trace_label} E2E Latency Over Time (Aggregated by {aggregation_func.__name__.title()})')
                axes[index].set_title(subplot_titles[index], y=-0.2)
                axes[index].set_xlabel('Time (minutes)')
                if index == 0:
                    axes[index].set_ylabel('E2E Latency (seconds)')
        # Adjust layout
        plt.tight_layout()
        plt.legend()
        # Place a common legend above the figure
        handles, labels = axes[0].get_legend_handles_labels()
        legend_order = {label: handle for handle, label in zip(handles, labels)}
        ordered_handles = [legend_order[label] for label in desired_order if label in legend_order]

        fig.legend(ordered_handles, desired_order, loc='upper center', ncol=len(desired_order), bbox_to_anchor=(0.5, 1.05), fontsize='large', frameon=False)
        # Hide individual legends from each subplot
        for ax in axes:
            ax.legend().set_visible(False)  # This ensures no subplot-specific legends are shown

        plt.savefig(os.path.join(save_directory, f"evaluation_trace_scatter_summary.pdf"), bbox_inches='tight', dpi=600)
        plt.show()
    
    # Analyze data
    analysis_results = count_and_analyze_non_200_status_codes(data)
    for key, (count, percentage, total) in analysis_results.items():
        print(f"{key}: {count} requests returned a non-200 status code, which is {percentage:.2f}% of {total} requests.")
    replace_non_200_with_max_latency(data)
    print_percentile_latencies(data)
    # plot_percentile_comparison(data)
    plot_aggregated_scatter(data, models, trace_labels, runtimes_trace, 10, lambda x: np.percentile(x, 90))
    exit(0)
    
    # Violin plot
    for trace_label in trace_labels:
        # Prepare data for the plot
        box_data = []
        for runtime in runtimes_trace:
            df = data[(model, trace_label, runtime)]
            df['Runtime'] = runtime  # Add a 'Runtime' column to distinguish data in the plot
            # Rename and reorder data according to the new setup
            df['Runtime'] = df['Runtime'].map(runtime_names)
            box_data.append(df)
        combined_df = pd.concat(box_data)
        # Ensure that the DataFrame uses the new runtime order
        category_order = [runtime_names[runtime] for runtime in runtime_order]
        combined_df['Runtime'] = pd.Categorical(combined_df['Runtime'], categories=category_order, ordered=True)
        
        if trace_label == 'Sporadic':
            # Generate the violin plot for Sporadic workflow
            plt.figure(figsize=(5, 3))
            sns.violinplot(x='Runtime', y='E2ELatency', data=combined_df, order=category_order, inner=None)
            plt.xlabel('')
            plt.ylabel('E2E Latency (seconds)')
            plt.savefig(os.path.join(save_directory, f"evaluation_trace_{trace_label}.pdf"), bbox_inches='tight', dpi=600)
            plt.show()
        else:
            # Calculate percentiles for bar chart with error bars
            p50 = combined_df.groupby('Runtime')['E2ELatency'].quantile(0.50)
            p90 = combined_df.groupby('Runtime')['E2ELatency'].quantile(0.90)
            p99 = combined_df.groupby('Runtime')['E2ELatency'].quantile(0.99)
            
            # Calculate errors
            error_lower = p90 - p50
            error_upper = p99 - p90
            
            plt.figure(figsize=(5, 3))
            plt.bar(p90.index, p90, yerr=[error_lower.values, error_upper.values], capsize=5, color='skyblue')
            # Set y-axis to logarithmic scale
            plt.yscale('log')
            plt.xlabel('')
            plt.ylabel('E2E Latency (seconds)')
            # plt.title(f'{trace_label} Latency with Error Bars (P90 with P50, P99)')
            plt.savefig(os.path.join(save_directory, f"evaluation_trace_{trace_label}.pdf"), bbox_inches='tight', dpi=600)
            plt.show()
    
    # # Recommended Plots
    # for trace_label in trace_labels:
    #     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharey=True)
    #     fig.suptitle(f'Comparison of Runtimes for {model} with Trace: {trace_label}')

    #     # Prepare data for plots
    #     box_data = []
    #     for runtime in runtimes_trace:
    #         df = data[(model, trace_label, runtime)]
    #         df['Runtime'] = runtime
    #         box_data.append(df)
    #     combined_df = pd.concat(box_data)
        
    #     # Box Plot
    #     axes[0].set_title('Box Plot of E2E Latency')
    #     sns.boxplot(ax=axes[0], x='Runtime', y='E2ELatency', data=combined_df)
    #     axes[0].set_xlabel('Runtime')
    #     axes[0].set_ylabel('E2E Latency (seconds)')

    #     # Violin Plot
    #     axes[1].set_title('Violin Plot of E2E Latency')
    #     sns.violinplot(ax=axes[1], x='Runtime', y='E2ELatency', data=combined_df)
    #     axes[1].set_xlabel('Runtime')

    #     # # Line Plot
    #     # axes[2].set_title('E2E Latency Over Time')
    #     # for runtime in runtimes_trace:
    #     #     df = data[(model, trace_label, runtime)]
    #     #     sns.lineplot(ax=axes[2], x=df['Timestamp'].to_numpy(), y=df['E2ELatency'].to_numpy(), data=df, label=runtime)
    #     # axes[2].set_xlabel('Timestamp (minute)')
    #     # axes[2].set_ylabel('E2E Latency (seconds)')
        
    #     # axes[2].legend(title='Runtime')

    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     plt.show()
        
    # # Scatter graph
    # for model in models:
    #     for trace_label in trace_labels:
    #         plt.figure(figsize=(10, 5))
    #         for runtime in runtimes_trace:
    #             df = data[(model, trace_label, runtime)]
    #             plt.scatter(df['Timestamp'].to_numpy(), df['E2ELatency'].to_numpy(), alpha=0.5, s=20, label=f"{runtime}-{trace_label}")
    #         plt.title(f'{trace_label} E2E Latency Over Time')
    #         plt.xlabel('Timestamp (minute)')
    #         plt.ylabel('E2E Latency (seconds)')
    #         plt.legend()
    #         plt.grid(True)
    #         plt.show()
    
    # # CDF
    # for model in models:
    #     for trace_label in trace_labels:
    #         plt.figure(figsize=(10, 5))
    #         for runtime in runtimes_trace:
    #             df = data[(model, trace_label, runtime)]
    #             # Sorting data for CDF
    #             sorted_latencies = np.sort(df['E2ELatency'])
    #             yvals = np.arange(len(sorted_latencies))/float(len(sorted_latencies)-1)
    #             plt.plot(sorted_latencies, yvals, label=runtime)
    #         plt.title(f'{trace_label} CDF of E2E Latencies')
    #         plt.xlabel('E2E Latency (seconds)')
    #         plt.ylabel('CDF')
    #         plt.legend()
    #         plt.grid(True)
    #         plt.show()
    
    # # Hist graph
    # plt.figure(figsize=(10, 5))
    # for model in models:
    #     for trace_label in trace_labels:
    #         for runtime in runtimes_trace:
    #             if runtime == 'sagemaker':
    #                 df = pd.read_csv(os.path.join(os.path.dirname(__file__), f"../results/trace/{runtime}-{model}-{trace_label}.csv"))
    #             else:
    #                 df = pd.read_csv(os.path.join(os.path.dirname(__file__), f"../results/trace/{runtime}-{model}-{trace_label}-1m.csv"))
    #             plt.hist(df['E2ELatency'], label=f"{runtime}-{trace_label}")
    # plt.title('E2E Latency Over Time')
    # plt.xlabel('E2E Latency (seconds)')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
def draw_evaluation_performance_breakdown():
    # Sample data
    data = {
        'Version': ['V1', 'V2', 'V3', 'V4'],
        'Execution Time': [240, 220, 180, 150],  # Execution time decreases
        'Memory Usage': [1200, 1100, 1000, 900]  # Memory usage decreases
    }

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()

    # Bar plot for execution time
    color = 'tab:blue'
    ax1.set_xlabel('Version')
    ax1.set_ylabel('Execution Time (seconds)', color=color)
    ax1.bar(data['Version'], data['Execution Time'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a twin Axes sharing the x-axis for memory usage
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Memory Usage (MB)', color=color)
    ax2.plot(data['Version'], data['Memory Usage'], color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and show the plot
    plt.title('Optimization Effects Over Versions')
    fig.tight_layout()  # Adjust layout to make room
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
    # draw_evaluation_performance_breakdown()