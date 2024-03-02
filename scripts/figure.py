import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import os

# model_name_list = ["bloom-560m", "bert-large-uncased"]
model_name_list = ["bloom-560m", 
                   "flan-t5-small", "flan-t5-base", "flan-t5-large", 
                   "bert-base-uncased", "bert-large-uncased"]
# runtime_config = ["base"]
runtime_config = ["base", "opt"]

def fetch_data_from_file():
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

def draw_comparison_base():
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
    # plt.savefig("model_comparison_chart.png")
    plt.show()

if __name__ == "__main__":
    draw_comparison_base()