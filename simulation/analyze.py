import os
import numpy as np

def read_data(basedir):
    dir_name=[]
    for item in med_trace_list:
        dir_name.append("/med_"+str(item))
    for item in tail_trace_list:
        dir_name.append("/tail_"+str(item))
                       
    keepalive = []
    time = []
                        
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
        
    return(keepalive, time)

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
    

if __name__ == "__main__":
    med_trace_list = [249] 
    tail_trace_list = [9]
    combined_list = med_trace_list + tail_trace_list
    
    technique_list = ["keepalive", "oracle"]
    keepalive_list = []
    time_list = []
    
    for tech in technique_list:        
        keepalive, time = read_data(os.path.join(os.path.dirname(__file__), f"../results/simulation/prewarm/{tech}"))
        keepalive_list.append(keepalive)
        time_list.append(time)
        
    for i, trace in enumerate(combined_list):
        print(f"Trace: {combined_list[i]}, idx {i}")
        for j, tech in enumerate(technique_list):
            keepalive = keepalive_list[j]
            time = time_list[j]

            trace_keepalive = keepalive[i]
            trace_time = time[i]
            print(f"Technique: {tech}")
            
            print(f"STAT: Keepalive cost")
            stats(trace_keepalive)
            print(f"STAT: Service time")
            stats(trace_time)