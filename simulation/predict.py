import os
import shutil

def write_results(basedir):
    if os.path.exists(basedir):
        shutil.rmtree(basedir)
    os.makedirs(basedir)
    
    dir_name=[]
    for item in med_trace_list:
        dir_name.append("/med_"+str(item))
    for item in tail_trace_list:
        dir_name.append("/tail_"+str(item))
    
    for i in range(len(dir_name)):
        os.makedirs(basedir+dir_name[i])
        
        with open(basedir+dir_name[i]+'/true_list.txt', 'w') as f:
            for item in real_list[i]:
                f.write("%s\n" % item)
        
        with open(basedir+dir_name[i]+'/predicted_list.txt', 'w') as f:
            for item in predicted_list[i]:
                f.write("%s\n" % item)
    
        with open(basedir+dir_name[i]+'/keepalive_cost_list.txt', 'w') as f:
            for item in keepalive_cost_list[i]:
                f.write("%s\n" % item)
    
        with open(basedir+dir_name[i]+'/running_cost_list.txt', 'w') as f:
            for item in running_cost_list[i]:
                f.write("%s\n" % item)
            
        with open(basedir+dir_name[i]+'/time_list.txt', 'w') as f:
            for item in time_list[i]:
                f.write("%s\n" % item)
            
        # with open(basedir+dir_name[i]+'/gb_sec_list.txt', 'w') as f:
        #     for item in gb_sec_list[i]:
        #         f.write("%s\n" % item)
                
                
        # with open(basedir+dir_name[i]+'/selected_system_list.txt', 'w') as f:
        #     for item in selected_system_list[i]:
        #         f.write("%s\n" % item)

        # with open(basedir+dir_name[i]+'/weighted_value_list.txt', 'w') as f:
        #     for item in weighted_value_list[i]:
        #         f.write("%s\n" % item)

    
    # with open(basedir+'/total_memory_used_list.txt', 'w') as f:
    #     for item in total_memory_used_list:
    #         f.write("%s\n" % item)

def controller_timeoracle():
    for j in range(local_window, len(trace_list[0])):
        for i in range(len(trace_list)):
            real_value=trace_list[i][j]
            real_list[i].append(real_value)
            predicted_list[i].append(real_value)
    
    selected_system = 2
    
def controller_keepalive():
    time_period=10
    
    for i in range(len(trace_list)):
        predicted_list[i].append(0)
    
    for j in range(local_window, len(trace_list[0])):
        for i in range(len(trace_list)):
            real_value=trace_list[i][j]
            if len(predicted_list[i])-1==len(real_list[i]):
                for s in range(time_period):            
                    predicted_list[i].append(real_value)
            real_list[i].append(real_value)
        
    for i in range(len(trace_list)):
        predicted_list[i].pop()
    
    selected_system = 1

def run():
    keepalive_cost_list=[[] for i in range(len(trace_list))]
    running_cost_list=[[] for i in range(len(trace_list))]
    time_list=[[] for i in range(len(trace_list))]
    for j in range(len(predicted_list[0])):  
        ##main running
        for i in range(len(predicted_list)):
            
            if selected_system!=0:
                exe_time=exe_time_list[i]
                cs_time=cs_time_gpu[i]
                pr=predicted_list[i][j]
                re=real_list[i][j]
                
                keepalive_cost_list[i].append(pr*cost_per_min_gpu)
                running_cost_list[i].append(re*cost_per_min_gpu*exe_time)
                
                ##time part
                if re>pr:
                    time_val_list=[exe_time for s in range(int(pr))]
                    for s in time_val_list:
                        time_list[i].append(s)
                    time_val_list=[exe_time+cs_time for s in range(int(re-pr))]
                    for s in time_val_list:
                        time_list[i].append(s)
                else:
                    time_val_list=[exe_time for s in range(int(re))]
                    for s in time_val_list:
                        time_list[i].append(s)

            # if selected_system==0:
            #     exe_time=exe_time_costly[i]
            #     cs_time=cs_time_costly[i]
            #     pr=predicted_list[i][j]
            #     re=real_list[i][j]
            #     mem=mem_list[i]
            #     cost=cost_per_sec_per_mb[0]
                
            #     keepalive_cost_list[i].append(0)
            #     running_cost_list[i].append(re*mem*cost*exe_time)
            #     gb_sec_list[i].append(re*mem*exe_time)
                
            #     ##time part
            #     time_val_list=[exe_time+cs_time for s in range(int(re))]
            #     for s in time_val_list:
            #         time_list[i].append(s)
        print(j)

    return (keepalive_cost_list, running_cost_list, time_list)        

if __name__ == "__main__":
    med_trace_list = [249] 
    tail_trace_list = [9]
    # med_trace_list = [249, 757, 1385, 1489, 1717, 1721] 
    # tail_trace_list = [9,15,18,19,21,22]
    
    exe_time_list = [0.2] * 12
    cs_time_gpu = [1] * 12
    cs_time_gpu_kserve = [3] * 12
    cs_time_download = [1] * 12
    
    cost_per_min_cpu = 0.0416 / 60
    cost_per_min_gpu = 0.526 / 60

    filename_list=[]

    for item in med_trace_list:
        filename_list.append(os.path.join(os.path.dirname(__file__), "./main_traces/med_"+str(item)+".txt"))
    for item in tail_trace_list:
        filename_list.append(os.path.join(os.path.dirname(__file__), "./main_traces/tail_"+str(item)+".txt"))

    trace_list=[]
    for file in filename_list:
        with open(file) as f1:
            trace=f1.read().splitlines()
            trace=[float(i) for i in trace]
        trace_list.append(trace)            

    trace_list=[[i*7 for i in trace] for trace in trace_list]##
    
    local_window=60#
    prediction_history_window=local_window#
    
    # # Minor test
    # truncate = 20
    # trace_list=[trace[:truncate + local_window] for trace in trace_list]
    # print(trace_list)
    
    # Keep alive 10 minutes
    real_list=[[] for i in range(len(trace_list))]
    predicted_list=[[] for i in range(len(trace_list))]
    # selected_system_list=[[] for i in range(len(trace_list))]
    selected_system = 1

    controller_keepalive()
    
    print(real_list)
    print(predicted_list)
    
    keepalive_cost_list, running_cost_list, time_list = run()
    print(keepalive_cost_list)
    print(running_cost_list)
    print(time_list)
    
    write_results(os.path.join(os.path.dirname(__file__), "../results/simulation/prewarm/keepalive"))
    
    # Oracle
    real_list=[[] for i in range(len(trace_list))]
    predicted_list=[[] for i in range(len(trace_list))]
    selected_system_list=[[] for i in range(len(trace_list))]

    controller_timeoracle()
    
    print(real_list)
    print(predicted_list)
    
    keepalive_cost_list, running_cost_list, time_list = run()
    print(keepalive_cost_list)
    print(running_cost_list)
    print(time_list)
    
    write_results(os.path.join(os.path.dirname(__file__), "../results/simulation/prewarm/oracle"))
    