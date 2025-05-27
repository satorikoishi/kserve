import os
import shutil
import numpy as np
from numpy import fft
import optuna
from optuna.samplers import NSGAIISampler
import optuna.visualization as vis
import pandas as pd

def compute_avg_cost_and_latency(keepalive_cost_list, time_list):
    avg_cost = sum(x for trace in keepalive_cost_list for x in trace) / \
            sum(len(trace) for trace in keepalive_cost_list)
    avg_latency = sum(x for trace in time_list for x in trace) / \
                sum(len(trace) for trace in time_list)
    return avg_cost, avg_latency

def compute_overall_cost(keepalive_cost_list, time_list):
    cost_per_min_gpu = 0.526 / 60
    total_cost = sum(x for trace in keepalive_cost_list for x in trace)
    total_service_time = sum(x for trace in time_list for x in trace)
    return total_cost + total_service_time * cost_per_min_gpu

# def objective(trial): 
#     global real_list, predicted_list, selected_system
    
#     # Suggest values for alpha and beta
#     alpha = trial.suggest_float("alpha", 1.5, 3.0, step=0.1)
#     beta = trial.suggest_float("beta", 0.0, 1.5, step=0.1)
    
#     # Reset global state
#     real_list = [[] for _ in range(len(trace_list))]
#     predicted_list = [[] for _ in range(len(trace_list))]
    
#     controller_fft_biasplus(alpha, beta)
#     keepalive_cost_list, running_cost_list, time_list = run()
#     cost, latency = compute_avg_cost_and_latency(keepalive_cost_list, time_list)

#     # Return both objectives: cost and latency to minimize
#     return cost, latency

def objective(trial): 
    global real_list, predicted_list, selected_system
    
    # Suggest values for alpha and beta
    alpha = trial.suggest_float("alpha", 1.5, 3.0, step=0.1)
    beta = trial.suggest_float("beta", 0.0, 1.5, step=0.1)
    
    # Reset global state
    real_list = [[] for _ in range(len(trace_list))]
    predicted_list = [[] for _ in range(len(trace_list))]
    
    controller_fft_biasplus(alpha, beta)
    keepalive_cost_list, running_cost_list, time_list = run()
    cost = compute_overall_cost(keepalive_cost_list, time_list)

    # Return both objectives: cost and latency to minimize
    return cost

def fourierExtrapolation(x, n_predict):
    n = x.size
    n_harm = harmonics              # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))
    
    selected = indexes[:1 + n_harm * 2]
    
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    # print(f"p0t {p[0] * t}")
    # print(f"index {indexes}")
    
    for loop_idx, i in enumerate(selected):
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        term = ampli * np.cos(2 * np.pi * f[i] * t + phase)
        restored_sig += term
        # print(f"term {term}")
        # print(f"rs {restored_sig}")
        
        # if loop_idx == 0:
        #     first_rs = restored_sig.copy()
        #     print(f"first_rs {first_rs}")
        # if loop_idx == len(selected) - 1:
        #     final_rs = restored_sig.copy()
        #     print(f"final_rs {final_rs}")
        #     print(f"rs_diff {final_rs - first_rs}")
            
    return restored_sig + p[0] * t

# def fourierExtrapolationWithFactor(x, n_predict, factor=2.0):
#     n = x.size
#     n_harm = harmonics              # number of harmonics in model
#     t = np.arange(0, n)
#     p = np.polyfit(t, x, 1)         # find linear trend in x
#     x_notrend = x - p[0] * t        # detrended x
#     x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
#     f = fft.fftfreq(n)              # frequencies
#     indexes = list(range(n))
#     # sort indexes by frequency, lower -> higher
#     indexes.sort(key = lambda i: np.absolute(f[i]))
 
#     t = np.arange(0, n + n_predict)
#     restored_sig = np.zeros(t.size)
#     for i in indexes[:1 + n_harm * 2]:
#         if i not in (0, n//2):
#             ampli = factor * np.abs(x_freqdom[i]) / n    # amplitude with factor
#         else:
#             ampli = np.abs(x_freqdom[i]) / n    # amplitude
#         phase = np.angle(x_freqdom[i])          # phase
#         restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
#     return restored_sig + p[0] * t

def fourierExtrapolationBias(x, n_predict, alpha=2.0, beta=1.0):
    n = x.size
    n_harm = harmonics              # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))
    selected = indexes[:1 + n_harm * 2]
    
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    
    # print(f"p0t {p[0] * t}")
    # print(f"selected {selected}")
    
    for loop_idx, i in enumerate(selected):
        # print(f"n {n}, i {i}, loop_idx {loop_idx}")
        if i not in (0, n//2):
            ampli = alpha * np.abs(x_freqdom[i]) / n    # amplitude with factor
        else:
            ampli = np.abs(x_freqdom[i]) / n    # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        term = ampli * np.cos(2 * np.pi * f[i] * t + phase)
        if i in (0, n//2):
            bias = 1.0
        else:
            bias = beta
        bias_term = np.where(term >= 0,
                    term,
                    bias  * term)
        restored_sig += bias_term
        # print(f"term {term}")
        # print(f"bias {bias}, bias term {bias_term}")
        # print(f"rs {restored_sig}")
        
        # if loop_idx == 0:
        #     first_rs = restored_sig.copy()
        #     print(f"first_rs {first_rs}")
        # if loop_idx == len(selected) - 1:
        #     final_rs = restored_sig.copy()
        #     print(f"final_rs {final_rs}")
        #     print(f"rs_diff {final_rs - first_rs}")
        
    return restored_sig + p[0] * t

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
        while len(predicted_list[i]) > len(real_list[i]):
            predicted_list[i].pop()
    
    selected_system = 1

def controller_fft():
    for j in range(local_window, len(trace_list[0])):
        for i in range(len(trace_list)):
            training_trace=np.array(trace_list[i][j-local_window:j])
            n_predict = 1
            extrapolation = fourierExtrapolation(training_trace, n_predict)
            pred_value=extrapolation[len(extrapolation)-1]
            # print(f"Extrap: {extrapolation}")
            # print(f"Prediction: {pred_value}")
            if pred_value <0:
                pred_value=0
            else:
                pred_value=round(pred_value)
            
            real_value=trace_list[i][j]
            real_list[i].append(real_value)
            predicted_list[i].append(pred_value)
            
# def controller_fft_factor():
#     for j in range(local_window, len(trace_list[0])):
#         for i in range(len(trace_list)):
#             training_trace=np.array(trace_list[i][j-local_window:j])
#             n_predict = 1
#             extrapolation = fourierExtrapolationWithFactor(training_trace, n_predict)
#             pred_value=extrapolation[len(extrapolation)-1]
#             if pred_value <0:
#                 pred_value=0
#             else:
#                 pred_value=round(pred_value)
            
#             real_value=trace_list[i][j]
#             real_list[i].append(real_value)
#             predicted_list[i].append(pred_value)

def controller_fft_bias():
    for j in range(local_window, len(trace_list[0])):
        for i in range(len(trace_list)):
            training_trace=np.array(trace_list[i][j-local_window:j])
            n_predict = 1
            extrapolation = fourierExtrapolationBias(training_trace, n_predict)
            pred_value=extrapolation[len(extrapolation)-1]
            # print(f"Extrap: {extrapolation}")
            # print(f"Prediction: {pred_value}")
            if pred_value <0:
                pred_value=0
            else:
                pred_value=round(pred_value)
            
            real_value=trace_list[i][j]
            real_list[i].append(real_value)
            predicted_list[i].append(pred_value)

def controller_fft_biasplus(alpha=2.0, beta=0.0):
    for j in range(local_window, len(trace_list[0])):
        for i in range(len(trace_list)):
            training_trace=np.array(trace_list[i][j-local_window:j])
            n_predict = 1
            extrapolation = fourierExtrapolationBias(training_trace, n_predict, alpha, beta)
            pred_value=extrapolation[len(extrapolation)-1]
            # print(f"Extrap: {extrapolation}")
            # print(f"Prediction: {pred_value}")
            if pred_value <0:
                pred_value=0
            else:
                pred_value=round(pred_value)
            
            real_value=trace_list[i][j]
            real_list[i].append(real_value)
            predicted_list[i].append(pred_value)

def run():
    keepalive_cost_list=[[] for i in range(len(trace_list))]
    running_cost_list=[[] for i in range(len(trace_list))]
    time_list=[[] for i in range(len(trace_list))]
    time_val_counter = 0
    for j in range(len(predicted_list[0])):  
        ##main running
        for i in range(len(predicted_list)):
            exe_time=exe_time_list[i]
            cs_time=cs_time_gpu[i]
            cs_download_time = cs_time_download[i]
            pr=predicted_list[i][j]
            re=real_list[i][j]
            
            keepalive_cost_list[i].append(pr*cost_per_min_gpu)
            running_cost_list[i].append(re*cost_per_min_gpu*exe_time)
            
            ##time part
            if re>pr:
                time_val_list=[exe_time for s in range(int(pr))]
                for s in time_val_list:
                    time_list[i].append(s)
                if selected_system == 1:
                    time_val_list=[exe_time+cs_time for s in range(int(re-pr))]
                elif selected_system == 2:
                    if time_val_counter % 10 == 0:
                        time_val_list=[exe_time+cs_time+cs_download_time for s in range(int(re-pr))]
                    else:
                        time_val_list = [exe_time+cs_time for s in range(int(re-pr))]
                    time_val_counter += 1  # Update the counter
                else:
                    raise ValueError(f"Unexpected selected system {selected_system}")
                for s in time_val_list:
                    time_list[i].append(s)
            else:
                time_val_list=[exe_time for s in range(int(re))]
                for s in time_val_list:
                    time_list[i].append(s)

            # print(f"{j} {i} {re} {pr} {time_val_list}")
                
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
        # print(j)

    return (keepalive_cost_list, running_cost_list, time_list)        

if __name__ == "__main__":
    # med_trace_list = [249] 
    # tail_trace_list = [9]
    med_trace_list = [249, 1385, 1489, 1717, 1721] 
    tail_trace_list = [9,15,18,19,21,22]
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
    
    harmonics=10#
    local_window=60#
    prediction_history_window=local_window#
    
    # # Minor test
    # truncate = 2
    # trace_list=[trace[:truncate + local_window] for trace in trace_list]
    # print(trace_list)
    
    # ##################### Study ##########################
    
    # real_list=[[] for i in range(len(trace_list))]
    # predicted_list=[[] for i in range(len(trace_list))]
    # selected_system = 1
    
    # # Create multi-objective study
    # # study = optuna.create_study(
    # #     directions=["minimize", "minimize"],
    # #     sampler=NSGAIISampler(seed=42)
    # # )
    # study = optuna.create_study(
    #     direction="minimize",
    #     sampler=NSGAIISampler(seed=42)
    # )

    # # Run tuning
    # study.optimize(objective, n_trials=1)  # Increase n_trials for better frontier

    # # Show best Pareto-optimal configs
    # print("Best Pareto front trials:")
    # for trial in study.best_trials:
    #     print(f"Alpha: {trial.params['alpha']}, Beta: {trial.params['beta']}, Cost: {trial.value:.2f}")
    # pareto_data = [{
    #     "alpha": t.params["alpha"],
    #     "beta": t.params["beta"],
    #     "cost": t.value,
    # } for t in study.best_trials]

    # df = pd.DataFrame(pareto_data)
    # df.to_csv(os.path.join(os.path.dirname(__file__), "pareto_front.csv"), index=False)

    # # Visualization
    # fig_pareto = vis.plot_pareto_front(study)
    # fig_pareto.write_html(os.path.join(os.path.dirname(__file__), "pareto_front.html"))
    # fig_pareto.show()
    # vis.plot_contour(study).show()
    # exit(0)
    
    # ##################### Study End ##########################
    
    # Keep alive 10 minutes
    real_list=[[] for i in range(len(trace_list))]
    predicted_list=[[] for i in range(len(trace_list))]
    selected_system = 2

    controller_keepalive()
    
    print(real_list)
    print(predicted_list)
    
    keepalive_cost_list, running_cost_list, time_list = run()
    print(keepalive_cost_list)
    print(running_cost_list)
    print(time_list)
    
    write_results(os.path.join(os.path.dirname(__file__), "../results/simulation/prewarm/keepalive"))
    
    # fft
    real_list=[[] for i in range(len(trace_list))]
    predicted_list=[[] for i in range(len(trace_list))]
    controller_fft()
    
    print(real_list)
    print(predicted_list)
    
    keepalive_cost_list, running_cost_list, time_list = run()
    print(keepalive_cost_list)
    print(running_cost_list)
    print(time_list)
    
    write_results(os.path.join(os.path.dirname(__file__), "../results/simulation/prewarm/fft"))
    
    # # Single factor
    # real_list=[[] for i in range(len(trace_list))]
    # predicted_list=[[] for i in range(len(trace_list))]
    # selected_system_list=[[] for i in range(len(trace_list))]

    # controller_fft_factor()
    
    # print(real_list)
    # print(predicted_list)
    
    # keepalive_cost_list, running_cost_list, time_list = run()
    # print(keepalive_cost_list)
    # print(running_cost_list)
    # print(time_list)
    
    # write_results(os.path.join(os.path.dirname(__file__), "../results/simulation/prewarm/fft_factor"))
    
    # Bias 2, 1
    real_list=[[] for i in range(len(trace_list))]
    predicted_list=[[] for i in range(len(trace_list))]

    controller_fft_bias()
    
    print(real_list)
    print(predicted_list)
    
    keepalive_cost_list, running_cost_list, time_list = run()
    print(keepalive_cost_list)
    print(running_cost_list)
    print(time_list)
    
    write_results(os.path.join(os.path.dirname(__file__), "../results/simulation/prewarm/fft_bias"))
    
    # Bias 2, 0
    real_list=[[] for i in range(len(trace_list))]
    predicted_list=[[] for i in range(len(trace_list))]

    controller_fft_biasplus()
    
    print(real_list)
    print(predicted_list)
    
    keepalive_cost_list, running_cost_list, time_list = run()
    print(keepalive_cost_list)
    print(running_cost_list)
    print(time_list)
    
    write_results(os.path.join(os.path.dirname(__file__), "../results/simulation/prewarm/fft_biasplus"))
    
    # Oracle
    real_list=[[] for i in range(len(trace_list))]
    predicted_list=[[] for i in range(len(trace_list))]
    selected_system = 1

    controller_timeoracle()
    
    print(real_list)
    print(predicted_list)
    
    keepalive_cost_list, running_cost_list, time_list = run()
    print(keepalive_cost_list)
    print(running_cost_list)
    print(time_list)
    
    write_results(os.path.join(os.path.dirname(__file__), "../results/simulation/prewarm/oracle"))
    
    # Bias 2, 0, multi-level
    real_list=[[] for i in range(len(trace_list))]
    predicted_list=[[] for i in range(len(trace_list))]

    controller_fft_biasplus()
    
    print(real_list)
    print(predicted_list)
    
    keepalive_cost_list, running_cost_list, time_list = run()
    print(keepalive_cost_list)
    print(running_cost_list)
    print(time_list)
    
    write_results(os.path.join(os.path.dirname(__file__), "../results/simulation/prewarm/fallserve"))
    