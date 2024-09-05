import pandas as pd
import phaselocking_utils
import numpy as np
import bisect

animal = input('Animal (JC315, JC283, or JC274): ')

#Initialize the output csv file
output_file = animal + '_phaselocking_correct.csv'
pd.DataFrame(columns = ['Animal', 'Day', 'Session', 'Tet Par', '% time kept by speed filter', '% time kept by peak filter', '% of time kept by correct filter',
                        '% time kept by all filters',
                        'Num ca1 clusters', 'ca1 firing cells', 'ca1 selective cells', '% Ca1 selective', 
                        'Num pfc clusters', 'pfc firing cells', 'pfc selective cells', '% pfc selective']
                        ).to_csv(output_file, index = False)

#Import parameters
params = pd.read_csv('phaselocking_parameters.csv', header = [0, 1])
days = params[(animal, 'days')].tolist()
days = [str(int(i)) for i in days if not pd.isna(i)]
num_trials_1 = params[(animal, 'num trials 1')].tolist()
num_trials_1 = [int(i) for i in num_trials_1 if not pd.isna(i)]
tet_par = params[(animal, 'tet par')].tolist()
tet_par = [int(i) for i in tet_par if not pd.isna(i)]
tet_total = params[(animal, 'num tetrodes')].tolist()
tet_total = [int(i) for i in tet_total if not pd.isna(i)]

#I could also split this by individual correct arms:
    # rewarded_arms_dict = {'JC315': [4, 7], 'JC283': [5, 8], 'JC274': [4, 8]} #In terms of arm_id_andrea in the Correct_behavior_vlad csvs


for day_index in range(len(days)):
    day = days[day_index]
    tetrode = tet_par[day_index]
    num_tetrodes = tet_total[day_index]
    for session in ['training1', 'training2']:
        results = [animal, day, session, tetrode] #Initialize the row to be added to the output csv
        filepath = animal + '-data/' + animal + '-' + day + '/' + animal + '-' + day + '_' + session #Where the .whl, .clu, etc. are located
        speed_filepath = 'one_drive_data/eight_arm_fig_data_andrea/analysis/' + animal + '-' + day + '/' + animal + '_' + day + '_' + session + '.speed_vladVersion'
            #The speed files, stored in the one drive 
                #Note that speed is set to 0 outside of trials, so this file does not work when the animal is running back in the outer ring
        inst_phase, filtered_eegh = phaselocking_utils.get_phase(filepath, tetrode, num_tetrodes)
        speed_filter = phaselocking_utils.filter_for_speed(speed_filepath)
        results.append(sum(speed_filter) / len(speed_filter)) #% time kept by speed filter
        peak_filter = phaselocking_utils.peaks_filter(filtered_eegh) 
        results.append((sum(peak_filter) / len(peak_filter))) #% time kept by peak filter
        res = phaselocking_utils.to_int_list(filepath + '.res')

        #To deal with the exeption that happens for training2 sessions in JC274 where the res file has spikes occuring after the end of the eegh recording
        if res[-1] // 4 > len(filtered_eegh) - 1: #-1 because it's in indices
            print(f'res is longer than eegh: {res[-1] // 4} is longer than {len(filtered_eegh)} in eegh frames (5 kHz).')
            cutoff_index = bisect.bisect_right(res, 4 * (len(filtered_eegh) - 1) + 3) #adding 3 as I use floor division
            print(f'initial max res: {res[-1]}')
            res = res[:cutoff_index]
            print(f'final max res: {res[-1]}')
        res_frames = np.arange(res[-1])

        #Get correct trials
        correct_trials_csv = pd.read_csv('eight_arm_new/Andrea_project/Corrected_behavior_vlad/' + animal + '-' + day + '_' + session + '_data.csv')
            #These .csv files of correct trials corresponding to the .trials files are in the github repository
        correct_trials = correct_trials_csv['CorrectBool'].tolist()
        correct_trials = [bool(val) for val in correct_trials]
        trials_filepath = 'one_drive_data/eight_arm_fig_data_andrea/analysis/' + animal + '-' + day + '/' + animal + '_' + day + '_' + session + '.trials_vladVersion'
            #These files are in the one drive folder

        #Make a filter to keep spikes that happen during correct trials
        with open(trials_filepath, 'r') as file:
            lines = file.readlines()
            trial_timestamps = [line.split() for line in lines]
        timestamps_to_keep = [x for i, x in enumerate(trial_timestamps) if correct_trials[i]]
        correct_filter = [False] * (res[-1] // 512) #the length of the correct filter needs to be res[-1] // 512 in length
        for start, end in timestamps_to_keep:
            for i in range(int(start), int(end) + 1):
                correct_filter[i] = True
        results.append((sum(correct_filter) / len(correct_filter))) #Percent of time kept by the correct filter

        time_filtered = [x for x in res_frames if peak_filter[x // 4] and speed_filter[x // 512] and correct_filter[x // 512]] 
        results.append((len(time_filtered)/ len(res_frames))) #% time kept by all filters  

        clu = phaselocking_utils.to_int_list(filepath + '.clu')
        _, ca1_clusters, pfc_right_clusters = phaselocking_utils.describe_clusters(animal + '-data/' + animal + '-' + day + '/' + animal + '-' + day)
            #pfc_clusters, ca1_clusters, pfc_right_clusters

        cell_types_dict = {
            'CA1': ca1_clusters,
            'PFC': pfc_right_clusters
        }

        for cell_type in ['CA1', 'PFC']:
            a_bars = [] #preferred phase
            r_vals = [] #MVL
            p_vals = []
            selective_cells = []
            firing_cell_count = 0
            results.append(len(cell_types_dict[cell_type])) #Total count of clusters in either CA1 or PFC
            for cell in cell_types_dict[cell_type]:
                res_for_one_cell = [x for i, x in enumerate(res) if clu[i] == cell]
                res_filtered = [x for x in res_for_one_cell if peak_filter[x // 4] and speed_filter[x // 512]]# eegh_fs = 5kHz and res frames are in 20 kHz
                avg_freq = len(res_filtered) / (len(time_filtered) / 5_000)
                if avg_freq >= 0.25: #Keeping only cells with an avg. firing rate of 0.25 Hz across the time kept by all filters
                        #This criteria is used by Nardin et al., 2023
                    firing_cell_count += 1
                    phases = [inst_phase[i // 4] for i in res_filtered]
                    a_bar, r, p_val = phaselocking_utils.stats(phases)
                    if p_val < 0.05:
                        a_bars.append(a_bar)
                        r_vals.append(r)
                        p_vals.append(p_val)
                        selective_cells.append(cell)
            save_filepath =  animal + '-data/' + animal + '-' + day + '/' + animal + '-' + day + '_' + session + '_' + cell_type + '_correct_phaselock.npz'
            np.savez(save_filepath, mean_direction = a_bars, median_vector_lengths = r_vals, p_values = p_vals, selective_cells = selective_cells)
            results.append(firing_cell_count)
            results.append(len(selective_cells))
            results.append((len(selective_cells) / firing_cell_count))
        df = pd.DataFrame([results])
        df.to_csv(output_file, mode = 'a', header = False, index = False) #mode 'a' means append (vs. overwriting the file)
        print(f'{animal} {day} {session} done')