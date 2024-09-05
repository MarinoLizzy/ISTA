import pandas as pd 
import phaselocking_utils
import bisect
import numpy as np

#To determine the preferred direction, MVL, and p-values of phase locked cells in the first half second half of training sessions, saving them as .npz files

animal = input('Animal (JC315, JC283, or JC274): ')

params = pd.read_csv('phaselocking_parameters.csv', header = [0, 1]) #the first two columns are headers
days = params[(animal, 'days')].tolist() #indexing by animal as the first header and days as the second
days = [str(int(i)) for i in days if not pd.isna(i)]
tet_par = params[(animal, 'tet par')].tolist()
tet_par = [int(i) for i in tet_par if not pd.isna(i)]
tet_total = params[(animal, 'num tetrodes')].tolist()
tet_total = [int(i) for i in tet_total if not pd.isna(i)]

def calculate(res, cell_type):
    a_bars = []
    r_vals = []
    p_vals = []
    selective_cells = []
    firing_cell_count = 0
    for cell in cell_types_dict[cell_type]:
        res_for_one_cell = [x for i, x in enumerate(res) if clu[i] == cell]
        res_filtered = [x for x in res_for_one_cell if peak_filter[x // 4] and speed_filter[x // 512]]
        avg_freq = len(res_filtered) / (len_eegh / 2 / 5_000)
        if avg_freq >= 0.25:
            firing_cell_count += 1
            phases = [inst_phase[i // 4] for i in res_filtered]
            a_bar, r, p_val = phaselocking_utils.stats(phases)
            if p_val < 0.05:
                a_bars.append(a_bar)
                r_vals.append(r)
                p_vals.append(p_val)
                selective_cells.append(cell)
    return a_bars, r_vals, p_vals, selective_cells, firing_cell_count

for i in range(len(days))[:1]:
    day = days[i]
    for session in ['training1', 'training2'][:1]:
        if day == '20220306' and session == 'training2':
            continue
        tetrode = tet_par[i]
        num_tetrodes = tet_total[i]
        speed_filepath = 'one_drive_data_speed/eight_arm_fig_data_andrea/analysis/' + animal + '-' + day + '/' + animal + '_' + day + '_' + session + '.speed_vladVersion' 
        filepath = animal + '-data/' + animal + '-' + day + '/' + animal + '-' + day + '_' + session

        #Get phase of HPC
        inst_phase, filtered_eegh = phaselocking_utils.get_phase(filepath, tetrode, num_tetrodes)
        speed_filter = phaselocking_utils.filter_for_speed(speed_filepath)
        # results.append(sum(speed_filter) / len(speed_filter)) #% time kept by speed filter
        peak_filter = phaselocking_utils.peaks_filter(filtered_eegh) 
            #To note that this is not based on the Hilbert transform but instead just the scipy.signal peaks function, there might be a discrepancy here
        # results.append((sum(peak_filter) / len(peak_filter))) #% time kept by peak filter


        #Get spikes
        res = phaselocking_utils.to_int_list(filepath + '.res')
        len_eegh = len(filtered_eegh)
        if res[-1] >= 4 * len_eegh + 3: #adding 3 because I use floor division later
            print(f'res is longer than eegh: {res[-1] // 4} is longer than {len_eegh} in eegh frames (5 kHz).')
            # import bisect
            cutoff_index = bisect.bisect_right(res, 4 * len_eegh + 3) #adding 3 as I use floor division #might need a  - 1) at the end of filtered_eegh)
            print(f'initial max res: {res[-1]}')
            res = res[:cutoff_index]
            print(f'final max res: {res[-1]}')
        # res_frames = np.arange(res[-1])
        # time_filtered = [x for x in res_frames if peak_filter[x // 4] and speed_filter[x // 512]] 
        # results.append((len(time_filtered)/ len(res_frames))) #% time kept by both filters

        #The max time that a spike could occur is len(eegh) / 2 times 4 since res frames are in 20 kHz and eegh is in 5 kHz
        res_middle = bisect.bisect_right(res, 2 * len_eegh)
        res_first_half = res[:res_middle]
        res_second_half = res[res_middle:]

        #For JC315 20240407, first half had len 353784 and second half had len 344224, seems right


        #Characterize clusters
        clu = phaselocking_utils.to_int_list(filepath + '.clu')
        _, ca1_clusters, pfc_right_clusters = phaselocking_utils.describe_clusters(animal + '-data/' + animal + '-' + day + '/' + animal + '-' + day)
            #pfc_clusters, ca1_clusters, pfc_right_clusters

        cell_types_dict = {
            'CA1': ca1_clusters,
            'PFC': pfc_right_clusters
        }

        for cell_type in ['CA1', 'PFC']:
            a_bar1, r_val1, p_val1, selective_cells1, firing_cell_count1 = calculate(res_first_half, cell_type)
            a_bar2, r_val2, p_val2, selective_cells2, firing_cell_count2 = calculate(res_second_half, cell_type)

            save_filepath =  animal + '-data/' + animal + '-' + day + '/' + animal + '-' + day + '_' + session + '_' + cell_type + '_phaselock_halves.npz'
            np.savez(save_filepath,
                    mean_direction1 = a_bar1, median_vector_lengths1 = r_val1, p_values1 = p_val1, selective_cells1 = selective_cells1, firing_count1 = firing_cell_count1,
                    mean_direction2 = a_bar2, median_vector_lengths2 = r_val2, p_values2 = p_val2, selective_cells2 = selective_cells2, firing_count2 = firing_cell_count2,
                    )
    
        print(f'{animal} {day} {session} done')