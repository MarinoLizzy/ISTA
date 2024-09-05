import pandas as pd 
import phaselocking_utils
import numpy as np
#Also requres scipy.signal

'''
Filepath for eegh and res and clu and des files: animal + '-data/' + animal + '-' + day + '/' + animal + '-' + day + '_' + session
Filepath for speed: 'eight_arm_fig_data_andrea/analysis/' + animal + '-' + day + '/' + animal + '_' + day + '_' + session + '.speed'
Req's 'phaselocking_parameters.csv'
'''

def calculate_phase_locking(animal, day, session, tetrode, num_tetrodes):
    '''
    Purpose:
        Save values for individual days as .npz files: 'mean_direction', 'median_vector_lengths', 'p_values', 'selective_cells'
        
    Parameters:
        Animal (str): 'JC315', 'JC283', or 'JC274'
        Day (str): a day of training of the corresponding animal (format ex: '20220304')
        Session (str): 'training1' or 'training2'
        Tetrode (int): tet_par
        Num_tetrodes (int): total number of tetrodes in that day's recordings
    
    Return values (lists of floats or strings): values to be appended to a .csv, in the same order as the listed .csv headers:
        'Animal', 'Day', 'Session', 'Tet Par', '% time kept by speed filter', '% time kept by peak filter', '% time kept by both filters',
        'Num ca1 clusters', 'ca1 firing cells', 'ca1 selective cells', '% Ca1 selective', 
        'Num pfc clusters', 'pfc firing cells', 'pfc selective cells', '% pfc selective'
    '''
    results = [animal, day, session, tetrode] #Initialize the row to be added to the csv
    filepath = animal + '-data/' + animal + '-' + day + '/' + animal + '-' + day + '_' + session
    speed_filepath = 'eight_arm_fig_data_andrea/analysis/' + animal + '-' + day + '/' + animal + '_' + day + '_' + session + '.speed'
        
    inst_phase, filtered_eegh = phaselocking_utils.get_phase(filepath, tetrode, num_tetrodes)
    speed_filter = phaselocking_utils.filter_for_speed(speed_filepath)
    results.append(sum(speed_filter) / len(speed_filter)) #% time kept by speed filter
    peak_filter = phaselocking_utils.peaks_filter(filtered_eegh) 
    results.append((sum(peak_filter) / len(peak_filter))) #% time kept by peak filter
    

    res = phaselocking_utils.to_int_list(filepath + '.res')
    if res[-1] // 4 >= len(filtered_eegh) - 1: #-1 because it's in indices
        print(f'res is longer than eegh: {res[-1] // 4} is longer than {len(filtered_eegh)} in eegh frames (5 kHz).')
        import bisect
        cutoff_index = bisect.bisect_right(res, 4 * (len(filtered_eegh) - 1) + 3) #adding 3 as I use floor division
        print(f'initial max res: {res[-1]}')
        res = res[:cutoff_index]
        print(f'final max res: {res[-1]}')
    res_frames = np.arange(res[-1])
    time_filtered = [x for x in res_frames if peak_filter[x // 4] and speed_filter[x // 512]] 
    results.append((len(time_filtered)/ len(res_frames))) #% time kept by both filters
    
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
        # for cell in pfc_right_clusters:
        for cell in cell_types_dict[cell_type]:
            res_for_one_cell = [x for i, x in enumerate(res) if clu[i] == cell]
            # eegh_fs = 5kHz and res frames are in 20 kHz
            res_filtered = [x for x in res_for_one_cell if peak_filter[x // 4] and speed_filter[x // 512]]
            avg_freq = len(res_filtered) / (len(time_filtered) / 5_000)
            # print(len(res_filtered) / len(res_for_one_cell))
            if avg_freq >= 0.25:
                firing_cell_count += 1
                phases = [inst_phase[i // 4] for i in res_filtered]
                a_bar, r, p_val = phaselocking_utils.stats(phases)
                if p_val < 0.05:
                    a_bars.append(a_bar)
                    r_vals.append(r)
                    p_vals.append(p_val)
                    selective_cells.append(cell)
        save_filepath =  animal + '-data/' + animal + '-' + day + '/' + animal + '-' + day + '_' + session + '_' + cell_type + '_phaselock.npz' #animal + '_phaselock_vals/'
        np.savez(save_filepath, mean_direction = a_bars, median_vector_lengths = r_vals, p_values = p_vals, selective_cells = selective_cells)
        results.append(firing_cell_count)
        results.append(len(selective_cells))
        results.append((len(selective_cells) / firing_cell_count))
    print(f'{animal} {day} {session} done')
    return results

animal = input('Animal (JC315, JC283, or JC274): ')


params = pd.read_csv('phaselocking_parameters.csv', header = [0, 1]) #the first two columns are headers
days = params[(animal, 'days')].tolist() #indexing by animal as the first header and days as the second
days = [str(int(i)) for i in days if not pd.isna(i)]
tet_par = params[(animal, 'tet par')].tolist()
tet_par = [int(i) for i in tet_par if not pd.isna(i)]
tet_total = params[(animal, 'num tetrodes')].tolist()
tet_total = [int(i) for i in tet_total if not pd.isna(i)]


#Name the output file and its column headers
output_file = animal + '_phaselocking.csv'
pd.DataFrame(columns = ['Animal', 'Day', 'Session', 'Tet Par', '% time kept by speed filter', '% time kept by peak filter', '% time kept by both filters',
                        'Num ca1 clusters', 'ca1 firing cells', 'ca1 selective cells', '% Ca1 selective', 
                        'Num pfc clusters', 'pfc firing cells', 'pfc selective cells', '% pfc selective']
                        ).to_csv(output_file, index = False)

for i in range(len(days)):
    for session in ['training1', 'training2']:
        results = calculate_phase_locking(animal, days[i], session, tet_par[i], tet_total[i])
        df = pd.DataFrame([results])
        df.to_csv(output_file, mode = 'a', header = False, index = False) #mode 'a' means append (vs. overwriting the file)
        #header=False prevents writing the header over and over
