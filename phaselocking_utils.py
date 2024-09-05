import numpy as np
from scipy.signal import butter, sosfilt, hilbert
from scipy.signal import find_peaks

eegh_fs = 5_000 #Hz

whl_fs = 39.0625 #Hz

def get_phase(filepath, tet_par, num_tetrodes):
    '''
    Purpose: Filter the eegh data for the theta frequency range and determine instantaneous phase values.
    Parameters:
        filepath: (str) filepath, not including .eegh
        tet_par: (int) which tetrode of eegh data to use - not the absolute tetrode number 
            Input this differently for different days depending on which has the most clusters extracted (suggesting low impedance)
        num_tetrodes: (int) how many total tetrodes are included in the recording (to be included in the parameters .csv)
    Return values (np.arrays):
        inst_phase: the phase at each point in the filtered data, in radians (note this does not wrap from 0 to 2pi)
        filtered: theta-filtered eegh signal
    '''
    #import lfp
    eegh_raw = np.fromfile(filepath + '.eegh', dtype = np.uint16)
    eegh = eegh_raw[int(tet_par*len(eegh_raw)/num_tetrodes) : int((tet_par+1)*len(eegh_raw)/num_tetrodes)] #take one tetrode
    
    #filter lfp
    sos = butter(3, [5, 12], btype = 'bandpass', output = 'sos', analog = False, fs = eegh_fs)
        #Different studies use different acceptable range values here, ex. 5-28 (Kaefer), 5-15 (Nardin), 5-10 (Siapas)
            #This will affect what percent of the data is filtered out later when using a interpeak-inverval filter for 7-10Hz
            #But also it doesn't seem to work to just make this band narrower and not include the additional interpeak-interval filter
                #Even though we're really only looking for signal in 7-10 Hz
    filtered = sosfilt(sos, eegh)
    
    #hilbert transform
    analytic_signal = hilbert(filtered)
    inst_phase = np.unwrap(np.angle(analytic_signal))

    return inst_phase, filtered

def filter_for_speed(speed_filepath): 
    '''
    Purpose: Create a filter so that data (both eegh and spikes) will only be kept if the animal is in motion (>= 5 cm/s)
    Parameter: speed_filepath: (str) full filepath to speed data, at the same fs as the .whl files.
    Return value (np.array of bool): 'True' for .whl timestamps where the animal is in motion.
    '''
    file = open(speed_filepath)
    speed = file.readlines()
    file.close()
    for i in range(len(speed)):
        speed[i] = float(speed[i].strip('\n'))

    speed_filter = np.array([i >= 5 for i in speed])
    speed_filter = np.append(speed_filter, False) #To handle the case in which there's a spike in the last small portion of time

    return speed_filter

def peaks_filter(filtered):
    '''
    Purpose: Create a filter to keep data when the animal is in motion and the interpeak interval of the theta-filter is between 7-10 Hz.
    Parameters:
        filtered (np.array): theta filtered signal (outputted by the get_phase function)
        speed_filter (np.array of bool): Whether the animal is in motion
    Return value: (np.array of bool): True when the animal is in motion and the theta-filtered signal is 7-10 Hz, in alignment with eegh frames.
    '''
    peaks, _ = find_peaks(filtered)
    
    interpeak_filter = np.zeros(len(filtered))
    for i in range(len(peaks) - 1):
        interpeak_interval = (peaks[i + 1] - peaks[i]) / eegh_fs 
        if interpeak_interval >= 0.1 and interpeak_interval <= 0.14:
            for i in range(peaks[i], peaks[i + 1]):
                interpeak_filter[i] = 1
        #the absolute first and last chunks of this will remain zero/False, but these will probably be screened out by the speed filter anyway
    
    interpeak_filter_bool = interpeak_filter.astype(bool)

    return interpeak_filter_bool

def to_int_list(filename):
    '''
    Purpose: To import .res and .clu
        .clu file is the cluster ID (putative neuron ID) for the corresponding .res file spikes
        .res file is the "frame" at sampling frequency 20kHz in which spikes occur
    Parameter: Full file path + ending of the .res or .clu file
    Return value (list of ints): Either the .res or .clu values
    '''
    file = open(filename)
    file_list = file.readlines()
    file.close()
    for i in range(len(file_list)):
        file_list[i] = int(file_list[i].strip('\n'))

    return file_list

def stats(spike_phase):
    '''
    From J.H. Zar Biostatistical Analysis 5th Ed., 2010, p. 625
    Purpose: to return some statistical values for an individual putative cell
    Parameter: (list of floats [rad]) the phase of the theta hippocampal oscillation during the spikes for one putative neuron
        This is calculated in calculate_phase_locking() 
    Return values:
    a_bar (float, [rads]): preferred firing phase (resultant vector direction)
        Ranges from (-pi, pi], as this is the range of outputs for np.arctan2()
    r: mean resultant vector length (MVL) -- ranges from 0-1
    p-val: an approximation to the probability of Rayleigh's R
    '''
    n = len(spike_phase) #sample size
    X = (1/n) * sum([np.cos(i) for i in spike_phase]) #x-coord of the resultant vector
    Y = (1/n) * sum([np.sin(i) for i in spike_phase]) #y-coord of the resultant vector
    r = np.sqrt(X**2 + Y**2) #Mean resultant vector length
    R = n * r #R = Rayleigh's R
    # z = n * r**2 #z = Rayleigh's z
    a_bar = np.arctan2(Y, X) #Mean direction, in rad
    p_val = np.exp((1 + 4*n + 4*(n**2 - R**2))**0.5 - (1 + 2*n))

    return a_bar, r, p_val

def describe_clusters(filepath):
    '''
    Parameter: (str) filepath without '.des' or '.des_full'
    Return values: All lists of ints. Putative neuron numbers in accordance with the .clu file:
        pfc_clusters: putative pyramidal neurons in the pfc
        ca1_clusters: putative pyramidal neurons in the ca1
        pfc_right_clusters: putative pyramidal neurons only in the right hemisphere of the pfc
            - Since the Kaefer thesis only uses ipsalateral pfc neurons, and all ca1 recordings are in the right hemisphere
    '''
    file = open(filepath + '.des')
    des_raw = file.readlines()
    file.close()
    des = [i.strip('\n') for i in des_raw]
    pfc_clusters = [i + 2 for i, x in enumerate(des) if x == 'pp'] #selects all putative prefrontal pyramidal neurons (bilateral)
        #The value given in clu is described by des[value - 2] since cluster 1 is noise
    ca1_clusters = [i + 2 for i, x in enumerate(des) if x == 'p1'] #selects all putative hippocampal pyramidal neurons (right hemisphere)
    
    file = open(filepath + '.des_full')
    des_raw = file.readlines()
    file.close()
    des_full = [i.strip('\n') for i in des_raw]
    pfc_right_clusters = [i + 2 for i, x in enumerate(des_full) if x == 'pfc_right']
    
    return pfc_clusters, ca1_clusters, pfc_right_clusters


if __name__ == '__main__':
    def calculate_phase_locking(animal, day, session, tetrode = 20):
        '''
        Return values (lists of ints or floats)
            a_bars: preferred direction of a cell -- note this is in radians and ranges from -pi to pi (as this is the range of arctan)
        '''
        filepath = animal + '-data/' + animal + '-' + day + '/' + animal + '-' + day + '_' + session
        speed_filepath = 'eight_arm_fig_data_andrea/analysis/' + animal + '-' + day + '/' + animal + '_' + day + '_' + session + '.speed'
            
        inst_phase, filtered_eegh = get_phase(filepath, tetrode_number=20)
        speed_filter = filter_for_speed(speed_filepath)
        peak_filter = peaks_filter(filtered_eegh) 
        
        print(f'Time kept by peak_filter: {(sum(peak_filter) / len(peak_filter)) * 100:.0f}%') 
        
        res = to_int_list(filepath + '.res')
        res_frames = np.arange(res[-1])
        time_filtered = [x for x in res_frames if peak_filter[x // 4] and speed_filter[x // 512]] 

        print(f'Kept by both filters: {(len(time_filtered)/ len(res_frames)) * 100:.0f}%')
        
        clu = to_int_list(filepath + '.clu')
        _, ca1_clusters, _ = describe_clusters(animal + '-data/' + animal + '-' + day + '/' + animal + '-' + day)
            #pfc_clusters, ca1_clusters, pfc_right_clusters


        a_bars = [] #preferred phase
        r_vals = [] #MVL
        p_vals = []
        selective_cells = []
        firing_cell_count = 0
        # for cell in pfc_right_clusters:
        for cell in ca1_clusters:
            res_for_one_cell = [x for i, x in enumerate(res) if clu[i] == cell]
            # eegh_fs = 5kHz and res frames are in 20 kHz
            res_filtered = [x for x in res_for_one_cell if peak_filter[x // 4] and speed_filter[x // 512]]
            avg_freq = len(res_filtered) / (len(time_filtered) / 5_000)
            # print(len(res_filtered) / len(res_for_one_cell))
            if avg_freq >= 0.25:
                firing_cell_count += 1
                phases = [inst_phase[i // 4] for i in res_filtered]
                a_bar, r, p_val = stats(phases)
                if p_val < 0.05:
                    a_bars.append(a_bar)
                    r_vals.append(r)
                    p_vals.append(p_val)
                    selective_cells.append(cell)
        print(f'{len(selective_cells)} of {firing_cell_count} are selective, {(len(selective_cells) / firing_cell_count) * 100:.0f}%')
        return a_bars, r_vals, p_vals, selective_cells
    animal = 'JC315'
    # day = days_andrea[3][1]

    day = input('day:')
    # day = '20240331'
    tetrode = input('tetrode:')
    # tetrode = 20
    # session = 'training1'
    session = input('session:')


    print(animal, day, session, 'tetrode ' + str(tetrode) + ' ca1 cells') 
    a_bars, r_vals, p_vals, selective_cells = calculate_phase_locking(animal, day, session, tetrode = tetrode)

    filepath = animal + '-data/' + animal + '-' + day + '/' + animal + '-' + day + '_' + session
    np.savez(filepath + '_phase-lock_ca1.npz', mean_direction = a_bars, median_vector_lengths = r_vals, p_values = p_vals)