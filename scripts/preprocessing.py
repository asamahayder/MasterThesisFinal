import numpy as np
import math
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
from scipy.signal import correlate
import logger
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.decomposition import PCA




def preprocess_data_frequency_domain(data, params):
    np.random.seed(42)

    for d in data:
        for scan in d['scan']:
            # Standardizing the signal
            signal = scan['forward_scan']['signal']
            signal = (signal - np.mean(signal)) / np.std(signal)
            scan['forward_scan']['signal'] = signal

            # Normalizing the signal
            """ signal = scan['forward_scan']['signal']
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
            scan['forward_scan']['signal'] = signal """

        for ref in d['ref']:
            # Standardizing the signal
            signal = ref['forward_scan']['signal']
            signal = (signal - np.mean(signal)) / np.std(signal)
            ref['forward_scan']['signal'] = signal

            # Normalizing the signal
            """ signal = ref['forward_scan']['signal']
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
            ref['forward_scan']['signal'] = signal """

    for d in data:
        for scan in d['scan']:
            scan['fft'] = np.fft.rfft(scan['forward_scan']['signal'])
            scan['fft_amp'] = np.abs(scan['fft'])
            scan['fft_freq'] = np.fft.rfftfreq(len(scan['forward_scan']['signal'])) * (10**(-12)) # Converting to THz
        
        for ref in d['ref']:
            ref['fft'] = np.fft.rfft(ref['forward_scan']['signal'])
            ref['fft_amp'] = np.abs(ref['fft'])
            ref['fft_freq'] = np.fft.rfftfreq(len(ref['forward_scan']['signal'])) * (10**(-12)) # Converting to THz
    

    # only using the first scan for each pulse
    final_data = [d['scan'][0]['fft_amp'] for d in data if not 'bare' in d['samplematrix_fixed']]
    labels = [d['samplematrix_fixed'].split()[2] for d in data if not 'bare' in d['samplematrix_fixed']]
    ids = [d['samplematrix_fixed'].split()[1] for d in data if not 'bare' in d['samplematrix_fixed']]

    # creating a dict where for each sample id, we have all the scans

    all_scans = {}
    for d in data:
        if 'bare' in d['samplematrix_fixed']:
            continue
        
        all_scans[d['samplematrix_fixed'].split()[1]] = [np.asarray(scan['fft_amp']) for scan in d['scan']]


    le = LabelEncoder()
    le.classes_ = np.array(["g/PBS", "PBS"])
    labels = le.transform(labels)

    X = np.asarray(final_data)
    y = np.asarray(labels)
    groups = np.asarray(ids)

    return X, y, groups, all_scans

def preprocess_data_simple(data, params):
    np.random.seed(42)

    # Aligning the pulses
    reference_pulse = data[0]['scan'][0]['forward_scan']['signal']
    def find_shift(reference, pulse):
        correlation = correlate(pulse, reference)
        shift = np.argmax(correlation) - len(pulse)
        return shift
    
    max_shift = 0
    for pulse in tqdm(data, desc='Processing Pulses'):
        for scan in pulse['scan']:
            signal = scan['forward_scan']['signal']
            shift = find_shift(reference_pulse, signal)
            aligned_signal = np.roll(signal, -shift)
            if abs(shift) > max_shift:
                max_shift = abs(shift)
            scan['forward_scan']['signal'] = aligned_signal

        for ref in pulse['ref']:
            signal_forward = ref['forward_scan']['signal']
            shift_forward = find_shift(reference_pulse, signal_forward)
            aligned_pulse_forward = np.roll(signal_forward, -shift_forward)
            if abs(shift_forward) > max_shift:
                max_shift = abs(shift_forward)
            ref['forward_scan']['signal'] = aligned_pulse_forward


    # using max shift to cut off the ends of the pulses
    for pulse in tqdm(data, desc='Cutting off ends of pulses'):
        for scan in pulse['scan']:
            scan['forward_scan']['signal'] = scan['forward_scan']['signal'][max_shift:-max_shift]
        
        for ref in pulse['ref']:
            ref['forward_scan']['signal'] = ref['forward_scan']['signal'][max_shift:-max_shift]

    
    for d in data:
        for scan in d['scan']:
            # Standardizing the signal
            signal = scan['forward_scan']['signal']
            signal = (signal - np.mean(signal)) / np.std(signal)
            scan['forward_scan']['signal'] = signal

            # Normalizing the signal
            signal = scan['forward_scan']['signal']
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
            scan['forward_scan']['signal'] = signal

        for ref in d['ref']:
            # Standardizing the signal
            signal = ref['forward_scan']['signal']
            signal = (signal - np.mean(signal)) / np.std(signal)
            ref['forward_scan']['signal'] = signal

            # Normalizing the signal
            signal = ref['forward_scan']['signal']
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
            ref['forward_scan']['signal'] = signal
    

    # subtracting ref
    for pulse in data:
       ref_forward_scans = [ref['forward_scan']['signal'] for ref in pulse['ref']]
       avg_ref =  np.mean(ref_forward_scans, axis=0)
       for scan in pulse['scan']:
           scan['forward_scan']['signal'] = scan['forward_scan']['signal'] - avg_ref
 

    # subtracting bare
    treated_data = [d for d in data if not 'bare' in d['samplematrix_fixed']]
    bare_data = [d for d in data if 'bare' in d['samplematrix_fixed']]
    for pulse in treated_data:
        corresponding_bare = [d for d in bare_data if d['samplematrix_fixed'].split()[1] == pulse['samplematrix_fixed'].split()[1]][0]
        bare_forward_scans = [scan['forward_scan']['signal'] for scan in corresponding_bare['scan']]
        avg_bare = np.mean(bare_forward_scans, axis=0) 
        for scan in pulse['scan']:
            scan['forward_scan']['signal'] = scan['forward_scan']['signal'] - avg_bare
 
    
    # These values were found by experimenting and inspecting the resulting pulses
    window_start = 3750
    window_end = 4850

    final_data = []
    labels = []
    ids = []

    # final_data = [d['scan'][0]['forward_scan']['signal'][window_start:window_end] for d in data if not 'bare' in d['samplematrix_fixed']]
    # labels = [d['samplematrix_fixed'].split()[2] for d in data if not 'bare' in d['samplematrix_fixed']]
    # ids = [d['samplematrix_fixed'].split()[1] for d in data if not 'bare' in d['samplematrix_fixed']]

    for d in treated_data:
        for scan in d['scan']:
            final_data.append(scan['forward_scan']['signal'][window_start:window_end])
            labels.append(d['samplematrix_fixed'].split()[2])
            ids.append(d['samplematrix_fixed'].split()[1])

    le = LabelEncoder()
    le.classes_ = np.array(["g/PBS", "PBS"])
    labels = le.transform(labels)

    X = np.asarray(final_data)
    y = np.asarray(labels)
    groups = np.asarray(ids)

    return X, y, groups

def preprocess_data_time_domain(data, params):
    plt.rcParams.update({'font.size': 25})
    for d in data:
        for scan in d['scan']:
            # Standardizing the signal
            signal = scan['forward_scan']['signal']
            signal = (signal - np.mean(signal)) / np.std(signal)
            scan['forward_scan']['signal'] = signal

            # Normalizing the signal
            signal = scan['forward_scan']['signal']
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
            scan['forward_scan']['signal'] = signal

        for ref in d['ref']:
            # Standardizing the signal
            signal = ref['forward_scan']['signal']
            signal = (signal - np.mean(signal)) / np.std(signal)
            ref['forward_scan']['signal'] = signal

            # Normalizing the signal
            signal = ref['forward_scan']['signal']
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
            ref['forward_scan']['signal'] = signal
    

    # subtracting ref
    for pulse in data:
       ref_forward_scans = [ref['forward_scan']['signal'] for ref in pulse['ref']]
       avg_ref =  np.mean(ref_forward_scans, axis=0)
       for scan in pulse['scan']:
           scan['forward_scan']['signal'] = scan['forward_scan']['signal'] - avg_ref


    # creating a dict where for each sample id, we have all the scans
    all_scans = {}
    for d in data:
        if 'bare' in d['samplematrix_fixed']:
            continue
        
        all_scans[d['samplematrix_fixed'].split()[1]] = [np.asarray(scan['forward_scan']['signal']) for scan in d['scan']]

    """
    final_data = []
    labels = []
    ids = []

    for d in data:
        if 'bare' in d['samplematrix_fixed']:
            continue
        for scan in d['scan']:
            final_data.append(scan['forward_scan']['signal'])
            labels.append(d['samplematrix_fixed'].split()[2])
            ids.append(d['samplematrix_fixed'].split()[1])
    """


    final_data = [d['scan'][0]['forward_scan']['signal'] for d in data if not 'bare' in d['samplematrix_fixed']]
    labels = [d['samplematrix_fixed'].split()[2] for d in data if not 'bare' in d['samplematrix_fixed']]
    ids = [d['samplematrix_fixed'].split()[1] for d in data if not 'bare' in d['samplematrix_fixed']]

    le = LabelEncoder()
    le.classes_ = np.array(["g/PBS", "PBS"])
    labels = le.transform(labels)

    X = np.asarray(final_data)
    y = np.asarray(labels)
    groups = np.asarray(ids)


    all_data_for_plotting = []
    labels_for_plotting = []

    for d in data:
        if 'bare' in d['samplematrix_fixed']:
            continue
        for scan in d['scan']:
            all_data_for_plotting.append(scan['forward_scan']['signal'])
            labels_for_plotting.append(d['samplematrix_fixed'].split()[2])

    labels_for_plotting = le.transform(labels_for_plotting)
    labels_for_plotting = np.asarray(labels_for_plotting)

    pca = PCA()
    pca.fit(all_data_for_plotting)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.9) + 1
    X_pca = pca.transform(all_data_for_plotting)


    # calculating the correlation between the principal components and the target variable
    correlations = []
    for i in range(X_pca.shape[1]):
        corr = np.corrcoef(X_pca[:, i], labels_for_plotting)[0, 1]
        correlations.append(corr)

    correlation_df = pd.DataFrame({
        'Principal Component': [f'PC{i+1}' for i in range(X_pca.shape[1])],
        'Correlation with Class': correlations
    })

    # making two subplots:
    # a plot of the data plotted on the first two principal components
    # a plot of the data plotted on the principal components with the highest correlation with the target variable

    plt.subplots(1, 2, figsize=(20, 10))
    plt.subplot(1, 2, 1)
    unique_labels = np.unique(labels_for_plotting)
    for cls, marker in zip(unique_labels, ['o', 'x']):
        plt.scatter(X_pca[labels_for_plotting == cls, 0], X_pca[labels_for_plotting == cls, 1], label=le.inverse_transform([cls])[0], marker=marker)
    
    plt.xlabel("1st PC")
    plt.ylabel("2nd PC")
    plt.title("Data plotted on the first two PCs")
    plt.legend()

    plt.subplot(1, 2, 2)
    positive_corr_idx = correlation_df['Correlation with Class'].idxmax()
    negative_corr_idx = correlation_df['Correlation with Class'].idxmin()
    for cls, marker in zip(unique_labels, ['o', 'x']):
        plt.scatter(X_pca[labels_for_plotting == cls, positive_corr_idx], X_pca[labels_for_plotting == cls, negative_corr_idx], label=le.inverse_transform([cls])[0], marker=marker)

    plt.xlabel(f'PC{positive_corr_idx + 1} (Highest Positive Correlation)')
    plt.ylabel(f'PC{negative_corr_idx + 1} (Highest Negative Correlation)')
    plt.xticks(rotation=45)
    plt.title('Data Visualization on PCs with Highest Correlations')
    plt.legend()

    plt.tight_layout()
    plt.savefig("temp_plots/PCA_plot.png")
    plt.close()


    

    plt.figure(figsize=(10, 5))
    #plt.plot(correlation_df['Principal Component'], correlation_df['Correlation with Class'], marker='o')
    plt.bar(correlation_df['Principal Component'], correlation_df['Correlation with Class'])
    plt.xlabel('Principal Component')
    plt.ylabel('Correlation with Class')
    plt.xticks([])

    plt.title('Correlation of PCs with Class')
    plt.tight_layout()
    plt.savefig("temp_plots/PCA_correlated_plot.png")
    plt.close()


    return X, y, groups, all_scans

def preprocess_data(data, params):
    np.random.seed(42)
    
    reference_pulse = data[0]['scan'][0]['forward_scan']['signal']
    min_index = np.argmin(reference_pulse)
    max_index = np.argmax(reference_pulse)
    middle_index = math.floor((min_index+max_index)/2)

    pulse_window_size = params['pulse_window_size']
    window_start = middle_index - pulse_window_size
    window_end = middle_index + pulse_window_size

    # Function to find the shift using cross-correlation
    def find_shift(reference, pulse):
        correlation = correlate(pulse, reference)
        shift = np.argmax(correlation) - len(pulse)
        return shift

    max_shift = 0
    aligned_data = []
    for pulse in tqdm(data, desc='Processing Pulses'):
        for scan in pulse['scan']:
            signal_forward = scan['forward_scan']['signal']
            signal_backward = scan['backward_scan']['signal']
            shift_forward = find_shift(reference_pulse, signal_forward)
            shift_backward = find_shift(reference_pulse, signal_backward)
            aligned_pulse_forward = np.roll(signal_forward, -shift_forward)
            aligned_pulse_backward = np.roll(signal_backward, -shift_backward)

            if abs(shift_forward) > max_shift:
                max_shift = abs(shift_forward)
            if abs(shift_backward) > max_shift:
                max_shift = abs(shift_backward)

            #aligned_pulse_forward = aligned_pulse_forward[window_start:window_end] # Cropping the signal to the window so all pulses have the same length
            #aligned_pulse_backward = aligned_pulse_backward[window_start:window_end] # Cropping the signal to the window so all pulses have the same length
            scan['forward_scan']['aligned'] = aligned_pulse_forward
            scan['backward_scan']['aligned'] = aligned_pulse_backward

            aligned_data.append(aligned_pulse_forward)
            aligned_data.append(aligned_pulse_backward)
        for ref in pulse['ref']:
            signal_forward = ref['forward_scan']['signal']
            signal_backward = ref['backward_scan']['signal']
            shift_forward = find_shift(reference_pulse, signal_forward)
            shift_backward = find_shift(reference_pulse, signal_backward)
            aligned_pulse_forward = np.roll(signal_forward, -shift_forward)
            aligned_pulse_backward = np.roll(signal_backward, -shift_backward)

            if abs(shift_forward) > max_shift:
                max_shift = abs(shift_forward)
            if abs(shift_backward) > max_shift:
                max_shift = abs(shift_backward)
            #aligned_pulse_forward = aligned_pulse_forward[window_start:window_end] # Cropping the signal to the window so all pulses have the same length
            #aligned_pulse_backward = aligned_pulse_backward[window_start:window_end] # Cropping the signal to the window so all pulses have the same length
            ref['forward_scan']['aligned'] = aligned_pulse_forward
            ref['backward_scan']['aligned'] = aligned_pulse_backward

            aligned_data.append(aligned_pulse_forward)
            aligned_data.append(aligned_pulse_backward)

        # Averaging the reference pulses to get a single reference pulse
        ref_forward_scans_aligned = [ref['forward_scan']['aligned'] for ref in pulse['ref']]
        ref_backward_scans_aligned = [ref['backward_scan']['aligned'] for ref in pulse['ref']]
        pulse['avg_ref_aligned'] = np.mean(ref_forward_scans_aligned + ref_backward_scans_aligned, axis=0)


    logger.log("Max shift: ", max_shift)
    logger.log("Shape of aligned data before cutting: ", np.array(aligned_data).shape)

    # using max shift to cut off the ends of the pulses
    for pulse in tqdm(data, desc='Cutting off ends of pulses'):
        for scan in pulse['scan']:
            # cutting of each end by max_shift
            scan['forward_scan']['aligned'] = scan['forward_scan']['aligned'][max_shift:-max_shift]
            scan['backward_scan']['aligned'] = scan['backward_scan']['aligned'][max_shift:-max_shift]
        
        pulse['avg_ref_aligned'] = pulse['avg_ref_aligned'][max_shift:-max_shift]

    
    aligned_data = [pulse[max_shift:-max_shift] for pulse in aligned_data]

    logger.log("Shape of aligned data after cutting: ", np.array(aligned_data).shape)
    

    plt.figure(figsize=(15, 5))

    # Pick random 30 pulses from aligned data to plot

    random_indices = np.random.choice(len(aligned_data), 30, replace=False)


    for i in random_indices:
        x = np.arange(len(aligned_data[i]))
        y = aligned_data[i]

        min_index = np.argmin(y)
        max_index = np.argmax(y)
        middle_index = math.floor((min_index + max_index) / 2)
        zoom_start = middle_index - 50
        zoom_end = middle_index + 50

        plt.plot(x[zoom_start:zoom_end], y[zoom_start:zoom_end], label='Pulse ' + str(i))

    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Zoomed Pulses")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("temp_plots/signals_aligned_zoomed.png")
    plt.close()

    # plotting without zoom

    plt.figure(figsize=(15, 5))

    for i in random_indices:
        x = np.arange(len(aligned_data[i]))
        y = aligned_data[i]

        plt.plot(x, y, label='Pulse ' + str(i))

    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Pulses")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("temp_plots/signals_aligned.png")
    plt.close()




    # Using the averaged reference pulse to remove baseline noise from the average pulse

    for pulse in tqdm(data, desc='Removing Baseline Noise using air reference'):
        for scan in pulse['scan']:
            signal_forward = scan['forward_scan']['aligned']
            signal_backward = scan['backward_scan']['aligned']
            
            scan['forward_scan']['cleaned'] = signal_forward - pulse['avg_ref_aligned']
            scan['backward_scan']['cleaned'] = signal_backward - pulse['avg_ref_aligned']


    # plotting without zoom

    plt.figure(figsize=(15, 5))

    i = 0

    # Pick random 10 pulses from cleaned data to plot

    random_indices = np.random.choice(len(data), 5, replace=False)

    for idx in random_indices:
        pulse = data[idx]
        for scan in pulse['scan']:
            x = np.arange(len(scan['forward_scan']['cleaned']))
            y = scan['forward_scan']['cleaned']
            plt.plot(x, y, label='Pulse ' + str(i))
            i += 1
            
            x = np.arange(len(scan['backward_scan']['cleaned']))
            y = scan['backward_scan']['cleaned']
            plt.plot(x, y, label='Pulse ' + str(i))
            i += 1


    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Cleaned Pulses")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("temp_plots/signal_air_subtracted.png")
    plt.close()


    treated_data = [d for d in data if not 'bare' in d['samplematrix_fixed']]
    final_data = []
    final_groups = []
    labels = []

    for i in tqdm(range(len(treated_data)), desc='Processing Treated Data'):
        treated = treated_data[i]

        # getting the id
        id = treated['samplematrix_fixed'].split()[1]

        # forward and backwards
        all_treated_scans = [scan['forward_scan']['cleaned'] for scan in treated['scan']] + [scan['backward_scan']['cleaned'] for scan in treated['scan']]

        for j in range(len(all_treated_scans)):
            final_data.append(all_treated_scans[j])
            final_groups.append(id)
            labels.append(treated['samplematrix_fixed'].split()[2])
                

    final_data = np.asarray(final_data)
    final_groups = np.asarray(final_groups)
    labels = np.asarray(labels)


    # plotting the final data

    plt.figure(figsize=(15, 5))

    # picking 30 random indicies from final data

    random_indices = np.random.choice(len(final_data), 30, replace=False)

    for i in random_indices:
        x = np.arange(len(final_data[i]))
        y = final_data[i]

        plt.plot(x, y, label='Pulse ' + str(i))

    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Final Data")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("temp_plots/signal_bare_subtracted.png")
    plt.close()


    # Label encoding the labels to 0s and 1s
    le = LabelEncoder()
    le.classes_ = np.array(["g/PBS", "PBS"])
    labels = le.transform(labels)

    X = final_data
    y = labels
    groups = final_groups


    
    return X, y, groups