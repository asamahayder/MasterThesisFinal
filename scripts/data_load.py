import os
import pickle
from datetime import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
import logger
from joblib import Parallel, delayed

def load_data(_, params):
    plt.rcParams.update({'font.size': 20})
    np.random.seed(42)
    data_folder_path = params["path_to_data_folder"]

    # Loading individual files and create the final data object
    files = os.listdir(data_folder_path)

    if "READ_ME.txt" in files:
        files.remove("READ_ME.txt")

    def load_file(file):
        with open(os.path.join(data_folder_path, file), 'rb') as f:
            return pickle.load(f)

    data = Parallel(n_jobs=-1)(delayed(load_file)(file) for file in tqdm(files, desc="Loading files"))

    for d in data:
        timestamp = d['time']
        d['date'] = datetime.fromisoformat(timestamp)

    data = sorted(data, key=lambda x: x['date'])

    # performing basic preprocessing

    # Fixing labeling errors
    data[51]['samplematrix'] = 'sample 7 PBS'
    data[51]['conc'] = 0.0

    data[52]['samplematrix'] = 'sample 8 g/PBS'
    data[52]['conc'] = 2.5

    data[53]['samplematrix'] = 'sample 6 g/PBS'
    data[53]['conc'] = 2.5


    # Fixing issue with sample ids due to multiple days
    for d in data:
        d['samplematrix_fixed'] = d['samplematrix']

    for d in data[37:]:
        values = d['samplematrix'].split()
        if len(values) > 1:
            id = int(values[1])
            new_id = id+18
            new_samplematrix = values[0] + " " + str(new_id) + " " + values[2]
            d['samplematrix_fixed'] = new_samplematrix


    # Removing air and NC as these are irrelevant for our purpose
    data = [d for d in data if not d['samplematrix_fixed'] == 'air']
    data = [d for d in data if not d['samplematrix_fixed'].split()[2] == 'NC']

    # Creating sample ID as separate field in data dict
    for d in data:
        d['sample_id'] = d['samplematrix_fixed'].split()[1]

    # Plotting signals
    x = data[4]['scan'][0]['forward_scan']['time']
    y = data[4]['scan'][0]['forward_scan']['signal']

    min_index = np.argmin(y)
    max_index = np.argmax(y)
    middle_index = math.floor((min_index + max_index) / 2)
    zoom_start = middle_index - 100
    zoom_end = middle_index + 100

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, y, label='Full Signal')
    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Full Pulse")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x[zoom_start:zoom_end], y[zoom_start:zoom_end], label='Zoomed Signal')
    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Zoomed Pulse")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("temp_plots/signals_visualized.png")
    plt.close()




    # logging Shape of the data
    logger.log("Raw Data Shape")
    logger.log("Number of samples: ", len(data))
    logger.log("Number of scans per sample: ", len(data[0]['scan']), " x2 (forward and backward scans)")
    logger.log("Number of time points per scan: ", len(data[0]['scan'][0]['forward_scan']['time']))
    logger.log("")




    # Plotting multiple signals onto same time axis zoomed in
    # the pulses are colored as a gradient from blue to red to show the progression of the pulses

    plt.figure(figsize=(15, 5))

    number_of_pulses = len(data)

    for i in range(0, number_of_pulses):
        x = data[i]['scan'][0]['forward_scan']['time']
        y = data[i]['scan'][0]['forward_scan']['signal']

        min_index = np.argmin(y)
        max_index = np.argmax(y)
        middle_index = math.floor((min_index + max_index) / 2)
        zoom_start = middle_index - 50
        zoom_end = middle_index + 50

        color = plt.cm.coolwarm(i / number_of_pulses)

        plt.plot(x[zoom_start:zoom_end], y[zoom_start:zoom_end], label='Pulse ' + str(i), color=color)

    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Zoomed Pulses")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("temp_plots/signals_timeshift_visualized.png")
    plt.close()







    plt.figure(figsize=(15, 5))

    number_of_pulses = len(data)

    # Extract unique days and map each day to a color
    unique_days = sorted(set(point['date'].day for point in data))
    day_to_color = {day: plt.cm.viridis(i / len(unique_days)) for i, day in enumerate(unique_days)}

    for i in range(0, number_of_pulses):
        x = data[i]['scan'][0]['forward_scan']['time']
        y = data[i]['scan'][0]['forward_scan']['signal']

        min_index = np.argmin(y)
        max_index = np.argmax(y)
        middle_index = math.floor((min_index + max_index) / 2)
        zoom_start = middle_index - 50
        zoom_end = middle_index + 50

        day = data[i]['date'].day
        color = day_to_color[day]

        plt.plot(x[zoom_start:zoom_end], y[zoom_start:zoom_end], label=f'Pulse {i} (Day {day})', color=color)

    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Zoomed Pulses Colored by Day")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("temp_plots/signals_timeshift_visualized_by_day.png")
    plt.close()
    





    # Plotting a single pulse

    plt.figure(figsize=(15, 5))

    number_of_pulses = len(data[0]['scan'])

    for i in range(0, number_of_pulses):
        x = data[0]['scan'][i]['forward_scan']['time']
        y = data[0]['scan'][i]['forward_scan']['signal']

        min_index = np.argmin(y)
        max_index = np.argmax(y)
        middle_index = math.floor((min_index + max_index) / 2)
        zoom_start = middle_index - 50
        zoom_end = middle_index + 50

        color = plt.cm.coolwarm(i / number_of_pulses)

        plt.plot(x[zoom_start:zoom_end], y[zoom_start:zoom_end], label='Pulse ' + str(i), color=color)

    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Zoomed Pulses")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("temp_plots/signals_timeshift_visualized_for_single_pulse.png")
    plt.close()

    return data