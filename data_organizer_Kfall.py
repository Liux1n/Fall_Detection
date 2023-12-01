import os
import pandas as pd
import re
import numpy as np
from scipy.signal import resample
import torch
import time


def process_data(sensor_folder, 
                 label_file, 
                 window_size,
                 threshold,
                 num_window_fall_data,
                 num_window_not_fall_data
                 ):
        # 100fps
        window_size = window_size

        #threshold = 0.1
        threshold = threshold
        num_window_fall_data = num_window_fall_data
        #num_window_fall_data = 10
        num_threshold = int(window_size * threshold)
        #print(num_threshold)
        #num_window_not_fall_data = 3
        num_window_not_fall_data = num_window_not_fall_data

        # create a dataframe for the window and label
        window_data = pd.DataFrame(columns=['label', 'data'])
        #print(window_data)
        #print(label)
        label_data = pd.read_excel(label_file)

        fall_file = {'file_name' : [], 'Fall_onset_frame' : [], 'Fall_impact_frame' : []}
        #print(task_code)
        # Iterate over each row in the label file
        for col, row in label_data.iterrows():
            #print(col, row)
            if not pd.isna(row['Task Code (Task ID)']):

                #print(row['Task Code (Task ID)'])
                match = re.search(r'\((.*?)\)', str(row['Task Code (Task ID)']))
                if match:
                    task_id = match.group(1)
                    #print(task_id)
                idx = 0
                
                while True:
                    if col+idx+1 == len(label_data) or not pd.isna(label_data.iloc[col+idx+1, 0]):
                        trial_id = label_data.iloc[col+idx, 2]
                        file_name = f'S06T{task_id}R0{trial_id}.csv'
                        Fall_onset_frame = label_data.iloc[col+idx, 3]
                        Fall_impact_frame = label_data.iloc[col+idx, 4]
                        fall_file['file_name'].append(file_name)
                        fall_file['Fall_onset_frame'].append(Fall_onset_frame)
                        fall_file['Fall_impact_frame'].append(Fall_impact_frame)
                        #print(file_name, Fall_onset_frame, Fall_impact_frame)
                        break
                    trial_id = label_data.iloc[col+idx, 2]
                    file_name = f'S06T{task_id}R0{trial_id}.csv'
                    Fall_onset_frame = label_data.iloc[col+idx, 3]
                    Fall_impact_frame = label_data.iloc[col+idx, 4]
                    fall_file['file_name'].append(file_name)
                    fall_file['Fall_onset_frame'].append(Fall_onset_frame)
                    fall_file['Fall_impact_frame'].append(Fall_impact_frame)
                    idx += 1

        

        #iterare over each file in the sensor data folder
        for file in os.listdir(sensor_folder):

            if file in fall_file['file_name']:

                # give the mathcing fall data file name
                fall_data = fall_file['file_name'][fall_file['file_name'].index(file)]
                # open the fall data file
                data_file = os.path.join(sensor_folder, fall_data)
                data = pd.read_csv(data_file)
                # drop the column of TimeStamp(s)
                data = data.drop(columns=['TimeStamp(s)'])
                data['label'] = 0
   
                Fall_onset_frame = fall_file['Fall_onset_frame'][fall_file['file_name'].index(file)]
                Fall_impact_frame = fall_file['Fall_impact_frame'][fall_file['file_name'].index(file)]

                # label: 0 for before fall, 1 for pre-impact fall, 2 for pos-impact fall
                # add a column of label to the first column
                # frame count smaller than Fall_onset_frame is labeled as 0,
                # frame count between Fall_onset_frame and Fall_impact_frame is labeled as 1,
                # frame count between Fall_impact_frame and (Fall_impact_frame+50) is labeled as 2,(removed)
                # frame count larger than (Fall_impact_frame+50) removed for labeling

                # Label the rows as 'pre-impact fall'
                data.loc[(data['FrameCounter'] >= Fall_onset_frame) & (data['FrameCounter'] < Fall_impact_frame), 'label'] = 1

                #data.loc[(data['FrameCounter'] >= Fall_impact_frame) & (data['FrameCounter'] <= Fall_impact_frame + 50), 'label'] = 2
                # remove the column of FrameCounter
                data = data.drop(columns=['FrameCounter'])

                # TODO: now we have the widow with widnow_size = 256. We have now three labels: 0, 1 for not FAll and FALL
                #  for the window.
                # To translate the temporal (i.e. per-sample) labels described above into window labels we adopted to the 
                # following criteria:
                #    • each window containing at least 10% of readings labeled as FALL was tagged as FALL altogether;
                #    • each non-FALL window in which the majority of samples was labeled as ALERT is tagged as ALERT;
                #    • any other window was tagged to the BKG class.

                if len(data) - window_size > len(data)/2:
                    for i in range(num_window_fall_data):
                        # randomly select a starting index for the window
                        start_index = np.random.randint(0, len(data) - window_size)
                        # select the window of data
                        window = data.iloc[start_index : start_index + window_size]

                        # randomly select 10 windows from the fall data
                        #each window containing at least 10% of readings labeled as FALL was tagged as FALL altogether;
                        if len(window[window['label'] == 1]) >= num_threshold:
                            
                            window = window.drop(columns=['label'])
                            values = np.array(window.values)
                    
                            new_window = pd.DataFrame({'label': [1], 'data': [values]})
                            window_data = pd.concat([window_data, new_window], ignore_index=True)
                            #print(new_window['label'])
                
                elif 0 < len(data) - window_size < len(data)/2:
                    # sample half number of windows
                    for i in range(int(num_window_fall_data/2)):
                        # randomly select a starting index for the window
                        start_index = np.random.randint(0, len(data) - window_size)
                        # select the window of data
                        window = data.iloc[start_index : start_index + window_size]

                        # randomly select 10 windows from the fall data
                        #each window containing at least 10% of readings labeled as FALL was tagged as FALL altogether;
                        if len(window[window['label'] == 1]) >= num_threshold:
                            
                            window = window.drop(columns=['label'])
                            values = np.array(window.values)
                    
                            new_window = pd.DataFrame({'label': [1], 'data': [values]})
                            window_data = pd.concat([window_data, new_window], ignore_index=True)
                            
                else:
                    # resample the data to the window_size if the data is too short
                    window = resample(data, window_size)
                    if len(window[window['label'] == 1]) >= num_threshold:
                        window = window.drop(columns=['label'])
                        values = np.array(window.values)
                        new_window = pd.DataFrame({'label': [1], 'data': [values]})
                        window_data = pd.concat([window_data, new_window], ignore_index=True)
                        print(new_window['label'])

                
            else:
                #print(f"{file} is not a fall data file.")
                data_file = os.path.join(sensor_folder, file)

                data = pd.read_csv(data_file)
                data = data.drop(columns=['TimeStamp(s)'])
                data = data.drop(columns=['FrameCounter'])
                data['label'] = 0

                if len(data) - window_size > len(data)/2:



                    # randomly select 3 windows from the non-fall data
                    for i in range(num_window_not_fall_data):
                        # randomly select a starting index for the window
                        
                        start_index = np.random.randint(0, len(data) - window_size)
                        # select the window of data
                        window = data.iloc[start_index : start_index + window_size]
                        window = window.drop(columns=['label'])
                        window = np.array(window.values)
                        new_window = pd.DataFrame({'label': [0], 'data': [window]})
                        window_data = pd.concat([window_data, new_window], ignore_index=True)
                        #print(new_window)

                elif 0 < len(data) - window_size < len(data)/2:
                    # sample half number of windows
                    for i in range(int(num_window_not_fall_data/2)):
                        # randomly select a starting index for the window
                        start_index = np.random.randint(0, len(data) - window_size)
                        # select the window of data
                        window = data.iloc[start_index : start_index + window_size]
                        window = window.drop(columns=['label'])
                        window = np.array(window.values)
                        new_window = pd.DataFrame({'label': [0], 'data': [window]})
                        window_data = pd.concat([window_data, new_window], ignore_index=True)

                else:
                    v
                    data = data.drop(columns=['label'])
                    window = resample(data, window_size)
                    new_window = pd.DataFrame({'label': [0], 'data': [window]})
                    window_data = pd.concat([window_data, new_window], ignore_index=True)
   
        

        data = window_data['data']
        label = window_data['label']

        # concatenate the data indices
        data = data.values
        label = np.array(label.values)
        #label = label.values
        data = np.concatenate(data.tolist()).reshape(-1, window_size, 9)

        return data, label


def DataOrganizer(sensor_data_folder,
                       label_data_folder, 
                       window_size,
                       threshold = 0.1,
                       num_window_fall_data = 10,
                       num_window_not_fall_data = 3):
    
    window_size = window_size
    threshold = threshold
    num_window_fall_data = num_window_fall_data
    num_window_not_fall_data = num_window_not_fall_data

    all_data = []
    all_label = []
    # delete the hidden file .DS_Store
    if '.DS_Store' in os.listdir(sensor_data_folder):
        os.remove(os.path.join(sensor_data_folder, '.DS_Store'))
    
    # get length of the sensor data folder
    num_sensor_data = len(os.listdir(sensor_data_folder))
    count = 0
    # iterate over each file in the sensor data folder and use process_data function to process the data
    for file in os.listdir(sensor_data_folder):
        #print(file)
        print(f"Processing {count+1}/{num_sensor_data} folder...")

        sensor_folder = os.path.join(sensor_data_folder, file)
        label_file_name = file.split('.')[0] + '_label.xlsx'
        label_file = os.path.join(label_data_folder, label_file_name)

        data, label = process_data(sensor_folder, 
                                   label_file, 
                                   window_size,
                                   threshold,
                                   num_window_fall_data,
                                   num_window_not_fall_data
                                   )

        all_data.append(data)
        all_label.append(label)
        count += 1
    
    # Concatenate the all_data and all_label lists
    all_data = np.concatenate(all_data)
    all_label = np.concatenate(all_label)
    
    return all_data, all_label






