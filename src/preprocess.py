'''
Date: Sat Jun 23, 2022
Author: Ha Le
This file contains function to preprocess the data.
'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Global variables
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

# setting up legends for the plots
red_patch = mpatches.Patch(color='red', label='workingPC')
blue_patch = mpatches.Patch(color='blue', label='stand')
green_path = mpatches.Patch(color = 'green', label = 'stand+walk+stairs')
yellow_path = mpatches.Patch(color = 'y', label = 'walk')
magenta_path = mpatches.Patch(color = 'm', label = 'stairs')
cyan_path = mpatches.Patch(color = 'c', label = 'walk+talk')
orange_path = mpatches.Patch(color = 'orange', label = 'talk')
patches = [red_patch, blue_patch, green_path, yellow_path, magenta_path, cyan_path, orange_path]

def load_dataset(subject: int):
    '''
    Loads the dataset for the given subject.
    subject: int: subject number
    '''
    if subject <= 0 or subject > 15:
        print("Subject must be in the range of 1-15.")
        return None
    subject_df = pd.read_csv(CURRENT_PATH + "/../data/raw_data/" + str(subject) + '.csv'
                        , names = ['timestamp', 'x', 'y', 'z', 'activity'])

    return subject_df

def mark_activity(activity: int):
    '''
    Given the id of the activity, output the color code of such activity.
    activity: int: activity number
    '''
    look_up = ['w', 'r', 'g', 'b', 'y', 'm', 'c', 'orange']
    if activity < 0 or activity > 7:
        print("Activity must be in the range of 0-7.")
        print("Activity: " + str(activity))
        return None
    return look_up[activity]

def split_data_by_activity(subject_df: pd.DataFrame):
    '''
    Splits the dataframe by activity.
    subject_df: pd.DataFrame: dataframe of the subject
    Returns: list of pd.DataFrame: list of dataframes of each activity
    '''
    activity_dfs = []
    timestamp_lst = subject_df['timestamp'].tolist()
    activity_lst = subject_df['activity'].tolist()
    x_lst = subject_df['x'].tolist()
    y_lst = subject_df['y'].tolist()
    z_lst = subject_df['z'].tolist()
    current_activity = [[], [], [], [], []]
    for i in range(1, len(activity_lst)):
        if activity_lst[i] != activity_lst[i-1]:
            activity_dfs.append(pd.DataFrame({'timestamp': current_activity[0]
                                            , 'x': current_activity[1],
                                            'y': current_activity[2],
                                            'z': current_activity[3],
                                            'activity': current_activity[4]}))
            current_activity = [[], [], [], [], []]
        current_activity[0].append(timestamp_lst[i])
        current_activity[1].append(x_lst[i])
        current_activity[2].append(y_lst[i])
        current_activity[3].append(z_lst[i])
        current_activity[4].append(activity_lst[i])
    return activity_dfs

def visualize_data(subject: int):
    '''
    Visualizes the data of the given subject.
    subject: int: subject number
    '''
    subject_df = load_dataset(subject)
    fig, axs = plt.subplots(3, 1, figsize=(10, 7)) # pylint suppresssion
    activity_dfs = split_data_by_activity(subject_df)
    for activity_df in activity_dfs:
        activity_color = mark_activity(activity_df['activity'][0])
        directions = ['x', 'y', 'z']
        for j in range(3):
            axs[j].plot(activity_df['timestamp'], activity_df[directions[j]], color=activity_color)
    plt.legend(handles=patches, loc='upper right', ncol=4)
    plt.show()

def downsampling_activity(activity_df, sampling_rate = "125ms"):
    '''
    Downsamples the activity sequence input as dataframe.
    activity_df: pd.DataFrame: dataframe of the activity
    Returns: pd.DataFrame: downsampled dataframe
    '''
    # assigning timestamp
    activity = activity_df['activity'].tolist()[0]
    base = pd.date_range("0:00", freq="19.23ms", periods=len(activity_df)).tolist()
    activity_df['timestamp'] = base
    activity_df['timestamp'] = pd.to_datetime(activity_df['timestamp'])
    activity_df = activity_df.set_index('timestamp')
    # downsampling
    activity_df = activity_df.resample(sampling_rate).sum()
    activity_df['activity'] = activity
    activity_df = activity_df.reset_index()
    return activity_df

def visualize_sequence(activity_df: pd.DataFrame, sampling_rate = "125ms"):
    '''
    Visualizes the sequence of the given activity.
    activity_df: pd.DataFrame: dataframe of the activity
    '''
    activity_df = downsampling_activity(activity_df, sampling_rate)
    activity_color = mark_activity(activity_df['activity'][0])
    fig, axs = plt.subplots(3, 1, figsize=(10, 7)) # pylint suppresssion
    directions = ['x', 'y', 'z']
    for j in range(3):
        axs[j].plot(activity_df['timestamp'], activity_df[directions[j]], color=activity_color)
    plt.legend(handles=patches, loc='upper right', ncol=4)
    plt.title('Activity: ' + str(activity_color))
    plt.show()

#test downsampling
# subject_df = load_dataset(1)
# activity_df = split_data_by_activity(subject_df)[1]
# downsampled_activity_df = downsampling_activity(activity_df)
# print(downsampled_activity_df)
# print(f"original length: {len(activity_df)}")
# print(f"downsampled length: {len(downsampled_activity_df)}")
# visualize_sequence(activity_df)
# visualize_data(1)
