import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

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
        return 
    subject_df = pd.read_csv(CURRENT_PATH + "/../data/raw_data/" + str(subject) + '.csv', names = ['timestamp', 'x', 'y', 'z', 'activity'])

    return subject_df

def mark_activity(activity: int):
    '''
    Given the id of the activity, output the color code of such activity.
    activity: int: activity number
    '''
    if activity < 0 or activity > 7:
        print("Activity must be in the range of 0-7.")
        print("Activity: " + str(activity))
        return 
    if activity == 0:
        return 'w'
    if activity == 1:
        return "r"
    elif activity == 2:
        return "g"
    elif activity == 3:
        return "b"
    elif activity == 4:
        return "y"
    elif activity == 5:
        return "m"
    elif activity == 6:
        return "c"
    else:
        return "orange"

def split_data_by_activity(subject_df: pd.DataFrame):
    '''
    Splits the dataframe by activity.
    subject_df: pd.DataFrame: dataframe of the subject
    Returns: list of pd.DataFrame: list of dataframes of each activity
    '''
    activity_dfs = []
    timestamp_lst = subject_df['timestamp'].tolist()
    activity_lst = subject_df['activity'].tolist()
    xs = subject_df['x'].tolist()
    ys = subject_df['y'].tolist()
    zs = subject_df['z'].tolist()
    current_activity = [[], [], [], [], []]
    for i in range(1, len(activity_lst)):
        if activity_lst[i] != activity_lst[i-1]:
            activity_dfs.append(pd.DataFrame({'timestamp': current_activity[0], 'x': current_activity[1], 'y': current_activity[2], 'z': current_activity[3], 'activity': current_activity[4]}))
            current_activity = [[], [], [], [], []]
        current_activity[0].append(timestamp_lst[i])
        current_activity[1].append(xs[i])
        current_activity[2].append(ys[i])
        current_activity[3].append(zs[i])
        current_activity[4].append(activity_lst[i])
    return activity_dfs

def visualize_data(subject: int):
    subject_df = load_dataset(subject)
    fig, axs = plt.subplots(3, 1, figsize=(10, 7))
    activity_dfs = split_data_by_activity(subject_df)
    for i in range(len(activity_dfs)):
        current_df = activity_dfs[i]
        activity_color = current_df['activity'].apply(lambda x: mark_activity(x)).tolist()[0]
        dir = ['x', 'y', 'z']
        for j in range(3):
            axs[j].plot(current_df['timestamp'], current_df[dir[j]], color=activity_color)
    plt.legend(handles=patches, loc='upper right', ncol=4)
    plt.show()
            

test_subject = load_dataset(1)
print(split_data_by_activity(test_subject))
visualize_data(1)

