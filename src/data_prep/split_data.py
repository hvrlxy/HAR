import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import eda
import preprocess as pp

def get_activity_code(activity: pd.DataFrame):
    '''
    Given the id of the activity, output the color code of such activity.
    activity: int: activity
    Returns: str: id of the activity
    '''
    return activity['activity'][0]

def summarize_subject_activities(subject: int):
    '''
    Summarize the activities of the subject.
    subject: int: subject number
    '''
    # make sure that the subject is in the range of 1-15
    if subject < 1 or subject > 15:
        raise ValueError("Subject must be in the range of 1-15.")
    # load the dataframe of the subject
    subject_df = pp.load_dataset(subject)
    activity_dfs = pp.split_data_by_activity(subject_df)
    # summarize the activities of the subject
    activity_summary = {}
    for i in range(len(activity_dfs)):
        # get the activity code
        activity_code = get_activity_code(activity_dfs[i])
        # put the activity code and the activity labels into the dictionary
        activity_summary[i] = (activity_code, eda.activity_lookup[activity_code])
    return activity_summary

def split_data_by_interval(activity: pd.DataFrame,
                            periods: int = 260):
    '''
    Split the dataframe of the activity into intervals of periods,
    with 50% overlap between the intervals.
    activity: pd.DataFrame: dataframe of the activity
    periods: int: number of periods in each interval,
                default is 260 (5 seconds)
    Returns: list: list of dataframes of the intervals
    '''
    # get the number of samples in the activity
    num_samples = len(activity)
    # split the activity into intervals
    intervals = []
    for i in range(0, num_samples, periods//2):
        intervals.append(activity[i:i+periods])
    if len(intervals[-1]) < periods:
        intervals.pop()
    return intervals

def generate_1D_data(activity_window: pd.DataFrame):
    '''
    Generate the 1D data of the activity window.
    activity_window: pd.DataFrame: dataframe of the activity window
    Returns: list: list of 1D data of the activity window
    '''
    # get the number of samples in the activity window
    num_samples = len(activity_window)
    # generate the 1D data of the activity window
    data = {"min_x": None, "max_x": None, "mean_x": None, "std_x": None, "range_x": None,
            "min_y": None, "max_y": None, "mean_y": None, "std_y": None, "range_y": None,
            "min_z": None, "max_z": None, "mean_z": None, "std_z": None, "range_z": None}
    directions = ['x', 'y', 'z']
    # add data by directions
    for direction in directions:
        data[f"min_{direction}"] = activity_window[direction].min()
        data[f"max_{direction}"] = activity_window[direction].max()
        data[f"mean_{direction}"] = activity_window[direction].mean()
        data[f"std_{direction}"] = activity_window[direction].std()
        data[f"range_{direction}"] = activity_window[direction].max() - activity_window[direction].min()
    return data

def generate_data_per_activity(activity: pd.DataFrame):
    '''
    Generate the data of the activity.
    activity: pd.DataFrame: dataframe of the activity's raw dataset
    Returns: pd.DataFrame: dataframe of the activity's intervals data
    '''
    # get activitys'id
    activity_code = get_activity_code(activity)
    # split the activity into intervals
    intervals = split_data_by_interval(activity)
    # generate the data per activity
    data = {"min_x": [], "max_x": [], "mean_x": [], "std_x": [], "range_x": [],
            "min_y": [], "max_y": [], "mean_y": [], "std_y": [], "range_y": [],
            "min_z": [], "max_z": [], "mean_z": [], "std_z": [], "range_z": []}
    for interval in intervals:
        # generate the 1D data of the interval
        interval_data = generate_1D_data(interval)
        # add the 1D data to the data per activity
        for key in interval_data:
            data[key].append(interval_data[key])
    # add the data per activity to the dataframe
    activity_data = pd.DataFrame(data)
    # add the activity code to the dataframe
    activity_data['activity'] = activity_code
    return activity_data

def generate_data_per_subject(subject: int):
    '''
    Generate the data of the subject.
    subject: int: subject number
    Returns: pd.DataFrame: dataframe of the subject's data
    '''
    # make sure that the subject is in the range of 1-15
    if subject < 1 or subject > 15:
        raise ValueError("Subject must be in the range of 1-15.")
    # load the dataframe of the subject
    subject_df = pp.load_dataset(subject)
    # split the dataframe of the subject into activities
    activity_dfs = pp.split_data_by_activity(subject_df)
    # generate the data per activity
    data = []
    for activity in activity_dfs:
        data.append(generate_data_per_activity(activity))
    # concatenate the data per activity
    data = pd.concat(data)
    return data

# test
# subject_df = pp.load_dataset(1)
# activity_dfs = pp.split_data_by_activity(subject_df)
# activity_data = generate_data_per_activity(activity_dfs[1])
# print(generate_data_per_subject(1))