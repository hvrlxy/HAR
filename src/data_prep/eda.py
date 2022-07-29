'''
Date: Sat Jun 23, 2022
Author: Ha Le
This file contains function to perform EDA.
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime
from tabulate import tabulate
from fbprophet import Prophet
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from preprocess import load_dataset
from preprocess import split_data_by_activity
from preprocess import downsampling_activity
from suppress_output import suppress_stdout_stderr

# Global variables
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

#matching activity id to activity name
activity_lookup = {0: "None",
                    1 : "workingPC",
                    2 : "stand",
                    3 : "stand+walk+stairs",
                    4 : "walk",
                    5 : "stairs",
                    6 : "walk+talk",
                    7 : "talk"}

def calculate_sequence_length(activity_df: pd.DataFrame):
    '''
    Calculates the sequence length of the given activity.
    In this dataset, data are collected at 52 Hz
    activity_df: pd.DataFrame: activity dataframe
    '''
    return len(activity_df)/52

def calculate_activity_duration(subject_df: pd.DataFrame):
    '''
    Calculates the duration of the given activity.
    subject_df: pd.DataFrame: subject raw dataframe
    '''
    # get the activity dataframe
    activity_dfs = split_data_by_activity(subject_df)
    activity_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    # calculate the duration of each activity
    for activity in activity_dfs:
        activity_code = activity['activity'][0]
        activity_dict[activity_code] += calculate_sequence_length(activity)
    # return the dataframe containing the duration of each activity
    report_df = pd.DataFrame(columns = ['Activity', "Duration (in secs)"])
    for i in range(1, 8):
        report_df.loc[i] = [activity_lookup[i], activity_dict[i]]
    return report_df

def generate_sequence_length_report(file_name: str = "sequence_length_report.txt"):
    '''
    Generates a report of the sequence length of each activity.
    '''
    with open(CURRENT_PATH + f"/../../reports/tables/{file_name}",
                "w+", encoding="utf-8") as report_file:
        report_table = pd.DataFrame(columns = ['Subject', "workingPC",
                                                "stand", "stand+walk+stairs",
                                                "walk", "stairs", "walk+talk",
                                                "talk"])
        for subject in range(1, 16):
            subject_df = load_dataset(subject)
            # subject_df = subject_df.drop(subject_df[subject_df.activity == 0].index)
            if subject_df is None:
                continue
            activity_dfs = split_data_by_activity(subject_df)
            activity_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
            for activity in activity_dfs:
                activity_code = activity['activity'][0]
                activity_dict[activity_code] += calculate_sequence_length(activity)

            report_table.loc[subject] = [subject, activity_dict[1],
                                                activity_dict[2],
                                                activity_dict[3],
                                                activity_dict[4],
                                                activity_dict[5],
                                                activity_dict[6],
                                                activity_dict[7]]
        report_file.write("SEQUENCE LENGTH REPORT PER SUBJECT:\n")
        report_file.write(tabulate(report_table, headers='keys', tablefmt='psql'))

def merge_activity_sequence(activity_dfs: list):
    '''
    Merges the activity of the same id sequence into one dataframe.
    activity_dfs: list: list of activity dataframes
    '''
    # create a new dict to store the merged activity
    activity_df_dict = {1: None, 2: None, 3: None,
                        4: None, 5: None, 6: None, 7: None}
    for activity in activity_dfs:
        activity_code = activity['activity'][0]
        if activity_code == 0:
            continue
        if activity_df_dict[activity_code] is None: # first activity
            activity_df_dict[activity_code] = activity
        else:
            activity_df_dict[activity_code] = activity_df_dict[activity_code].append(activity)
    # store data into a list
    new_activity_dfs = []
    for i in range(1, 8):
        new_activity_dfs.append(activity_df_dict[i])
    return new_activity_dfs

def get_activity_stats(activity_df: pd.DataFrame):
    '''
    Returns the statistics of the given activity.
    activity_df: pd.DataFrame: activity dataframe
    '''
    reports = activity_df.describe()
    #drop the count row
    reports.drop(index = "count", inplace=True)
    #drop the timestamp and activity columns
    if "timestamp" in reports.index:
        reports.drop(index = "timestamp", inplace=True)
    if "activity" in reports.index:
        reports.drop(index = "activity", inplace=True)
    return reports.T

def generate_basic_stats_reports():
    '''
    Generates a report of the basic statistics of each activity.
    '''
    for subject in range(1, 16):
        # load the dataset
        with open(CURRENT_PATH + f"/../../reports/tables/basic_stats/basic_stats_report_{subject}.txt",
                                "w+", encoding="utf-8") as report_file:
            subject_df = load_dataset(subject)
            activity_dfs = split_data_by_activity(subject_df)
            activity_dfs = merge_activity_sequence(activity_dfs)
            # generate the report for each activity
            for activity in activity_dfs:
                activity_code = int(activity['activity'].tolist()[0])
                activity_stats = get_activity_stats(activity)

                report_file.write(f"BASIC STATS REPORT FOR SUBJECT {subject} "
                            + f"AND ACTIVITY {activity_lookup[activity_code]} ({activity_code})\n")
                report_file.write(tabulate(activity_stats, headers='keys', tablefmt='psql'))
                report_file.write("\n")

def generate_violin_plot(activity_dfs: pd.DataFrame, subject: int = 1):
    '''
    Generates a violin plot of the given subject.
    activity_dfs: pd.DataFrame: activity dataframe
    '''
    activity_dfs = merge_activity_sequence(activity_dfs)
    # generate the violin plot
    directions = ["x", "y", "z"]
    activity_lst = ["workingPC", "stand", "stand+\nwalk+\nstairs",
                    "walk", "stairs", "walk+\ntalk", "talk"]
    pos = [1,2,3,4,5,6,7] # position of the activities on the y-axis
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 7))
    for direction in directions:
        data = []
        for activity in activity_dfs:
            data.append(np.array(activity[direction].tolist(), dtype= np.float64))
        axs[directions.index(direction)].violinplot(data, pos,
                points=20, widths=0.3, vert=False, showmeans=True, showextrema=True,
                showmedians=True)
        axs[directions.index(direction)].set_title(f"Axis {direction}")
        # set the y-axis to reflect corresponding activity
        if direction == "x":
            axs[directions.index(direction)].set_yticks(pos)
            axs[directions.index(direction)].set_yticklabels(activity_lst)
    # set title and save the plot
    fig.suptitle(f"Violin plot of basic statistics, Subject {subject}")
    return plt
    

def generate_violin_basic_stats_report(file_name: str = "violin_basic_stats"):
    '''
    Generate the violin plot to compare different activities.
    '''
    for i in range(1, 16):
        # load the data
        report_file = CURRENT_PATH + f"/../../reports/imgs/basic_stats/violin_plots/{file_name}_{i}.png"
        subject_df = load_dataset(i)
        activity_dfs = split_data_by_activity(subject_df)
        activity_dfs = merge_activity_sequence(activity_dfs)
        # generate the violin plot
        directions = ["x", "y", "z"]
        activity_lst = ["workingPC", "stand", "stand+\nwalk+\nstairs",
                        "walk", "stairs", "walk+\ntalk", "talk"]
        pos = [1,2,3,4,5,6,7] # position of the activities on the y-axis
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 7))
        for direction in directions:
            data = []
            for activity in activity_dfs:
                data.append(np.array(activity[direction].tolist(), dtype= np.float64))
            axs[directions.index(direction)].violinplot(data, pos,
                    points=20, widths=0.3, vert=False, showmeans=True, showextrema=True,
                    showmedians=True)
            axs[directions.index(direction)].set_title(f"Axis {direction}")
            # set the y-axis to reflect corresponding activity
            if direction == "x":
                axs[directions.index(direction)].set_yticks(pos)
                axs[directions.index(direction)].set_yticklabels(activity_lst)
        # set title and save the plot
        fig.suptitle(f"Violin plot of basic statistics, Subject {i}")
        plt.savefig(report_file)

def generate_histogram(activity_dfs: pd.DataFrame, subject: int = 1):
    '''
    Generates a histogram of the given activity.
    activities_df: pd.DataFrame: activity dataframe
    '''
    # create a new dict to store the merged activity
    activity_dfs = merge_activity_sequence(activity_dfs)
    # initialize the histogram
    fig, axs = plt.subplots(nrows=7, ncols=1, figsize=(10, 15))
    fig.suptitle(f"Histogram of basic statistics, Subject {subject}")
    for activity in activity_dfs:
        activity_code = int(activity['activity'].tolist()[0])
        # need to create a new dataframe to store the data according
        # to the required format of a stacked histogram
        # read: https://seaborn.pydata.org/generated/seaborn.histplot.html
        hist_df = pd.DataFrame(columns=['data', 'axis'])
        data_lst = activity['x'].tolist() + activity['y'].tolist() + activity['z'].tolist()
        axis_lst = ['x' for i in range(len(activity))] + ['y' for i in range(len(activity))] + ['z' for i in range(len(activity))]
        hist_df['data'] = data_lst
        hist_df['axis'] = axis_lst
        # generate the histogram using seaborn
        sns.histplot(ax = axs[activity_code - 1],data=hist_df, x="data", hue="axis", bins=100)
        axs[activity_code - 1].set_title(f"Histogram of Activity {activity_lookup[activity_code]}")
    # increase the spaces between the plots
    fig.tight_layout(pad = 1.0)
    return plt

def generate_histogram_report(file_name:str = "histogram"):
    '''
    Generate the histogram of the data.
    '''
    for i in range(1, 16):
        # load the data
        report_file = CURRENT_PATH + f"/../../reports/imgs/basic_stats/histogram/{file_name}_{i}.png"
        subject_df = load_dataset(i)
        activity_dfs = split_data_by_activity(subject_df)
        activity_dfs = merge_activity_sequence(activity_dfs)
        # initialize the histogram
        fig, axs = plt.subplots(nrows=7, ncols=1, figsize=(10, 15))
        fig.suptitle(f"Histogram of basic statistics, Subject {i}")
        for activity in activity_dfs:
            activity_code = int(activity['activity'].tolist()[0])
            # need to create a new dataframe to store the data according
            # to the required format of a stacked histogram
            # read: https://seaborn.pydata.org/generated/seaborn.histplot.html
            hist_df = pd.DataFrame(columns=['data', 'axis'])
            data_lst = activity['x'].tolist() + activity['y'].tolist() + activity['z'].tolist()
            axis_lst = ['x' for i in range(len(activity))] + ['y' for i in range(len(activity))] + ['z' for i in range(len(activity))]
            hist_df['data'] = data_lst
            hist_df['axis'] = axis_lst
            # generate the histogram using seaborn
            sns.histplot(ax = axs[activity_code - 1],data=hist_df, x="data", hue="axis", bins=100)
            axs[activity_code - 1].set_title(f"Histogram of Activity {activity_lookup[activity_code]}")
        # increase the spaces between the plots
        fig.tight_layout(pad = 1.0)
        plt.savefig(report_file)

# OUTLIER DETECTION
def generate_timestamp(activity_df: pd.DataFrame, start = datetime.today(), period: int = None, freq: str = '1D'):
    '''
    Generates a timestamp for the given activity.
    activity_df: pd.DataFrame: activity dataframe
    start: datetime.datetime: start date of the timestamp
    period: int: period of the timestamp
    freq: str: frequency of the timestamp
    Returns: pd.date_range: timestamp list
    '''
    if period is None:
        period = len(activity_df)
    return pd.date_range(start, periods=period, freq=freq)

# Detecting outliers using forecasting
def in_sample_forecast(activity_df: pd.DataFrame, axis = 'x'):
    '''
    Returns the in-sample forecast of the given activity.
    activity_df: pd.DataFrame: activity dataframe
    '''
    if axis not in ['x', 'y', 'z']:
        raise ValueError("axis must be 'x', 'y' or 'z'")
    # the forecast_df need to have columns with names 'y' and 'ds'
    forecast_df = pd.DataFrame(columns=['ds', 'y'])
    # ds columns need to be datetime objects
    forecast_range = generate_timestamp(activity_df, start="0:00", freq="19.23ms")
    forecast_df['y'] = activity_df[axis]
    forecast_df['ds'] = forecast_range
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    # get the in-sample forecast
    forecast_model = Prophet()
    # the models will print out the iterations, suppress them
    with suppress_stdout_stderr():
        forecast_model.fit(forecast_df)
    forecast = forecast_model.predict(pd.DataFrame(forecast_df['ds']))
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def detect_fbprophet_outlier(forecast_df: pd.DataFrame, activity: list):
    '''
    Detects the outliers using fbprophet. Outliers are defined as points
    outside the confidence interval bandwidth.
    forecast_df: pd.DataFrame: forecast dataframe, including 'ds', 'yhat',
    'yhat_lower', 'yhat_upper'
    activity_df: list: list of actual data
    '''
    forecast_df['y'] = activity
    forecast_df = forecast_df[(forecast_df.y > forecast_df.yhat_upper)
                    | (forecast_df.y < forecast_df.yhat_lower)]
    return forecast_df

def plot_in_sample_forecast(activity_df: pd.DataFrame, file_path: str = None):
    '''
    Plots the in-sample forecast of the given activity.
    activity_df: pd.DataFrame: activity dataframe
    file_path: str: path to the file to save the plot
    '''
    directions = ['x', 'y', 'z']
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
    activity_code = activity_df['activity'].tolist()[0]
    for direction in directions:
        forecast = in_sample_forecast(activity_df, direction)
        activity_df['ds'] = forecast['ds']
        # plot the in-sample forecast
        axs[directions.index(direction)].plot(forecast['ds'], forecast['yhat'], label='forecast')
        # plot the confidence interval
        axs[directions.index(direction)].fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.5)
        # plot the actual data
        axs[directions.index(direction)].plot(activity_df['ds'], activity_df[direction], label='actual')
        # detect outliers
        outlier_df = detect_fbprophet_outlier(forecast, activity_df[direction])
        # plot the outliers
        axs[directions.index(direction)].scatter(outlier_df['ds'], outlier_df['y'], c='r', s=10, label='outlier')
        # set up title and legend
        axs[directions.index(direction)].set_title(f"In-sample forecast of {direction}-axis")
        axs[directions.index(direction)].legend()
    fig.suptitle(f"In-sample forecast of Activity {activity_lookup[activity_code].upper()} ({activity_code})")
    if file_path is not None:
        plt.savefig(file_path)
    return plt

def generate_fbprophet_outlier_plot(start_subject: int = 1, end_subject: int = 15):
    '''
    Generates the outlier plot for all the activities.
    '''
    for i in range(start_subject, end_subject + 1):
        try:
            # load the data
            report_file = CURRENT_PATH + f"/../../reports/imgs/outliers/fbprophet/{i}/"
            subject_df = load_dataset(i)
            activity_dfs = split_data_by_activity(subject_df)
            # activity_dfs = [downsampling_activity(activity_df) for activity_df in activity_dfs]
            for j in range(len(activity_dfs)):
                plot_in_sample_forecast(activity_dfs[j], file_path=report_file + f'activity_{j}.png')
        except Exception:
            continue

# detecting outliers using clustering techniques
def isolation_forest_detection(activity_df: pd.DataFrame, axis = 'x'):
    '''
    Detects the outliers using isolation forest.
    activity_df: pd.DataFrame: activity dataframe
    axis: str: axis to detect outliers
    Returns: pd.DataFrame: outlier dataframe'''
    if_model=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.1),max_features=1.0)
    # fit the model, suppress the stdout warnings
    with suppress_stdout_stderr():
        if_model.fit(activity_df[[axis]])
    outlier_df = pd.DataFrame()
    outlier_df[axis] = activity_df[axis]
    # predict raw anomaly score, -1.0 indicates anomaly, 1.0 is normal
    outlier_df['scores'] = if_model.decision_function(activity_df[[axis]])
    # threshold to define outliers
    outlier_df['anomaly'] = if_model.predict(activity_df[[axis]])
    return outlier_df

def plot_isolation_forest_outliers(activity_df: pd.DataFrame, file_path: str = None):
    '''
    Plots the outliers using isolation forest.
    activity_df: pd.DataFrame: activity dataframe
    file_path: str: path to the file to save the plot
    '''
    directions = ['x', 'y', 'z']
    # downsample the data
    # activity_df = downsampling_activity(activity_df)
    activity_code = activity_df['activity'].tolist()[0]
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
    for direction in directions:
        outlier_df = isolation_forest_detection(activity_df, direction)
        timestamp_lst = generate_timestamp(outlier_df)
        outlier_df['ds'] = timestamp_lst
        # plot the data
        axs[directions.index(direction)].plot(outlier_df['ds'], outlier_df[direction], label='actual')
        # highlight the outliers
        outlier_df = outlier_df[outlier_df['anomaly'] == -1.0]
        axs[directions.index(direction)].scatter(outlier_df['ds'], outlier_df[direction], c='r', s=10, label='outlier')
        # set up title and legend
        axs[directions.index(direction)].set_title(f"Isolation Forest outlier detection of {direction}-axis")
        axs[directions.index(direction)].legend()
    fig.suptitle(f"Isolation Forest outlier detection of Activity {activity_lookup[activity_code].upper()} ({activity_code})")
    if file_path is not None:
        plt.savefig(file_path)
    return plt

def generate_isolation_forest_outlier_plot(start_subject: int = 1, end_subject: int = 15):
    '''
    Generates the outlier plot using isolation forest algorithm
    for all the activities.
    '''
    for i in range(start_subject, end_subject + 1):
        try:
            # load the data
            report_file = CURRENT_PATH + f"/../../reports/imgs/outliers/clustering/{i}/"
            subject_df = load_dataset(i)
            activity_dfs = split_data_by_activity(subject_df)
            for j in range(len(activity_dfs)):
                plot_isolation_forest_outliers(activity_dfs[j], file_path=report_file + f'activity_{j}.png')
        except Exception:
            continue

# HANDLING OUTLIERS
# handling outliers using winsorization method
def winsorizing_ts(activity_df: pd.DataFrame, axes = ['x', 'y', 'z']):
    '''
    Handles the outliers using winsorizing method.
    activity_df: pd.DataFrame: activity dataframe
    axis: str: axis to handle outliers
    Returns: pd.DataFrame: fixed dataset
    '''
    # performing winsorizing on all indicated axes
    for axis in axes:
        winsored_values = np.array(activity_df[axis])
        # winsorize the top 5% and bottom 5% of the data
        winsored_values = winsorize(winsored_values, limits=[0.05, 0.05])
        #replace it into the dataframe
        activity_df[axis] = winsored_values
    return activity_df
    
def removing_outliers(activity_df: pd.DataFrame, removed_indices: list = []):
    '''
    Removes the outliers.
    activity_df: pd.DataFrame: activity dataframe
    axis: str: axis to handle outliers
    Returns: pd.DataFrame: fixed dataset
    '''
    # reset the indices incase it has been messed up
    activity_df.reset_index(drop=True, inplace=True)
    # remove the outliers
    activity_df.drop(removed_indices, axis=0, inplace=True)
    # reset the indices since some of them has been deleted
    activity_df.reset_index(drop=True, inplace=True)
    return activity_df


#test
# generate_sequence_length_report()
# generate_basic_stats_reports()
# generate_violin_basic_stats()
# generate_histogram()
# generate_fbprophet_outlier_plot()
# generate_isolation_forest_outlier_plot(start_subject=3)
