'''
Date: Sat Jun 23, 2022
Author: Ha Le
This file contains function to perform EDA.
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from preprocess import load_dataset
from preprocess import split_data_by_activity

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

def generate_sequence_length_report(file_name: str = "sequence_length_report.txt"):
    '''
    Generates a report of the sequence length of each activity.
    '''
    with open(CURRENT_PATH + f"/../reports/tables/{file_name}",
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
    reports.drop(columns=['timestamp', 'activity'], inplace=True)
    return reports.T

def generate_basic_stats_reports():
    '''
    Generates a report of the basic statistics of each activity.
    '''
    for subject in range(1, 16):
        # load the dataset
        with open(CURRENT_PATH + f"/../reports/tables/basic_stats/basic_stats_report_{subject}.txt",
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

def generate_violin_basic_stats(file_name: str = "violin_basic_stats"):
    '''
    Generate the violin plot to compare different activities.
    '''
    for i in range(1, 16):
        # load the data
        report_file = CURRENT_PATH + f"/../reports/imgs/basic_stats/{file_name}_{i}.png"
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
#test
# subject_df = load_dataset(1)
# activity_dfs = split_data_by_activity(subject_df)
# activity_dfs = merge_activity_sequence(activity_dfs)
# print(get_activity_stats(activity_dfs[3]))
# print(activity_dfs)
# for activity_df in activity_dfs:
#     print(calculate_sequence_length(activity_df))
# generate_sequence_length_report()
# generate_basic_stats_reports()
# generate_violin_basic_stats()
