'''
Date: Sat Jun 23, 2022
Author: Ha Le
This file contains function to perform EDA.
'''
import os
import pandas as pd
# import matplotlib.pyplot as plt
from tabulate import tabulate
from preprocess import load_dataset
from preprocess import split_data_by_activity

# Global variables
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

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
    report_file = open(CURRENT_PATH + f"/../reports/tables/{file_name}", "w+")
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
    report_file.close()
#test
# subject_df = load_dataset(1)
# activity_dfs = split_data_by_activity(subject_df)
# for activity_df in activity_dfs:
#     print(calculate_sequence_length(activity_df))
generate_sequence_length_report()
