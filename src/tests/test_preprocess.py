'''
Date: Sat Jun 23, 2022
Author: Ha Le
This file contains test function for the file preprocess.py.
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import *

def test_load_dataset():
    '''
    Test function for load_dataset fn() in preprocessing.py
    '''
    correct_length = [162501, 138001, 102341, 122201, 
                        160001, 140901, 163001, 138001, 166741, 
                        126801, 104451, 114702, 67651, 116101, 
                        103501]
    generate_len = []
    for i in range(1, 16):
        subject_df = load_dataset(i)
        generate_len.append(len(subject_df))
    assert correct_length == generate_len

def test_mark_activity():
    '''
    Test function for mark_activity fn() in preprocessing.py
    '''
    assert mark_activity(0) == 'w'
    assert mark_activity(1) == 'r'
    assert mark_activity(2) == 'g'
    assert mark_activity(3) == 'b'
    assert mark_activity(4) == 'y'
    assert mark_activity(5) == 'm'
    assert mark_activity(6) == 'c'
    assert mark_activity(7) == 'orange'
    assert mark_activity(10) is None

def test_split_activity():
    '''
    Test function for split_data_by_activity fn() in preprocessing.py
    '''
    correct_lst = [9, 10, 9, 9, 9, 11, 9, 11, 10, 9, 9, 9, 9, 9, 9]
    generate_lst = []
    for i in range(1, 16):
        subject_df = load_dataset(i)
        activity_dfs = split_data_by_activity(subject_df)
        generate_lst.append(len(activity_dfs))
    assert correct_lst == generate_lst