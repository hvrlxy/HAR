'''
Date: Sat Jun 23, 2022
Author: Ha Le
This file contains test function for the file split_data.py.
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_prep.split_data import *

def test_summarize_subject_activity():
    result_dict = {0: (1, 'workingPC'),
                    1: (2, 'stand'),
                    2: (3, 'stand+walk+stairs'),
                    3: (4, 'walk'),
                    4: (3, 'stand+walk+stairs'),
                    5: (5, 'stairs'),
                    6: (3, 'stand+walk+stairs'),
                    7: (6, 'walk+talk'),
                    8: (7, 'talk')}
    assert summarize_subject_activities(1) == result_dict