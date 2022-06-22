import numpy as np
import pandas as pd

def load_dataset(subject: int):
    if subject <= 0 or subject > 15:
        print("Subject must be in the range of 1-15.")
        return 
    subject_df = pd.read_csv('../data/' + str(subject) + '.csv', names = ['timestamp', 'x', 'y', 'z', 'activity'])

    return subject_df


