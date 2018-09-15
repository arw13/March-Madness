## Script to match teams to names to use to make a bracket

import numpy as np # linear algebra
NCAATourneySeedsimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import pi

data_dir = '../March-Madness/Final/'
df_pred = pd.read_csv(data_dir + 'MLP_noclipping_0.csv')
df_Teams = pd.read_csv(data_dir + 'Teams.csv')


