import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras import regularizers, optimizers
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from keras import backend as K
from sklearn.metrics import log_loss

from subprocess import check_output
print(check_output(["ls", "../Training"]).decode("utf8"))

# Load data
data_dir = '../Training/'
df_seeds = pd.read_csv(data_dir + 'WNCAATourneySeeds.csv')
df_tour =  pd.read_csv(data_dir + 'WNCAATourneyCompactResults.csv')

# extract seeds as integers
def seed_to_int(seed):
    #Get just the digits from the seeding. Return as int
    s_int = int(seed[1:3])
    return s_int
df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label
df_seeds.head()

# Drop extra cols
df_tour.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)

# Merge seeds and concatenate
df_winseeds = df_seeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
df_lossseeds = df_seeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})
df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'LTeamID'])
df_concat['SeedDiff'] = df_concat.WSeed - df_concat.LSeed
df_concat1 = df_concat[df_concat.Season<2014]
df_concat2 =  df_concat[df_concat.Season>=2014] 
print(df_concat2.shape)

# Win and loss dfs
df_wins = pd.DataFrame()
df_wins['SeedDiff'] = df_concat1['SeedDiff']
df_wins['Result'] = 1

df_losses = pd.DataFrame()
df_losses['SeedDiff'] = -df_concat1['SeedDiff']
df_losses['Result'] = 0

df_train = pd.concat((df_wins, df_losses))
df_train.head()

# Training set
df_winTest= pd.DataFrame()
df_winTest['SeedDiff'] = df_concat2['SeedDiff']
df_winTest['Result'] = 1

df_lossesTest = pd.DataFrame()
df_lossesTest['SeedDiff'] = -df_concat2['SeedDiff']
df_lossesTest['Result'] = 0

df_test = pd.concat((df_winTest, df_lossesTest))
print(df_train.shape)

# Select training and test data
X_train = df_train.iloc[:,0].values.reshape(-1,1)
y_train = df_train.Result.values
xDim = np.shape(X_train)[1]
X_train, y_train = shuffle(X_train, y_train)
X_test = df_test.iloc[:,0].values.reshape(-1,1)
y_test = df_test.Result.values


# Train the model
dropRate = 0.3
numBatch = 20
numEpoch = 120
learningRate = 1e-4

# MLP model
MLP = Sequential()
MLP.name = 'MLP'
MLP.add(Dense(5, input_dim=xDim, kernel_initializer='glorot_normal',activation = 'tanh'))
MLP.add(Dropout(dropRate))
MLP.add(Dense(200,kernel_initializer='glorot_normal',activation = 'tanh'))
MLP.add(Dropout(dropRate))
MLP.add(Dense(200, kernel_initializer='glorot_normal',activation = 'tanh'))
MLP.add(Dropout(dropRate))
MLP.add(Dense(1, kernel_initializer='glorot_normal', activation='sigmoid'))

# Compile model
adam = optimizers.Adam(lr=learningRate, amsgrad=True)
MLP.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# Fit model
MLP.fit(X_train, y_train, epochs=numEpoch, batch_size=numBatch, verbose=1)

# Test model
train_res = MLP.evaluate(X_test, y_test)
print(train_res)

# Make the sample
df_sample_sub = pd.read_csv(data_dir + 'WSampleSubmissionStage1.csv')
n_test_games = len(df_sample_sub)

def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))


X_test = np.zeros(shape=(n_test_games, 1))
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
    t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]
    diff_seed = t1_seed - t2_seed
    X_test[ii, 0] = diff_seed

# Make Predictions 
preds = MLP.predict_proba(X_test)[:,0]

sample_sub = pd.read_csv(data_dir + 'SampleSubmissionStage2.csv')
df_seeds = pd.read_csv(data_dir + 'NCAATourneySeeds.csv')
preds = np.clip(preds, 0.05, 0.95)
df_sample_sub.Pred = clipped_preds
df_sample_sub.head() 

filename = 'MLP'
save_dir = '../Training/'
c=0
ext = '.csv'
df_sample_sub.to_csv(save_dir+filename+ext, index=False)
