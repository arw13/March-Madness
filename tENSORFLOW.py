
# coding: utf-8

# # Tensorflow Model

# region Import
import tensorflow as tf
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from time import localtime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# endregion

#region Load Data
data_dir = '../NCAA/'
data_file = data_dir + 'MarchMadnessFeatures.csv'
df_features = pd.read_csv(data_file)
df_test = pd.read_csv(data_dir+'MarchMadnessTest.csv')

X = df_features.iloc[:,1:]
xDim = np.shape(X)[1]
X= X.values.reshape(-1,xDim)
Y = df_features.Result.values
Y = np.array([Y, -(Y-1)]).T  # The model currently needs one column for each class
# import pdb; pdb.set_trace()
X_test = df_test.iloc[:,1:]
xDimTest = np.shape(X_test)[1]
X_test = X_test.values.reshape(-1,xDimTest)
Y_test = df_test.Result.values
Y_test = np.array([Y_test, -(Y_test-1)]).T  # The model currently needs one column for each class

scaler = StandardScaler()
# X = scaler.fit_transform(X)
# X_test  = scaler.transform(X_test)


print('Feature vector dimension is: %.2f' % xDim)

# endregion


# Hyperparameter
learning_rate = 3e-4
training_epochs = 800#120
batch_size = 100 #25
display_step = 10

# Network Parameters
n_hidden_1 = 50
n_hidden_2 = 100
n_hidden_3 = 100
n_hidden_4 = 200
n_input = xDim # Number of feature
n_classes = 2

# region Set Up Network


# Assign placeholders for unknown values
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    drop_out = tf.nn.dropout(layer_1, keep_prob)
    # Hidden layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    # # Hidden layer with sigmoif activation
    # layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # layer_3 = tf.nn.sigmoid(layer_3)
    # # Hidden layer with sigmoif activation
    # layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    # layer_4 = tf.nn.sigmoid(layer_4)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Initializing the variables
init = tf.global_variables_initializer()
# endregion

# region Launch the graph
print("Training Model")

# Model save
saver = tf.train.Saver()

logloss = []
acc = []
accTest = []
loglossTest =[]
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X)/batch_size)
        X_batches = np.array_split(X, total_batch)
        Y_batches = np.array_split(Y, total_batch)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_step, cost], feed_dict={x: batch_x,
                                                          y: batch_y,
                                                          keep_prob: 0.5})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            # Training Cost
            logloss.append(avg_cost)
            # Define eval of accuracy
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            # Calculat train case accuracy
            acc.append(accuracy.eval({x: X, y: Y, keep_prob:1.0}))
            # Calulate test case accuracy
            accTest.append(accuracy.eval({x: X_test, y: Y_test, keep_prob:1.0}))
            # Find test cost
            _,cTest = sess.run([pred, cost], feed_dict={x: X_test, y:Y_test, keep_prob:1.0})
            loglossTest.append(cTest)
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}, costT={:.9f} ".format(avg_cost, cTest))

    print("Training Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Test Accuracy:", accuracy.eval({x: X_test, y: Y_test}))

    # Save Model
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)
    # global result
    # result = tf.argmax(pred, 1).eval({x: X_test, y: Y_test})
#endregion

# region Plot
plt.subplot(211)
costplt = plt.plot(logloss)
costTestplt = plt.plot(loglossTest)
plt.ylim(ymax=0.7)
plt.subplot(212)
accplt = plt.plot(acc)
accTestplt = plt.plot(accTest)
plt.show()


# tf.reset_default_graph
# endregion

# region Load data for predictions
print('Loading data for submission test')

data_dir = '../NCAA/Data/'
''' Make output predictions '''
df_sample_sub = pd.read_csv(data_dir + 'SampleSubmissionStage1.csv')
n_test_games = len(df_sample_sub)


def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))


# Load advanced stats and seeding to dataframe
df_seeds = pd.read_csv(data_dir + 'NCAATourneySeeds.csv')
df_adv = pd.read_csv(data_dir + 'MarchMadnessAdvStats.csv')
df_Xpred = pd.read_csv(data_dir + 'submissionData.csv')

xDim = np.shape(df_Xpred)[1]
X_pred = df_Xpred.values.reshape(-1,xDim)
# X_pred = scaler.transform(X_pred)

# Old Submission Load code
# def seed_to_int(seed):
#     '''Get just the digits from the seeding. Return as int'''
#     s_int = int(seed[1:3])
#     return s_int
#
#
# # Make the seeding an integer
# df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
# df_seeds.drop(columns=['Seed'], inplace=True) # This is the string label
# df_seeds.head()
#
#
# T1_seed = []
# T1_adv = []
# T2_adv = []
# T2_seed = []
# for ii, row in df_sample_sub.iterrows():
#     year, t1, t2 = get_year_t1_t2(row.ID)
#     t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
#     t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]
#     t1_adv = df_adv[(df_adv.TeamID == t1) & (df_adv.Season == year)].values[0]
#     t2_adv = df_adv[(df_adv.TeamID == t2) & (df_adv.Season == year)].values[0]
#     T1_seed.append(t1_seed)
#     T1_adv.append(t1_adv)
#     T2_seed.append(t2_seed)
#     T2_adv.append(t2_adv)
#
# T1_adv = [row[2:] for row in T1_adv]
# T2_adv = [row[2:] for row in T2_adv]
# T1_seed = np.reshape(T1_seed, [n_test_games,-1]).tolist()
# T2_seed = np.reshape(T2_seed, [n_test_games, -1]).tolist()
# X_pred = np.concatenate((T1_seed, T1_adv, T2_seed, T2_adv), axis=1)
# # endregion

# endregion

# region reload saved model and make predictions
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "/tmp/model.ckpt")
    print("Model restored.")
    print("Making predictions")
    preds = sess.run(pred, feed_dict={x: X_pred, keep_prob:1.0})

clipped_preds = np.clip(preds, 0.05, 0.95)
df_sample_sub.Pred = clipped_preds

filename ='TFmodel'
save_dir = '../NCAA/Submissions/'
c=0
ext = '.csv'
df_sample_sub.to_csv(save_dir+filename+ext, index=False)

#
# if os.path.exists(save_dir+filename+ext):
#     while os.path.exists(save_dir+filename+ext):
#         c+=1
#     filename = filename+'_'+str(c)
#     df_sample_sub.to_csv(save_dir+filename+ext, index=False)
# else:
#     df_sample_sub.to_csv(save_dir+filename+ext, index=False)
#
# endregion
tf.reset_default_graph
