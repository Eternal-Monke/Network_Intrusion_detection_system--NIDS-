#!/usr/bin/env python3

####################################
########### IMPORTS HERE ###########
####################################

import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler



##############################################
########### PREPROCESSING FUNCTION ###########
##############################################

# encodes elements like tcp and udp into numbers
# scales features to be in a range of [0,1]
# trains encoders/scalers on training data and reuses them on test data
def preprocess_data(df, encoders=None, scaler=None, fit=True):
    df = df.copy()
    
    drop_id_cols = ['id']
    df = df.drop(columns=drop_id_cols)

    categorical_cols = ['proto', 'service', 'state']
    if encoders is None:
        encoders = {}

    for col in categorical_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col])

    feature_cols = df.drop(columns=['label', 'attack_cat']).columns

    if fit:
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df, encoders, scaler





######################################
########### LOADS THE DATA ###########
######################################

# lists all of the column names we will be using for the data
columns = [
    'id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
    'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt',
    'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
    'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src',
    'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
    'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat', 'label'
]

# loads both the training and test sets and assigns the above columns to the data
train_df = pd.read_csv('data/UNSW_NB15_training-set.csv', names=columns, skiprows=1)
test_df = pd.read_csv('data/UNSW_NB15_testing-set.csv', names=columns, skiprows=1)


################################################
########### DISPLAYS SOME BASIC INFO ###########
################################################

# this shows the number of rows and columns and a the first couple of data entries
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Training data head (first couple of entires):")
print(train_df.head())


##########################################################
########### SUMMARY OF THE DATA IN LIST FORMAT ###########
##########################################################

print("\nClass distribution in training set:")
print(train_df['attack_cat'].value_counts())

print("\nClass distribution in test set:")
print(test_df['attack_cat'].value_counts())

# visualise the class distribution
plt.figure(figsize=(10, 4))
sns.countplot(data=train_df, x='attack_cat', order=train_df['attack_cat'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Class Distribution in Training Set')
plt.tight_layout()
plt.show()


##################################################################
########### PREPROCESS BOTH THE TRAINING AND TEST DATA ###########
##################################################################

train_df, encoders, scaler = preprocess_data(train_df, fit=True)
test_df, _, _ = preprocess_data(test_df, encoders=encoders, scaler=scaler, fit=False)

# shows the head of the data after preprocessing
print("\nAfter preprocessing:")
print(train_df[['proto', 'service', 'state']].head())




########################################################
########### SETTING UP THE DATA FOR TRAINING ###########
########################################################


# Drop unused columns to get feature matrices
# x_train and x_test are the input (without the info if it is an attack or it is safe)
# y_train and y_test are the actual attacks types
drop_cols = ['label', 'attack_cat']
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['attack_cat']

X_test = test_df.drop(columns=drop_cols)
y_test = test_df['attack_cat']


##########################################
########### TRAINING THE MODEL ###########
##########################################


# trains the model on the train data -> x_train being the large data and y_train being the "answer"
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# predict possible results on the test data set using what we trained
y_pred = clf.predict(X_test)




#########################################
########### RESULT EVALUATION ###########
#########################################


# evaluation of results
# in form of:
#   precision -> (how many predicted outputs were actually correct)
#   recall -> (how many actual outputs were actually correct)
#   F1-score -> (the mean of the precision and recall values)
#   support -> the number of samples of that label that we had

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, zero_division=0))

# plot the confusion matrix heatmap of the results
print("\n=== Confusion Matrix ===")
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), cmap='Blues', annot=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# shows which features are the most important for the model in figuring out the results
importances = clf.feature_importances_
features = X_train.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices][:10], y=features[indices][:10], palette='viridis')
plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


######################################
########### SAVING RESULTS ###########
######################################

# saves the trained model information into a model file
# saves the model, encoder, and scaler
joblib.dump(clf, 'model/unsw/random_forest_nsl_kdd.pkl')
joblib.dump(encoders, 'model/unsw/label_encoders.pkl')
joblib.dump(scaler, 'model/unsw/scaler.pkl')