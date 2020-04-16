import math
import pandas as pd
import numpy as np

from scipy.stats import pearsonr

df_classifier = pd.read_csv('./data/classifier-data.csv')
df_regressor = pd.read_csv('./data/regressor-data.csv')

####################################################################################################
# Question 1
####################################################################################################
pearson_df = df_regressor[['data', 'predicted (model)']]
corr = pearson_df.corr()
print('Pearsons r of Data and Prediction:\n', corr, '\n\n')

####################################################################################################
# Question 2
####################################################################################################
def rmse(data, predictions):

    square_sum = 0
    for datum, prediction in zip(data, predictions):
        square_sum += (prediction - datum) ** 2
    square_avg = square_sum / len(data)
    return math.sqrt(square_avg)
        

x, y = df_regressor[['data']].values.ravel(), df_regressor[['predicted (model)']].values.ravel()
rmse_val = rmse(x, y)
print("Regressor RMSE: %.3f\n" % rmse_val)

####################################################################################################
# Question 3
####################################################################################################
def mae(data, predictions):
    abs_sum = 0
    for datum, prediction in zip(data, predictions):
        abs_sum += abs(prediction - datum) 
    abs_avg = abs_sum / len(data)
    return abs_avg
        

mae_val = mae(x, y)
print("Regressor MAE: %.3f\n" % mae_val)

####################################################################################################
# Question 4
####################################################################################################
def get_accuracy(data, predictions, threshold):
    TP = 0
    TN = 0
    for datum, prediction in zip(data, predictions):
        TP += int( (prediction > threshold) and (datum == "Y") ) # Count up our true positives
        TN += int( (prediction < threshold) and (datum == "N") ) # Count up our true negativse
    return (TP + TN) / len(data)

x, y = df_classifier[['Data']].values.ravel(), df_classifier[['Predicted (Model)']].values.ravel()
acc = get_accuracy(x, y, 0.5)
print("Model Accuracy (thresh = 0.5): %f" % acc)

####################################################################################################
# Question 5
####################################################################################################
data_counts = df_classifier['Data'].value_counts()
percent_max = data_counts[data_counts.idxmax()] / sum(data_counts)
print(percent_max)

####################################################################################################
# Question 6
####################################################################################################
def get_kappa(data, predictions, threshold):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # Get the accuracy, type 1 and type 2 error
    for datum, prediction in zip(data, predictions):
        TP += int( (prediction > threshold) and (datum == "Y") ) # Count up our true positives
        TN += int( (prediction < threshold) and (datum == "N") ) # Count up our true negativse
        FP += int( (prediction > threshold) and (datum != "Y") ) # Count up our false positives
        FN += int( (prediction < threshold) and (datum != "N") ) # Count up our false negativse

    # Calculate the observed proportions agreement (p_0)
    p_o = (TP + TN) / len(data)

    # Calculate the probability of guessing yes and no at random
    p_yes = ((TP + FP)/len(data)) * ((TP + FN) / len(data))
    p_no = ((FN + TN)/len(data)) * ((FP + TN) / len(data))

    # Calculate the overall agreement probablity 
    p_e = p_yes + p_no

    return (p_o - p_e) / (1 - p_e)

x, y = df_classifier[['Data']].values.ravel(), df_classifier[['Predicted (Model)']].values.ravel()
kappa = get_kappa(x, y, 0.5)
print("Model kappa (thresh = 0.5): %f" % kappa)


####################################################################################################
# Question 7/8
####################################################################################################
def get_precision_and_recall(data, predictions, threshold):
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # Get the accuracy, type 1 and type 2 error
    for datum, prediction in zip(data, predictions):
        TP += int( (prediction > threshold) and (datum == "Y") ) # Count up our true positives
        TN += int( (prediction < threshold) and (datum == "N") ) # Count up our true negativse
        FP += int( (prediction > threshold) and (datum != "Y") ) # Count up our false positives
        FN += int( (prediction < threshold) and (datum != "N") ) # Count up our false negativse

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return precision, recall

x, y = df_classifier[['Data']].values.ravel(), df_classifier[['Predicted (Model)']].values.ravel()
precision, recall = get_precision_and_recall(x, y, 0.5)
print("Precision: %f\nRecall: %f\n" % (precision, recall))

####################################################################################################
# Question 11
####################################################################################################

from sklearn import metrics

df_classifier['Data'] = df_classifier['Data'].map({'Y': 1, 'N': 0})
x, y = df_classifier[['Data']].values.ravel(), df_classifier[['Predicted (Model)']].values.ravel()
fpr, tpr, thresholds = metrics.roc_curve(x, y)
print("AUC (A'): ",metrics.auc(fpr, tpr))

