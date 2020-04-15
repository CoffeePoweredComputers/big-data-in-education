from sklearn.tree import DecisionTreeClassifier #Python's default implementation uses CART (Classification and Regression Trees)
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import numpy as np


####################################################################################################
# Question 1
####################################################################################################

df = pd.read_csv('./data/BDEPennW1.csv') 

# Decision trees can handle categorical data but
# target terms should still be encoded in order to create 
# a confusion matrix at a later point. For this, we will 
# create a one-hot-encoding on all string type variables.
df_dummies = pd.get_dummies(df, columns=['SCHOOL','Class','CODER ','Activity'])

# Create an array with the variable you want to predict
y = np.array(df_dummies[["ONTASK"]])

# Remove the predictor variable from the list of A
x = df_dummies.drop(columns=['ONTASK'])

# build the decision trees
clf = DecisionTreeClassifier(min_samples_split=10)
clf.fit(x, y)

# Make predictions with your model
predictions = clf.predict(x)

# Compute Cohen's kappa for inter response similarity
kappa = cohen_kappa_score(y, predictions)

#output the results
print("Model kappa with all data included: %.2f " % kappa)

####################################################################################################
# Question 2
####################################################################################################

# Now lets drop STUDENTID from our x variables and rerun the model
x = df_dummies.drop(columns=['ONTASK', 'STUDENTID'])

# build the decision trees
clf = DecisionTreeClassifier(min_samples_split=10)
clf.fit(x, y)

# Make predictions with your model
predictions = clf.predict(x)

# Compute Cohen's kappa for inter response similarity
kappa = cohen_kappa_score(y, predictions)

#output the results
print("Model kappa with STUDENTID not included: %.2f " % kappa)

####################################################################################################
# Question 3
####################################################################################################

# Make the model even more generalizable by excluding overly specific variables
df_dummies = pd.get_dummies(df, columns=['Activity'])
x = df_dummies.drop(columns=['ONTASK', 'SCHOOL', 'Class', 'CODER ', 'UNIQUEID ', 'STUDENTID'])

# build the decision trees
clf = DecisionTreeClassifier(min_samples_split=10)
clf.fit(x, y)

# Make predictions with your model
predictions = clf.predict(x)

# Compute Cohen's kappa for inter response similarity
kappa = cohen_kappa_score(y, predictions)

#output the results
print("Model kappa with STUDENTID, UNIQUEID, CODER, Class, SCHOOL, ONSTASK not included: %.2f " % kappa)

####################################################################################################
# Question 5
####################################################################################################

# For this question we will be using Gaussian Naive Bayes instead of CART

from sklearn.naive_bayes import GaussianNB

# First train the model
gnb_clf = GaussianNB()
gnb_clf.fit(x,y.ravel())

# Then make the predictions
predictions = gnb_clf.predict(x)

# Compute the Cohen's kappa
kappa = cohen_kappa_score(y, predictions)

# Print it on up
print("model kappa using GaussianNB: %.2f" % kappa)

####################################################################################################
# Question 6
####################################################################################################

from sklearn.ensemble import GradientBoostingClassifier

# Train the model
xgb = GradientBoostingClassifier(learning_rate=0.75, n_estimators=100, min_samples_split=.1, min_samples_leaf=.05, min_weight_fraction_leaf=.05, random_state=5)
xgb.fit(x, y.ravel()) # Here we use the 'ravel' to flatten the array to a 1-dimensional array of labels

# Lets make some predictions
predictions = xgb.predict(x)

# And test those predictions
kappa = cohen_kappa_score(y, predictions)

# Print it up
print("model kappa using XGBClassifier: %.2f" % kappa)

####################################################################################################
# Question 7
####################################################################################################

from sklearn.model_selection import GroupKFold

# split our data 
gkf = GroupKFold(n_splits=10)
gkf.get_n_splits(10)

# Create a list of unique users and their indecies
group_dict = {}
groups = np.array([])

for index, row in df_dummies.iterrows():
    student_id = row['STUDENTID']
    if student_id not in group_dict:
        group_dict[student_id] = index
    groups = np.append(groups, group_dict[student_id])


# train and test all the data
kappa_sum = 0
print("Decision Tree")
for i, data_folds in enumerate(gkf.split(x, y, groups=groups)):

    train = data_folds[0]
    test = data_folds[1]

    # Get the individual fold of the data set
    x_train, x_test = x.loc[train], x.iloc[test]
    y_train, y_test = y[train], y[test]

    # Train this bad boy
    clf = DecisionTreeClassifier(min_samples_split=10)
    clf.fit(x_train, y_train)

    # Predict some stuff
    predictions = clf.predict(x_test)
    kappa = cohen_kappa_score(y_test, predictions)
    kappa_sum += float(kappa)
    print("%d# Fold: kappa=%.2f " % (i, kappa))

print("Avg Kappa: %0.2f" % (kappa_sum / 10))

####################################################################################################
# Question 8
####################################################################################################

# train and test all the data
kappa_sum = 0
print("Naive Bayes")
for i, data_folds in enumerate(gkf.split(x, y, groups=groups)):

    train = data_folds[0]
    test = data_folds[1]

    # Get the individual fold of the data set
    x_train, x_test = x.loc[train], x.iloc[test]
    y_train, y_test = y[train], y[test]
    
    # Train this bad boy
    clf = GaussianNB()
    clf.fit(x_train, y_train.ravel())

    # Predict some stuff
    predictions = clf.predict(x_test)
    kappa = cohen_kappa_score(y_test, predictions)
    kappa_sum += float(kappa)
    print("%d# Fold: kappa=%.2f " % (i, kappa))

print("Avg Kappa: %0.2f" % (kappa_sum / 10))


####################################################################################################
# Question 9
####################################################################################################

# train and test all the data
kappa_sum = 0
print("Extreme Gradient Boosting")
for i, data_folds in enumerate(gkf.split(x, y, groups=groups)):

    train = data_folds[0]
    test = data_folds[1]

    # Get the individual fold of the data set
    x_train, x_test = x.loc[train], x.iloc[test]
    y_train, y_test = y[train], y[test]
    
    # Train this bad boy
    xgb = GradientBoostingClassifier(learning_rate=0.75, n_estimators=100, min_samples_split=.1, min_samples_leaf=.05, min_weight_fraction_leaf=.05, random_state=5)
    xgb.fit(x_train, y_train.ravel()) # Here we use the 'ravel' to flatten the array to a 1-dimensional array of labels

    # Predict some stuff
    predictions = xgb.predict(x_test)
    kappa = cohen_kappa_score(y_test, predictions)
    kappa_sum += float(kappa)
    print("%d# Fold: kappa=%.2f " % (i, kappa))

print("Avg Kappa: %0.2f" % (kappa_sum / 10))



