# base code with Data manipulation credits to John Ellis
# coding: utf-8
# I have enhanced base line version with xgboost model

# In[23]:

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split


# ##### Load the data sets, fill missing values and convert Upc and Fineline to strings

# In[24]:

train = pd.read_csv('/Users/chidam/data_sci/data/Walmart/train-3.csv', header=0, index_col=None)
train.loc[train['Upc'].isnull(), 'Upc'] = 0
train.loc[train['DepartmentDescription'].isnull(), 'DepartmentDescription'] = 'UNKNOWN'
train.loc[train['FinelineNumber'].isnull(), 'FinelineNumber'] = 0
train['Upc'] = train['Upc'].astype(str).map(lambda x: x.split('.')[0]) # Theres better ways to get clean strings, easy fix for now
train['FinelineNumber'] = train['FinelineNumber'].astype(str).map(lambda x: x.split('.')[0])

test = pd.read_csv('/Users/chidam/data_sci/data/Walmart/test.csv', header=0, index_col=None)
test.loc[test['Upc'].isnull(), 'Upc'] = 0
test.loc[test['DepartmentDescription'].isnull(), 'DepartmentDescription'] = 'UNKNOWN'
test.loc[test['FinelineNumber'].isnull(), 'FinelineNumber'] = 0
test['Upc'] = test['Upc'].astype(str).map(lambda x: x.split('.')[0])
test['FinelineNumber'] = test['FinelineNumber'].astype(str).map(lambda x: x.split('.')[0])


# In[46]:

sample = pd.read_csv('/Users/chidam/data_sci/data/Walmart/sample_submission-2.csv')
columns=sample.columns[1:]


# ##### Using Pandas Crosstab, rotated out each department as an individial column. These '1''s and '0''s get summed to the VisitNumber level when the groupby happens soon. If you're curious, crosstabbing finelinenumber and upc code into the feature set has a very small impact to anything, but was quite expensive to create, so I took that out.

# In[25]:

departments_train = pd.crosstab(train.index, [train.DepartmentDescription])
departments_test = pd.crosstab(test.index, [test.DepartmentDescription])


# In[26]:

trainDeptDF = train.join(departments_train)
testDeptDF = test.join(departments_test)


# ##### Dropping the columns that will get merged back in from the other data set

# In[27]:

trainDeptDF = trainDeptDF.drop(['Weekday', 'Upc', 'DepartmentDescription', 'FinelineNumber', 'ScanCount'], axis=1)
testDeptDF = testDeptDF.drop(['Weekday', 'Upc', 'DepartmentDescription', 'FinelineNumber', 'ScanCount'], axis=1)


# ##### Group and sum the binary matrix from the department crosstab

# In[28]:

trainDeptDF = trainDeptDF.groupby(['TripType', 'VisitNumber']).sum()
testDeptDF = testDeptDF.groupby(['VisitNumber']).sum()

#print testDeptDF.head()
#print testDeptDF.info()
# Drop columns that will already exist in merged dataframe
trainDeptDF = pd.DataFrame(trainDeptDF.to_records())
trainDeptDF = trainDeptDF.drop(['TripType', 'VisitNumber'], axis=1)

testDeptDF = pd.DataFrame(testDeptDF.to_records())
testDeptDF = testDeptDF.drop(['VisitNumber'], axis=1)
#print testDeptDF.head()


# ##### These are the main groupings to get itemcounts and most common fineline, upc and department per VisitNumber

# In[29]:

trainVisitGroup = train.groupby(['TripType','VisitNumber'])
testVisitGroup = test.groupby(['VisitNumber'])
trainVisitGroup.head()


# In[30]:

aggregations = {
    'Weekday': {
        'Day': 'max'
    },
    'ScanCount': {
        'TotalItems': 'sum',
        'UniqueItems': 'count'
    },
    'DepartmentDescription': {
        'MostCommonDept': 'max'
    },
    'FinelineNumber': {
        'MostCommonFineline': 'max'
    },
    'Upc': {
        'MostCommonProduct': 'max'
    }
}


# In[31]:

trainDF = trainVisitGroup.agg(aggregations)
testDF = testVisitGroup.agg(aggregations)
trainDF.head()


# ##### Correct and flatten the column names from the MultiIndex

# In[32]:

trainDF.columns = [','.join(col).strip().split(',')[1] for col in trainDF.columns.values]
testDF.columns = [','.join(col).strip().split(',')[1] for col in testDF.columns.values]


# ##### Get a pure flat dataframe from the groupby dataframe by recasting it as DF from to_records()

# In[33]:

trainDF = pd.DataFrame(trainDF.to_records())
testDF = pd.DataFrame(testDF.to_records())
trainDF.head()


# ##### Now join our department features DF into the main testing and training DF's

# In[34]:

trainDF = trainDF.join(trainDeptDF)
testDF = testDF.join(testDeptDF)
trainDF.head()


# In[35]:

trainDF.info()


# ##### Convert the text features to numerics. Yes, there is room for code consolidation here... this was quick ;)

# In[36]:

# Convert text features to numbers
Days = list(enumerate(np.unique(trainDF.Day)))
Day_dict = { name : i for i, name in Days }
trainDF.Day = trainDF.Day.map(lambda x: Day_dict[x]).astype(int)

Departments = list(enumerate(np.unique(trainDF.MostCommonDept)))
MCD_dict = { name : i for i, name in Departments }
trainDF.MostCommonDept = trainDF.MostCommonDept.map(lambda x: MCD_dict[x]).astype(int)

Finelines = list(enumerate(np.unique(trainDF.MostCommonFineline)))
MCF_dict = { name : i for i, name in Finelines } 
trainDF.MostCommonFineline = trainDF.MostCommonFineline.map(lambda x: MCF_dict[x]).astype(int)

Products = list(enumerate(np.unique(trainDF.MostCommonProduct)))
MCP_dict = { name : i for i, name in Products } 
trainDF.MostCommonProduct = trainDF.MostCommonProduct.map(lambda x: MCP_dict[x]).astype(int)

Days = list(enumerate(np.unique(testDF.Day)))
Day_dict = { name : i for i, name in Days }
testDF.Day = testDF.Day.map(lambda x: Day_dict[x]).astype(int)

Departments = list(enumerate(np.unique(testDF.MostCommonDept)))
MCD_dict = { name : i for i, name in Departments }
testDF.MostCommonDept = testDF.MostCommonDept.map(lambda x: MCD_dict[x]).astype(int)

Finelines = list(enumerate(np.unique(testDF.MostCommonFineline)))
MCF_dict = { name : i for i, name in Finelines } 
testDF.MostCommonFineline = testDF.MostCommonFineline.map(lambda x: MCF_dict[x]).astype(int)

Products = list(enumerate(np.unique(testDF.MostCommonProduct)))
MCP_dict = { name : i for i, name in Products } 
testDF.MostCommonProduct = testDF.MostCommonProduct.map(lambda x: MCP_dict[x]).astype(int)


# ##### The UniqueItems feature seemed to hurt the model, so I removed it. Also the train set had a 'beauty' category that did not exist in the test set

# In[37]:

# Drop unwanted columns
trainDF = trainDF.drop(['UniqueItems', 'HEALTH AND BEAUTY AIDS'], axis=1) # This department doesn't exist in the test data
testDF = testDF.drop(['UniqueItems'], axis=1)


# ##### Implemeted feature sclaling, but it did not positively affect the score

# In[38]:

# # Feature Scaling - Did not seem to help the score at all

# newDF = pd.DataFrame(newDF.to_records())

# def scale_feature(featureName):
#         newDF[featureName] = (newDF[featureName] - newDF[featureName].min()) / (newDF[featureName].max() - newDF[featureName].min())
        
# scale_feature('MostCommonFineline')
# scale_feature('MostCommonProduct')


# ##### You can use this section below to run models against the training set and check accuracy against the held-out data

# In[39]:

# # Create the Random Forest Model on TRAINING DATA ONLY
# train, test = train_test_split(trainDF, test_size=0.25)

# training_set = train.iloc[0::, 2::].values
# target_set = train[['TripType']].values
# testing_set = test.iloc[0::, 2::].values
# testing_target_set = test[['TripType']].values

# forest = RandomForestClassifier(n_estimators=200, min_samples_split=10, random_state=415)
# forest = forest.fit(training_set, target_set)
# print forest.score(testing_set, testing_target_set)
# classes = forest.classes_
# prob = forest.predict_proba(testing_set)


# In[93]:

def runXGB(train_X, train_y, test_X):
    params = {}
    params["objective"] = "multi:softprob"
    params["eta"] = 0.1
    #params["min_child_weight"] = 1
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.85
    params["silent"] = 1
    params["max_depth"] = 5
    #params["max_delta_step"]=2
    params["seed"] = 0
    params['eval_metric'] = "mlogloss"
    params['num_class'] = 38
    params['gamma']=1



    plst = list(params.items())
    num_rounds = 300
    
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    xgtest = xgb.DMatrix(test_X)
    watchlist = [(xgtrain,'train')]
    evals_result = {}
    
    
    model = xgb.train(plst, xgtrain, num_rounds,
            watchlist,evals_result=evals_result,early_stopping_rounds=10)
    
    
    #model = xgb.train(plst, xgtrain, num_rounds)
    pred_test_y = model.predict(xgtest)
    return pred_test_y


# In[94]:

from sklearn import ensemble, feature_extraction, preprocessing
training_set = trainDF.iloc[0::, 2::].values
#target_set = trainDF[['TripType']].values
lbl_enc = preprocessing.LabelEncoder()
labels = trainDF.TripType.values
labels = lbl_enc.fit_transform(labels)


testing_set = testDF.iloc[0::, 1::].values

train_X = np.array(training_set).astype('float')
test_X = np.array(testing_set).astype('float')
print train_X.shape, test_X.shape
model_classes = []


# In[95]:

import xgboost as xgb
print "Running model.."	
pred = runXGB(train_X, labels, test_X)
prob=pred




#forest = RandomForestClassifier(n_estimators=1000, min_samples_split=10, random_state=415)
#forest = forest.fit(training_set, target_set)
#classes = forest.classes_
#prob = forest.predict_proba(testing_set)



# ##### All code below runs the model and bulids the file for submission

# In[89]:

#training_set = trainDF.iloc[0::, 2::].values
#target_set = trainDF[['TripType']].values
#testing_set = testDF.iloc[0::, 1::].values

#forest = RandomForestClassifier(n_estimators=1000, min_samples_split=10, random_state=415)
#forest = forest.fit(training_set, target_set)
#classes = forest.classes_
#prob = forest.predict_proba(testing_set)


# In[90]:

#print classes


# ##### Create the submission CSV file

# In[96]:

df = pd.DataFrame(pred, columns=sample.columns[1:])
VisitNumberDF = pd.DataFrame(testDF.iloc[0::,0:1].values, columns=['VisitNumber'])
ProbabilitiesDF = pd.DataFrame(pred, columns=sample.columns[1:])
ProbabilitiesDF.columns = ProbabilitiesDF.columns.map(lambda x: str(x))
SubmissionDF = VisitNumberDF.join(ProbabilitiesDF)


# In[97]:

SubmissionDF.to_csv('/Users/chidam/data_sci/data/Walmart/walmart_new_xgboost_submission.csv', index=False)


# In[ ]:



