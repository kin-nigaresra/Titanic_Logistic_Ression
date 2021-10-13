
#### TITANIC ####

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from helpers.data_prep import *
from helpers.eda import *


df = pd.read_csv('datasets/titanic.csv')
df.head()


# Data Pre-Processing and Furute Engineering;

def titanic_data_prep(dataframe):

    # FEATURE ENGINEERING
    dataframe["NEW_CABIN_BOOL"] = dataframe["Cabin"].isnull().astype('int')
    dataframe["NEW_NAME_COUNT"] = dataframe["Name"].str.len()
    dataframe["NEW_NAME_WORD_COUNT"] = dataframe["Name"].apply(lambda x: len(str(x).split(" ")))
    dataframe["NEW_NAME_DR"] = dataframe["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    dataframe['NEW_TITLE'] = dataframe.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataframe["NEW_FAMILY_SIZE"] = dataframe["SibSp"] + dataframe["Parch"] + 1
    dataframe["NEW_AGE_PCLASS"] = dataframe["Age"] * dataframe["Pclass"]

    dataframe.loc[((dataframe['SibSp'] + dataframe['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
    dataframe.loc[((dataframe['SibSp'] + dataframe['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

    dataframe.loc[(dataframe['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['Age'] >= 18) & (dataframe['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'

    dataframe.loc[(dataframe['Sex'] == 'male') & (dataframe['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['Sex'] == 'male') & ((dataframe['Age'] > 21) & (dataframe['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['Sex'] == 'male') & (dataframe['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['Sex'] == 'female') & (dataframe['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['Sex'] == 'female') & ((dataframe['Age'] > 21) & (dataframe['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['Sex'] == 'female') & (dataframe['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
    dataframe.columns = [col.upper() for col in dataframe.columns]

    # AYKIRI GOZLEM
    num_cols = [col for col in dataframe.columns if len(dataframe[col].unique()) > 20
                and dataframe[col].dtypes != 'O'
                and col not in "PASSENGERID"]

    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    # for col in num_cols:
    #    print(col, check_outlier(df, col))
    # print(check_df(df))


 # EKSIK GOZLEM
dataframe.drop(["TICKET", "NAME", "CABIN"], inplace=True, axis=1)
dataframe["AGE"] = dataframe["AGE"].fillna(dataframe.groupby("NEW_TITLE")["AGE"].transform("median"))

dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
dataframe.loc[(dataframe['SEX'] == 'male') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
dataframe.loc[(dataframe['SEX'] == 'female') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

    # LABEL ENCODING #


binary_cols = [col for col in dataframe.columns if len(dataframe[col].unique()) == 2 and dataframe[col].dtypes == 'O']

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    dataframe = rare_encoder(dataframe, 0.01)

    ohe_cols = [col for col in dataframe.columns if 10 >= len(dataframe[col].unique()) > 2]
    dataframe = one_hot_encoder(dataframe, ohe_cols)

    return dataframe


df_prep = titanic_data_prep(df)
df_prep.head()

# Logistic Regression Model;

y = df["SURVIVED"]
X = df.drop(["SURVIVED"], axis=1)

log_model = LogisticRegression().fit(X, y)

log_model.intercept_
log_model.coef_


# 10 katlı çapraz doğrulama yöntemi;
cv_results = cross_validate(log_model,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# Accuracy: 0.8249063670411985


cv_results['test_precision'].mean()
# Precision: 0.7925152411317675

cv_results['test_recall'].mean()
# Recall:  0.7395798319327731

cv_results['test_f1'].mean()
# F1-score: 0.7614244440185146

cv_results['test_roc_auc'].mean()
# AUC: 0.8671542879778175


random_user = X.sample(1, random_state=42)
random_user
log_model.predict(random_user)