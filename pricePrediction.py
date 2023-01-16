import numpy as np
import pandas as pd


housing = pd.read_csv("data.csv")


#Normal sampling
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)

print(f"Rows in train set: {len(train_set)}\nRows in test set:{len(test_set)}\n")


#stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# print("CHAS in train set")
# strat_train_set['CHAS'].value_counts()
# print("CHAS in test set")
# strat_test_set['CHAS'].value_counts()

# Separating attributes and label
housing = strat_train_set.drop("MEDV", axis = 1)
housing_labels = strat_train_set["MEDV"].copy()


#Creating pipeline to deal with missing values and also make the scale of attributes same

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

housing_num_tr = my_pipeline.fit_transform(housing)

#Using RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
print(model.predict(prepared_data))

print(list(some_labels))


#Evaluating using Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr,  housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

print(rmse_scores)


#Saving the model
from  joblib import dump, load
dump(model, 'RealEstate.joblib')
