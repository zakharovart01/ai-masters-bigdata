from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

#
# Dataset fields
#


#
# Model pipeline
#

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)] + ["day_number"]

numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        #    ('scaler', StandardScaler())
        ])

categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

fields = ["id", "label"] + numeric_features + categorical_features
fields_val = ["id"] + numeric_features + categorical_features

preprocessor = ColumnTransformer(
            transformers=[
                        ('num', numeric_transformer, numeric_features),
                        ('cat', 'drop', categorical_features)
            ]
            )

# Now we have a full prediction pipeline.
model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('logregression', LogisticRegression())
        ])
