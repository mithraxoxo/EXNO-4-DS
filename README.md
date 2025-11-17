# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
# FEATURE SCALING

import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("/content/bmi.csv")
df.head()
```

<img width="422" height="293" alt="image" src="https://github.com/user-attachments/assets/5e780854-49ca-404c-8cee-49e90d607fd4" />


```
df_null_sum=df.isnull().sum()
df_null_sum
```

<img width="227" height="245" alt="image" src="https://github.com/user-attachments/assets/a1f6255a-95c5-497c-a0b3-faec7a0e0871" />

```
df.dropna()
```

<img width="445" height="509" alt="image" src="https://github.com/user-attachments/assets/96ed6f05-f512-45b5-9af5-438192884836" />

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
# This is typically used in feature scaling,
#particularly max-abs scaling, which is useful
#when you want to scale data to the range [-1, 1]
#while maintaining sparsity (often used with sparse data).
```

<img width="596" height="316" alt="image" src="https://github.com/user-attachments/assets/c6673db6-a487-4d10-afb6-236ac33b029a" />


```
# Standard Scaling
```


```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```

<img width="443" height="233" alt="image" src="https://github.com/user-attachments/assets/00f8ca2f-65bd-4988-8385-a88810858479" />


```
sc=StandardScaler()
```
```
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```

<img width="463" height="417" alt="image" src="https://github.com/user-attachments/assets/db95ac43-b30c-4680-89ae-9e7de6b30576" />

```
#MIN-MAX SCALING:
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

<img width="458" height="420" alt="image" src="https://github.com/user-attachments/assets/dfbb0a33-13b5-43b5-9321-a08d3579c31e" />

```
#MAXIMUM ABSOLUTE SCALING:

from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

<img width="496" height="495" alt="image" src="https://github.com/user-attachments/assets/cce956a5-eda5-46fc-b7fe-113ebe22f9cf" />


```
#ROBUST SCALING

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```


<img width="444" height="239" alt="image" src="https://github.com/user-attachments/assets/025f3eb8-7d76-458a-be87-5d835c2c5737" />


```
#FEATURE SELECTION:
```
```
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```


<img width="454" height="414" alt="image" src="https://github.com/user-attachments/assets/80731cbc-3c8d-4fdd-957f-a472d3c33e7c" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```

<img width="317" height="609" alt="image" src="https://github.com/user-attachments/assets/b54dc995-2601-4f40-b724-e7cd2f991a85" />


```
# Chi_Square
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
#In feature selection, converting columns to categorical helps certain algorithms
# (like decision trees or chi-square tests) correctly understand and
 # process non-numeric features. It ensures the model treats these columns as categories,
  # not as continuous numerical values.
df[categorical_columns]
```


<img width="1010" height="480" alt="image" src="https://github.com/user-attachments/assets/aadff792-641b-47f5-8595-74dbdd43d2fd" />


```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
##This code replaces each categorical column in the DataFrame with numbers that represent the categories.
df[categorical_columns]
```


<img width="934" height="466" alt="image" src="https://github.com/user-attachments/assets/a2d9c9e3-8103-4a3f-91ad-0ac4a96bb511" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
#X contains all columns except 'SalStat' — these are the input features used to predict something.
#y contains only the 'SalStat' column — this is the target variable you want to predict.
```
```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

<img width="713" height="246" alt="image" src="https://github.com/user-attachments/assets/dd59f917-78f9-4460-83dd-bf861ce91f2e" />

```
y_pred = rf.predict(X_test)
y_pred
```

<img width="702" height="92" alt="image" src="https://github.com/user-attachments/assets/45dc70cc-ddc6-42c9-b8d5-c66ff35d1dcc" />

```
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```


<img width="492" height="416" alt="image" src="https://github.com/user-attachments/assets/8896debf-d615-410f-b8f0-5ffebc0f5bf7" />


```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```


<img width="998" height="481" alt="image" src="https://github.com/user-attachments/assets/c3e2150c-7c04-442e-8664-62f80c5d27d6" />

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```


<img width="939" height="480" alt="image" src="https://github.com/user-attachments/assets/16236371-0696-4611-b4bc-d5ecf8f3a487" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```


<img width="780" height="95" alt="image" src="https://github.com/user-attachments/assets/de7e88a1-251f-4433-a036-80f7924e752a" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

```


<img width="549" height="129" alt="image" src="https://github.com/user-attachments/assets/79c868c7-7a2e-49a5-b525-58bae743d755" />

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```


<img width="746" height="98" alt="image" src="https://github.com/user-attachments/assets/d98a8710-2484-4c0f-adb2-d5f09755f989" />


```
!pip install skfeature-chappers
```


<img width="1364" height="328" alt="image" src="https://github.com/user-attachments/assets/e4d30336-b173-4c01-a2a0-c48cbfaaed64" />

```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
```
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```


<img width="947" height="447" alt="image" src="https://github.com/user-attachments/assets/ec51377f-b5af-4e1d-b8bf-847fbb7fc263" />


```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```


<img width="985" height="471" alt="image" src="https://github.com/user-attachments/assets/27edc758-4d68-421b-a016-675e2a9519d5" />


```
X = df.drop(columns=['SalStat'])
y = df['SalStat']

k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)

selected_features_anova = X.columns[selector_anova.get_support()]

print("\nSelected features using ANOVA:")
print(selected_features_anova)

```


<img width="1115" height="67" alt="image" src="https://github.com/user-attachments/assets/8bf5861d-b98a-4745-8e6d-2dae1ca8ece0" />

```
# Wrapper Method
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
# List of categorical columns
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

# Convert the categorical columns to category dtype
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]

```

<img width="1087" height="531" alt="image" src="https://github.com/user-attachments/assets/5502b5b7-de12-4655-a4c9-5e4ab08286f5" />


```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```


<img width="906" height="475" alt="image" src="https://github.com/user-attachments/assets/6190d8c3-2191-450b-9676-0350f655bd3f" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
logreg = LogisticRegression()
```
```
n_features_to_select =6
```
```
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```


<img width="982" height="880" alt="image" src="https://github.com/user-attachments/assets/4c6799da-2c79-4752-920c-2f6de42a9846" />

# RESULT:
       Thus, the Feature selection and Feature scaling has been used on thegiven dataset and saved to the file successfully.
