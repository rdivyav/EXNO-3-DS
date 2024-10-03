## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data (1).csv")
df
```
![Screenshot 2024-10-03 110830](https://github.com/user-attachments/assets/7c611466-82c4-4ef3-9702-bb5c30fc68d9)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2024-10-03 110850](https://github.com/user-attachments/assets/300ff0b4-010f-4bd6-aa97-7c17fc5616e4)



```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2024-10-03 110902](https://github.com/user-attachments/assets/1707e0cc-1d85-4683-9009-0793d0a95c31)


```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2024-10-03 110912](https://github.com/user-attachments/assets/ab3b27d4-6830-421b-adad-82297b1d821c)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2024-10-03 110923](https://github.com/user-attachments/assets/dd5aa8c0-bc70-4585-a976-df5395f5485f)


```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2024-10-03 110936](https://github.com/user-attachments/assets/89446de0-3678-4f0b-b51c-5e1a4cecccdb)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2024-10-03 110950](https://github.com/user-attachments/assets/9ea4c134-de9b-4140-a380-2d1953e4a85f)


pip install --upgrade category_encoders
Collecting category_encoders
  Downloading category_encoders-2.6.4-py2.py3-none-any.whl.metadata (8.0 kB)
Requirement already satisfied: numpy>=1.14.0 in /usr/local/lib/python3.10/dist-packages (from category_encoders) (1.26.4)
Requirement already satisfied: scikit-learn>=0.20.0 in /usr/local/lib/python3.10/dist-packages (from category_encoders) (1.5.2)
Requirement already satisfied: scipy>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from category_encoders) (1.13.1)
Requirement already satisfied: statsmodels>=0.9.0 in /usr/local/lib/python3.10/dist-packages (from category_encoders) (0.14.3)
Requirement already satisfied: pandas>=1.0.5 in /usr/local/lib/python3.10/dist-packages (from category_encoders) (2.2.2)
Requirement already satisfied: patsy>=0.5.1 in /usr/local/lib/python3.10/dist-packages (from category_encoders) (0.5.6)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.0.5->category_encoders) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.0.5->category_encoders) (2024.2)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.0.5->category_encoders) (2024.2)
Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from patsy>=0.5.1->category_encoders) (1.16.0)
Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn>=0.20.0->category_encoders) (1.4.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn>=0.20.0->category_encoders) (3.5.0)
Requirement already satisfied: packaging>=21.3 in /usr/local/lib/python3.10/dist-packages (from statsmodels>=0.9.0->category_encoders) (24.1)
Downloading category_encoders-2.6.4-py2.py3-none-any.whl (82 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 82.0/82.0 kB 1.9 MB/s eta 0:00:00
Installing collected packages: category_encoders
Successfully installed category_encoders-2.6.4
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![Screenshot 2024-10-03 111014](https://github.com/user-attachments/assets/546261ee-8ea2-435c-89c8-26912a81e5ba)


```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![Screenshot 2024-10-03 111029](https://github.com/user-attachments/assets/916e00f1-4211-45df-9e88-94a5529d361e)


```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![Screenshot 2024-10-03 111040](https://github.com/user-attachments/assets/60d9abe9-f36d-4d7a-9829-89088f2dcbf4)


```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![Screenshot 2024-10-03 111053](https://github.com/user-attachments/assets/d742d72d-e15c-43e1-b622-af46ea3cd954)


```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2024-10-03 111106](https://github.com/user-attachments/assets/dd8f53e7-f4f6-4f6b-b82e-44f49f99f557)


```
np.reciprocal(df["Moderate Positive Skew"])
```
![Screenshot 2024-10-03 111119](https://github.com/user-attachments/assets/784f03ee-4d49-4ca2-a6ad-3f9990e96d1f)


```
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2024-10-03 111132](https://github.com/user-attachments/assets/80e1679c-9bbb-47d5-9d85-96256b5e137e)



np.square(df["Highly Positive Skew"])
![Screenshot 2024-10-03 111141](https://github.com/user-attachments/assets/59d4df70-0f79-45bf-8dc8-49fd70efe1c1)


```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![Screenshot 2024-10-03 111158](https://github.com/user-attachments/assets/e8f3ab3b-9743-4e3f-aaee-b589c137eee4)


```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![Screenshot 2024-10-03 111215](https://github.com/user-attachments/assets/1588bd1e-61ee-4523-8cc8-645aaad46623)


```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![Screenshot 2024-10-03 111229](https://github.com/user-attachments/assets/0479d844-28ef-4522-99dd-f5acbd7d5315)


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![Screenshot 2024-10-03 111254](https://github.com/user-attachments/assets/27ff308c-12ff-484a-b457-0dc3733350ad)


```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-10-03 111308](https://github.com/user-attachments/assets/dc5fa03a-6bba-4a1a-bd8f-45787e45f25d)


```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![Screenshot 2024-10-03 111320](https://github.com/user-attachments/assets/45cd7430-e828-4013-ad9d-5ec8a6c824c6)



```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-10-03 111333](https://github.com/user-attachments/assets/47dce18a-9208-4c98-80b9-95b72233ba67)


```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![Screenshot 2024-10-03 111350](https://github.com/user-attachments/assets/cd51265a-e3be-4ec2-9c47-ca780f623dd5)



```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```

![Screenshot 2024-10-03 111409](https://github.com/user-attachments/assets/225e9905-08c9-411d-b570-60a558705242)

   
# RESULT:
       Thus the given data is read ,performed Feature Encoding and Transformation process and saved the data to a file.

       
