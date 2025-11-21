# Titanic Survival Prediction ML Project


## Overview

This project is a complete Machine Learning analysis on the Titanic dataset. It involves data exploration, preprocessing, feature engineering, model training, and evaluation to predict passenger survival based on various features like age, sex, class, etc. The project is implemented in Python using libraries such as Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn.



## Dataset

The dataset used in this project is "train.csv" obtained from here: https://www.kaggle.com/code/mrisdal/exploring-survival-on-the-titanic/input?select=train.csv

It contains information about 891 passengers. Key features include:

### Data Dictionary

| Variable   | Definition                          | Key                                      |
|------------|-------------------------------------|------------------------------------------|
| survival  | Survival                            | 0 = No, 1 = Yes                          |
| pclass    | Ticket class                        | 1 = 1st, 2 = 2nd, 3 = 3rd                |
| sex       | Sex                                 |                                          |
| Age       | Age in years                        |                                          |
| sibsp     | No. of siblings / spouses aboard the Titanic | |
| parch     | No. of parents / children aboard the Titanic | |
| ticket    | Ticket number                       |                                          |
| fare      | Passenger fare (in British pounds (£)) |                                       |
| cabin     | Cabin number                        |                                          |
| embarked  | Port of Embarkation                 | C = Cherbourg, Q = Queenstown, S = Southampton |

- **Source**: The dataset is from Kaggle's Titanic competition.
- **Rows**: 891
- **Columns**: 12
- **Missing Values**: Handled for Age (filled with median), Embarked (filled with mode), and Cabin (dropped).


## Project Structure

```
titanic-ml-project/
├── Complete Titanic ML Analysis.ipynb  # Main Jupyter Notebook
├── Complete Titanic ML Analysis.pdf     # PDF export of the notebook
├── train.csv                            # Titanic dataset
├── headerImage.png                      # Header image used in the notebook
├── ending_image.jpg                     # Ending image used in the notebook
├── requirements.txt                     # Python dependencies
└── README.md                            # This file (explaining about this project)
```

## Data Preprocessing

- **Handling Missing Values**:
  - Age: Filled with median value.
  - Embarked: Filled with mode (most frequent value).
  - Cabin: Dropped due to following reasons:
        - high missing values (687 out of 891).
        - this column would also not be useful for ML related tasks.

- **Feature Engineering**:
  - These features are encoded so that we can easily apply machine-learning on it.
     a- sex
     b- embarkation-point
  - These new features are created out of the following:
     a- number of family members created from summing up no. of siblings , no. of spouse (aks SibSp)  and no. of parents (And/or) children (aka Parch).
  - Dropped irrelevant columns like PassengerId, Name, Ticket.



## Exploratory Data Analysis (EDA)

- **Basic Inspections**: Checked shape (891 rows, 12 columns), data types, and null values.
- **Visualizations**: 
  - Survival comparison by sex, class and etc.
  - Correlation heatmap.

Some findings done at the beginning:
- No. of rows: 891
- No. of columns: 12
- Missing values before handling: Age (177), Cabin (687), Embarked (2)

## Model Training and Evaluation

- **Models Used**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors (KNN)

- **Train-Test Split**: 70% of data was used for training and the remaining 30% for testing.
- **Evaluation Metrics**: Accuracy, Confusion Matrix.

Code example:
```
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

X = df[['Fare']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

plt.figure(figsize=(12,6))
plot_tree(model, feature_names=['Fare'], class_names=['Not Survived','Survived'], filled=True)
plt.show()

print("Accuracy:", model.score(X_test, y_test))

```

```
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

knn.score(X_test, y_test)
# Output: 0.7318435754189944

knn.predict([[1, 38, 71]])
# Output: array([1])


from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm
# Output: array([[91, 15],
#                [33, 40]])

import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted how many survived or not survived')
plt.ylabel('Truth of how many actually survived or not survived')
plt.show()

```


## Results observed (like from the above code snippet)

- **Model Performance** (example results; run notebook for exact):
  - KNN: Accuracy ~73%
  - CONCLUSION OF THIS KNN MODEL: A Female Person of 38 years of age and with an income of 71 pounds (British pounds) is likely to survive.
 




## Author

Prepared by:  
- **Name**: Om Satyawan Pathak  
- **Contact**: omsatyawanpathakwebdevelopment@gmail.com (or) omsatyawanpathakgit@gmail.com  

Feel free to contact me for any further clarifications.
