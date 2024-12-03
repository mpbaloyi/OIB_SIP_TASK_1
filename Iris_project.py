#Mikateko Petronella Baloyi
#Project Iris

import warnings
warnings.filterwarnings('ignore')
#importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

#loading the the Iris csv file

df=pd.read_csv(r'C:\Users\mihlotib\Desktop\Mika\Internship_Oasis Infobyte\Iris.csv')

#displaying the rows of the csv file(by default it will return the first 5 rows unless you specify the number)
#And inspecting the structure of the dataFrame
rows=df.head()
print(rows)

#Displaying the columns
cols=df.columns
print(cols)

#Displaying  the number of rows and columns
shyp=df.shape
print(shyp)

#Getting overview of the structure and characteristics of the Iris DataFrame
df.info()

#generating descriptive statistics of the Iris DataFrame
print(df.describe())

#Determining their data types

#renaming the species names and counting its values
df['Species'] = df['Species'].str.replace('Iris-', '')
print("\n",df['Species'].value_counts())

#creating a pie chart for the number of species
df['Species'].value_counts().plot(kind='pie',autopct='%0.1f%%',ylabel='',title='Species of Iris')
plt.show()

#comparing the central tendency, variability, and potential outliers for each species in the Iris dataFrame
sns.boxplot(data=df, x='Species', y='SepalLengthCm')
plt.title('Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length')

plt.tight_layout()
plt.show()

sns.boxplot(data=df, x='Species', y='SepalWidthCm')
plt.title('Sepal Width by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Width')
plt.tight_layout()
plt.show()

sns.boxplot(data=df, x='Species', y='PetalLengthCm')
plt.title('Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length')

plt.tight_layout()
plt.show()

sns.boxplot(data=df, x='Species', y='PetalWidthCm')
plt.title('Petal Width by Species')
plt.xlabel('Species')
plt.ylabel('Petal Width')

plt.tight_layout()
plt.show()

#creating and evaluating the performance of a classification model playing a distribution plot for the SepalLengthCm column
sns.displot(data = df, x = 'SepalLengthCm', kde = True)
plt.title('Distribevaluate the performance of a classification modelution of Sepal Length in CM')
plt.tight_layout() #making sure everything is visible
plt.show()
#creating and displaying a distribution plot for the SepalWidthCm column
sns.displot(data = df, x = 'SepalWidthCm', kde = True)
plt.title('Distribution of Sepal Width in CM')
plt.tight_layout()
plt.show()
#creating and displaying a distribution plot for the PetalLengthCm column
sns.displot(data = df, x = 'PetalLengthCm', kde = True)
plt.title('Distribution of Petal Length in CM')
plt.tight_layout()
plt.show()
#creating and displaying a distribution plot for the PetalWidthCm column
sns.displot(data = df, x = 'PetalWidthCm', kde = True)
plt.title('Distribution of Petal Width in CM')
plt.tight_layout()
plt.show()

# Pairplot to visualize the relationships between features
sns.pairplot(df.drop('Id', axis=1), hue='Species')
#plt.tight_layout()
plt.show()

# Correlation heatmap
numeric_col = df.select_dtypes(include=[np.number]) #selecting only the numerical columns from the Iris DataFrame
corr_matrix = numeric_col.corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu') #the colors can be(coolwarm,bwr or RdBu)
plt.tight_layout()
plt.show()

# Splitting the data into training and testing sets
X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(scaler,X_test,X_train)

# Training a K-Nearest Neighbors classifier
K_near = KNeighborsClassifier(n_neighbors=5)
K_near.fit(X_train, y_train)
y_pred = K_near.predict(X_test)

# evaluating the performance of a classification model
accuracy = accuracy_score(y_test, y_pred)
print("accurancy: \n",accuracy)
conf_matrix = confusion_matrix(y_test, y_pred)
print(" Matrix: \n",conf_matrix)
class_report = classification_report(y_test, y_pred)
print("Report \n",class_report)










