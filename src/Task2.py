import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data/titan.csv')
print('---- Head ---')
print(data.head())
print('---- Tail ---')
print(data.tail())
print('---- Describe ---')
print(data.describe())
print('---- Info ---')
print(data.info())
print('---- Sum null ---')
print(data.isnull().sum())

print('---- Replace null by value ---')

data['Age'].replace(to_replace=np.nan, value=int(data['Age'].mean()), inplace=True)
data['Age'] = data['Age'].astype(int)
print(data['Age'])

data['Ticket'] = data['Ticket'].str.replace('\D+', '', regex=True)

data['Cabin'] = data['Cabin'].fillna(method='ffill').fillna(method='bfill')

data['Embarked'] = data['Embarked'].fillna(method='ffill')

print('---- New data ---')

print(data)

sns.set_theme(style="darkgrid")

plt.figure(figsize=(10, 6))

sns.countplot(x='Survived', data=data)
plt.title('Survival Count')
plt.show()

sns.countplot(x='Sex', data=data)
plt.title('Sex Count')
plt.show()

sns.countplot(x='Sex', hue='Survived', data=data)
plt.title('Survived vs Sex')
plt.show()

sns.countplot(x='Survived', hue='Pclass', data=data)
plt.title('Survival by Passenger Class')
plt.show()