import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

cols=['TweetID', 'Topic', 'Target', 'Text']

training = pd.read_csv('data/task4/twitter_training.csv',names=cols)
validation = pd.read_csv('data/task4/twitter_validation.csv',names=cols)

# print(training.head())
# print(validation.head())

dataset = pd.concat([training,validation])

print(dataset.head())
# Describe() but for categorical/textual data
print(dataset.describe(include = 'object') )
dataset['Topic'].value_counts().plot(kind = 'bar')
plt.show()

plt.figure(figsize=(9, 7))
crosstab = pd.crosstab(index=dataset['Topic'], columns=dataset['Target'])
sns.heatmap(crosstab, cmap = 'jet')
plt.show()


topic_list = ' '.join(crosstab.index)
wc = WordCloud(width=1000, height=500).generate(topic_list)
plt.imshow(wc, interpolation='bilinear')
plt.show()