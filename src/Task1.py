import pandas as pd
import matplotlib.pyplot as plt

data_totl = pd.read_csv('data/POP.TOTL.csv')
data_man = pd.read_csv('data/POP.TOTL.MAN.csv')
data_fe = pd.read_csv('data/POP.TOTL.FE.csv')

df_totl = pd.DataFrame(data_totl)
df_man = pd.DataFrame(data_man)
df_fe = pd.DataFrame(data_fe)

df_totl.drop(columns=['Country Code','Indicator Name', 'Indicator Code','Country Name','Unnamed: 68'], inplace=True)
df_man.drop(columns=['Country Code','Indicator Name', 'Indicator Code','Country Name','Unnamed: 68'], inplace=True)
df_fe.drop(columns=['Country Code','Indicator Name', 'Indicator Code','Country Name','Unnamed: 68'], inplace=True)
df_sum = pd.DataFrame([df_man.sum(),df_fe.sum()]);
df_sum = df_sum.transpose()
df_sum.columns = ['Male','Female']
df_sum.plot();
plt.show()