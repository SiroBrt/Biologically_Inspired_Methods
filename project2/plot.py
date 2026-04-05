import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('pi.csv', header=None)
plt.plot(data.iloc[1:,0].values, data.iloc[1:,1].values)
plt.show()
data = pd.read_csv('eul.csv', header=None)
plt.plot(data.iloc[1:,0].values, data.iloc[1:,1].values)
plt.show()

