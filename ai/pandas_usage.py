import sys
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('sample.csv')

print(df)

print(df.head(3))

print(df['x'])

x = df['x']
y = df['y']

plt.savefig(r"/mnt/c/Users/gbsong/plt.png")

print(df.describe())

df_c = df - df.mean()

print(df_c.describe())

xx = df_c['x'] * df_c['x']
xy = df_c['x'] * df_c['y']

a = xy.sum() / xx.sum()

print(a)

plt.scatter(df_c['x'], df_c['y'])
plt.plot(df_c['x'], a*df_c['x'])

plt.savefig(r"/mnt/c/Users/gbsong/plt_with_a.png")
