import numpy as np

x = np.array([1, 2, 3])
y = np.array([2, 3.9, 6.1])

print(x.mean())
print(y.mean())

xc = x - x.mean()
yc = y - y.mean()

print(xc, yc)

xx = xc * xc
yy = yc * yc

print(xx, yy)

xy = xc * yc

print(xy.sum())

a = xy.sum() / xx.sum()

print(a)
