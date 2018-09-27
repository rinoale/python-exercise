w = ['mon', 'tue', 'wed']
f = ['coffee', 'milk', 'water']

d = {}
for x, y in zip(w, f):
    d[x] = y

print(d)

d = {x: y for x, y in zip(w, f)}

print(d)

d = {i for i in range(10) if i%2 == 0}

print(d)

def g():
    for i in range(10):
        yield i

g = g()

print(type(g))

g = (i for i in range(10))

print(type(g))

g = tuple(i for i in range(10) if i%2 == 0)

print(type(g))

for i in g:
    print(i)
