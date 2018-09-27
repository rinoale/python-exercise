l = ['Good morning', 'Good afternoon', 'Good night']

for i in l:
    print(i)

def greeting():
    yield 'Good morning'
    yield 'Good afternoon'
    yield 'Good night'

g = greeting()
print(g)
print('before')
print(next(g))
print('@@@')
print(next(g))
print('@@@')
print(next(g))
print('after')
print(g)

