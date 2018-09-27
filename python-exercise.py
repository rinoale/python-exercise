print('test')

a = 10
b = a
c = b
print(c)

num = 1
name = 'Mike'

print(num, type(num))
print(name, type(name))

print('Hi', 'Mike', sep=',', end='\n')

print(17//3)
print(17%3)
print(5**2)

print("""
line1
line2
line3
""")

print("""\
line1
line2
line3\
""")

print('Hi.' * 3 + 'Mike.')
print('Py''thon')

s = ('aaaaaaaaaaaaaaaaaaaaaaaa'
    'bbbbbbbbbbbbbbbbbbbbbbbb')

print(s)

word = 'python'

print(word[0:2])
print(word[:2])

word = 'j' + word[1:]

print(word)

print('a is {}'.format('a'))

print('a is {} {} {}'.format(1, 2, 3))

print('My name is {0} {1}'.format('Jun', 'Sakai'))

print('My name is {name} {family}. watashiha {family} {name}'.format(name='Jun', family='Sakai'))

name = 'Jun'
family = 'Sakai'
# print('My name is {name} {family}. watashiha {family} {name}'.format(name, family))
# upper example was incorrect

apples = 4

print(f"I have {apples} apples")

n= [1,2,3,4,5,6,7,8,9,10]

print(n[::2])
print(n[::-1])

s = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

s[2:5] = ['C', 'D', 'E']

print(s)

r = [1,2,3,4,5,1,2,3]

if 100 in r:
    print('exist')
else:
    print('not exist')

i = [1, 2, 3, 4, 5]
j = i
j[0] = 100
print('i = ', i)
print('j = ', j)

x = [1, 2, 3, 4, 5]
# y = x.copy()
y = x[:]

x[0] = 100
print('x = ', x)
print('y = ', y)

X = 20
Y = X
Y = 5
print(id(X))
print(id(Y))
print(X)
print(Y)

Z = [1, 2, 3]
C = Z
Z[0] = 4

print(id(Z))
print(id(C))
print(Z)
print(C)

seat = []

min = 0
max = 5

print(min <= len(seat) < max)

t = (1, 2, 3, 4, 5)

print(type(t))

t1 = 1,

print(type(t1))

i = 10
j = 20
tmp = i
i = j
j = tmp


a = 100
b = 200

a, b = b, a

print(a, b)

d = {'x': 10, 'y': 20}
d['z'] = 30

print(type(d))
print(d)

s = {1,2,3,4,4,4,5,6}

b = {2,3,3,6,7}

print(s-b)
print(s&b)
print(s|b)
print(s^b)

is_ok = 0
# 0.0, 0, '', [], (), {}, set() and False are false
if is_ok:
    print('OK!')
else:
    print('No!')

is_empty = None

if is_empty is None:
    print('None')


def print_info(func):
    def wrapper(*args, **kwargs):
        print('start')
        result = func(*args, **kwargs)
        print('end')
        return result
    return wrapper

@print_info
def add_num(a, b):
    return a+b

r = add_num(10, 20)
print(r)
