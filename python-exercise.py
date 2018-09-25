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
