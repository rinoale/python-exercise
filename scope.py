animal = 'cat'

def f():
    print(animal)
    animal = 'dog'
    print('local:', animal)

f()
print('global:', animal)
