# Types of Data

### int(integers)


```python
print(123456789111 + 1)
```

    123456789112
    


```python
print(123+4)
```

    127
    


```python
print(10)
```

    10
    


```python
print(type(10))
```

    <class 'int'>
    

### float(floating-point numbers) 


```python
print(4.2)
print(type(4.2))

print(4.)
print(.2)
print(.4e7)
print(4.2e-4)
```

    4.2
    <class 'float'>
    4.0
    0.2
    4000000.0
    0.00042
    

### str(string)


```python
print("Hactiv8")
print(type("Hactiv8"))
```

    Hactiv8
    <class 'str'>
    


```python
print("This stirng contains a single quote (') character.")
print('This string contains a double quote (") character.')
```

### bool(boolean)


```python
print(type(True))
print(type(False))
```

    <class 'bool'>
    <class 'bool'>
    

# Variable (Assignment, Types, Names)


```python
n = 300
print(n)
```

    300
    


```python
print(type(n))
```

    <class 'int'>
    


```python
a = b = c = 300
print(a, b , c)
```

    300 300 300
    


```python
var = 23.5
print(var)

var = "Now I'm a String"
print(var)
```

    23.5
    Now I'm a String
    


```python
name = "Hactiv8"
Age = 54
has_laptops = True
print(name, Age, has_laptops)
```

    Hactiv8 54 True
    


```python
age = 1
Age = 2
aGe = 3
AGE = 4
a_g_e = 5
_age = 6
age_ = 7
_AGE_ = 8

print(age, Age, aGe, AGE, a_g_e, _age, age_, _AGE_)
```

    1 2 3 4 5 6 7 8
    

# Operators & Expression


```python
a = 10
b = 20
a + b
```




    30




```python
a = 10
b = 20
a + b - 5
```




    25



### Aritmetic Operators


```python
# Here are some examples of these operators in use:

a = 4
b = 3

print (a + b)
print (a - b)
print (a * b)
print (a / b)
print (a % b)
print (a ** b)
```

    7
    1
    12
    1.3333333333333333
    1
    64
    


```python
# The result of standard division (/) is always a float, even if the dividend is evenly divisible by the divisor:

10/5
```




    2.0




```python
# Here are examples of the comparison operators in use:

a = 10
b = 20
print (a == b)
print (a != b)
print (a <= b)
print (a >= b)
```

    False
    True
    True
    False
    


```python
a = 30
b = 30
print (a == b)
print (a <= b)
print (a >= b)
```

    True
    True
    True
    

### Stirng Manupulation


```python
# + Operators

s = 'foo'
t = 'bar'
u = 'baz'

print(s + t)
print(s + t + u)

print('Hacktiv8' + 'PTP')
```

    foobar
    foobarbaz
    Hacktiv8PTP
    


```python
# * Operators

s = 'foo'
s * 4
```




    'foofoofoofoo'




```python
# in Operators

s = 'foo'
print(s in 'That food for us')
print(s in 'That good for us')
```

    True
    False
    


```python
# Case Conversion
s = 'HackTIV8'

# Captilize
print(s.capitalize())

# Lower
print(s.lower())

# Swapcase
print(s.swapcase())

# Title
print(s.title())

# Uppercase
print(s.upper())
```

    Hacktiv8
    hacktiv8
    hACKtiv8
    Hacktiv8
    HACKTIV8
    

# List


```python
a = ['foo', 'bar', 'baz', 'qux']
print(a)
```

    ['foo', 'bar', 'baz', 'qux']
    

### Lists are ordered


```python
a = ['foo', 'bar', 'qux']
b = ['baz', 'qux', 'foo']

a == b
```




    False



### Lists can be Arbitrary Objects


```python
a = [21.42, 'foobar', 3, 4, 'bark', False, 3.14159]
print(a)
```

    [21.42, 'foobar', 3, 4, 'bark', False, 3.14159]
    


```python
a = [21.42, 'foobar', 3, 4, 'bark', False, 3.14159]
print(type(a))
```

    <class 'list'>
    

### List elements can be accessed by index


```python
# Indeks List Positif dihitung dari depan dengan dimulai dari angka 0
a = ['foo', 'bar', 'baz', 'qux', 'quux', 'corge']
print(a[0])
print(a[5])
```

    foo
    corge
    


```python
# Indeks List Negatif dihitung dari akhir/belakang
a = ['foo', 'bar', 'baz', 'qux', 'quux', 'corge']
print(a[-1])
print(a[-6])
```

    corge
    foo
    


```python
# Slicing menggunakan expression a[m:n], mengembalikan porsi a dari indeks m ke n, tetapi tidak termasuk indeks n:
a = ['foo', 'bar', 'baz', 'qux', 'quux', 'corge']
a[2:5]
```




    ['baz', 'qux', 'quux']




```python
# The concatenation (+) and replication (*) operators:
print(a)

print(a + ['grault', 'garply'])
print(a * 2)
```

    ['foo', 'bar', 'baz', 'qux', 'quux', 'corge']
    ['foo', 'bar', 'baz', 'qux', 'quux', 'corge', 'grault', 'garply']
    ['foo', 'bar', 'baz', 'qux', 'quux', 'corge', 'foo', 'bar', 'baz', 'qux', 'quux', 'corge']
    


```python
# len(), min(), max()

print(a)

print(len(a))
print(min(a))
print(max(a))
```

    ['foo', 'bar', 'baz', 'qux', 'quux', 'corge']
    6
    bar
    qux
    

### Modifying a Single List Value


```python
a = ['foo', 'bar', 'baz', 'qux', 'quux', 'corge']
print(a)

a[2] = 10
a[-1] = 20

print(a)
```

    ['foo', 'bar', 'baz', 'qux', 'quux', 'corge']
    ['foo', 'bar', 10, 'qux', 'quux', 20]
    


```python
# A list item can be delated with the del command:

del a[3]
```


```python
print(a)
```

    ['foo', 'bar', 10, 'quux', 20]
    


```python
a = ['foo', 'bar', 'baz', 'qux', 'quux', 'corge']

print(a[1:4])
```

    ['bar', 'baz', 'qux']
    


```python
a[1:6]
```




    ['bar', 'baz', 'qux', 'quux', 'corge']




```python
a[1:]
```




    ['bar', 'baz', 'qux', 'quux', 'corge']




```python
a[0:]
```




    ['foo', 'bar', 'baz', 'qux', 'quux', 'corge']



### Modifying Multiple List Value


```python
a = ['foo', 'bar', 'baz', 'qux', 'quux', 'corge']

print(a[1:4])
a[1:4] = [1.1, 2.2, 3.3, 4.4,5.5]

print(a)
```

    ['bar', 'baz', 'qux']
    ['foo', 1.1, 2.2, 3.3, 4.4, 5.5, 'quux', 'corge']
    


```python
a[1:6]
```




    [1.1, 2.2, 3.3, 4.4, 5.5]



# Python Tuples


```python
t = ('foo', 'bar', 'qux', 'quux', 'corge')
print(t)
```

    ('foo', 'bar', 'qux', 'quux', 'corge')
    


```python
# packing and unpacking

(s1, s2, s3, s4) = ('foo', 'bar', 'baz',  'qux')
s1
```




    'foo'



# Python Dictionary


```python
MLB_team = {
    'Colorado' : 'Rockies',
    'Boston' : 'Red Sox',
    'Minnesota' : 'Twins',
    'Milwaukee' : 'Brewers',
    'Seattle' : 'Mariners',
}
```

### Accessing Dictionary Values


```python
print(MLB_team['Minnesota'])
print(MLB_team['Colorado'])
```

    Twins
    Rockies
    


```python
# Adding an entry to an existing dictionary is simply a matter of assigning a new key and value

MLB_team['Kansas City'] = 'Royals'
MLB_team
```




    {'Colorado': 'Rockies',
     'Boston': 'Red Sox',
     'Minnesota': 'Twins',
     'Milwaukee': 'Brewers',
     'Seattle': 'Mariners',
     'Kansas City': 'Royals'}




```python
# If you want to update an entry, you can just assign a new value to an existing key:

MLB_team['Seattle'] = 'Seahawks'
MLB_team
```




    {'Colorado': 'Rockies',
     'Boston': 'Red Sox',
     'Minnesota': 'Twins',
     'Milwaukee': 'Brewers',
     'Seattle': 'Seahawks',
     'Kansas City': 'Royals'}




```python
del MLB_team['Seattle']
MLB_team
```




    {'Colorado': 'Rockies',
     'Boston': 'Red Sox',
     'Minnesota': 'Twins',
     'Milwaukee': 'Brewers',
     'Kansas City': 'Royals'}



### Building a Dictionary Incrementally


```python
person = {}
type(person)

person['fname'] = 'Hack'
person['lname'] = 'PTP'
person['age'] = 51
person['spouse'] = 'Edna'
person['children'] = ['Ralph', 'Betty', 'Joey']
person['pets'] = {'dog': 'Fido', 'cat': 'Sox'}
```


```python
print(person['fname'])
print(person['lname'])
```

    Hack
    PTP
    


```python
print(person['children'])
print(person['children'][1])
```

    ['Ralph', 'Betty', 'Joey']
    Betty
    


```python
print(person['pets'])
print(person['pets']['cat'])
```

    {'dog': 'Fido', 'cat': 'Sox'}
    Sox
    


```python
# Built-in Methods
d = {'a': 10, 'b': 20, 'c': 30}

# items
print(d.items())

# keys
print(d.keys())

# values
print(d.values())
```

    dict_items([('a', 10), ('b', 20), ('c', 30)])
    dict_keys(['a', 'b', 'c'])
    dict_values([10, 20, 30])
    

### Line Continuation


```python
person1_age = 42
person2_age = 16
person3_age = 71

someone_is_of_working_age = (person1_age >= 18 and person1_age <= 65) or (person2_age >= 18 and person2_age <= 65) or (person3_age >= 18 and person3_age <= 65)
someone_is_of_working_age
```




    True




```python
someone_is_of_working_age = (
    (person1_age >= 18 and person1_age <= 65)
    or (person2_age >= 18 and person2_age <= 65) 
    or (person3_age >= 18 and person3_age <= 65)
)
someone_is_of_working_age
```




    True




```python

```
