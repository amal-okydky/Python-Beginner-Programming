# Function


```python
def my_function(p, p1):
  print(p*1)
```


```python
my_function(2,4)
```

    2
    


```python
def luas_persegi_panjang(panjang, lebar):
  luas = panjang * lebar
  return luas

lp = luas_persegi_panjang(panjang = 10, lebar = 20)
print(lp)
```

    200
    


```python
def cetak_aku(x):
  print(x)
  return

cetak_aku("Nama saya AP")
```

    Nama saya AP
    


```python
# Versi 1 Tanpa input, tanpa output
def salam():
  print('Selamat malam semua')
  print('Selamat datang dipertemuan ketiga kampus merdeka')
  return

# Versi 2 dengan input, tanpa output
def salam(nama):
  print('Selamat malam ', nama)
  print('Selamat datang dipertemuan ketiga kampus merdeka')
  return

salam(nama = 'AP')
```

    Selamat malam  AP
    Selamat datang dipertemuan ketiga kampus merdeka
    


```python
# Versi 3 Dengan input dan output
def luas_persegi_panjang(panjang, lebar):
  luas = panjang * lebar
  return luas
```


```python
luas_persegi_panjang(10, 20)
```




    200




```python
luas_persegi_panjang(lebar = 5, panjang = 3) 
```




    15




```python
luas_persegi_panjang(panjang = 15, lebar = 6)
```




    90



## Definition


```python
def my_function(a,b):
  "Function untuk menghitung luas"
```


```python
 print(2 * 4)
```

    8
    

### Calling a Function


```python
# Function definition is here
def printme( str ):
  "This prints a passed string into this function"
  print(str)
  return

# Now you can call printme function
printme("I'm first call to user defined function!")
printme("Again second call to the same function")
```

    I'm first call to user defined function!
    Again second call to the same function
    

### Pass by reference vs value


```python
# Function definition is here
def changeme ( mylist ):
  "This changes a passed list into this function"
  mylist.append([1,2,3,4]);
  print("Values inside the function: ", mylist)
  return

# Now you can call changme funtion
mylist = [10,20,20];
changeme( mylist );
print("Values outside the function: ", mylist)
```

    Values inside the function:  [10, 20, 20, [1, 2, 3, 4]]
    Values outside the function:  [10, 20, 20, [1, 2, 3, 4]]
    


```python
# Function definition is here
def changeme ( mylist ):
  "This changes a passed list into this function"
  mylist = [1,2,3,4]; # This would assig new reference in mylist
  print("Values inside the function: ", mylist)
  return

# Now you can call changme funtion
mylist = [10,20,20];
changeme( mylist );
print("Values outside the function: ", mylist)
```

    Values inside the function:  [1, 2, 3, 4]
    Values outside the function:  [10, 20, 20]
    

# Function Arguments


```python
# Function definition is here
sum = lambda arg1, arg2: arg1 + arg2;

def sum(arg1, arg2):
  arg1 + arg2

# Now you can call sum as a funtion
print("Value of total : ", sum( 10, 20 ))
print("Value of total : ", sum ( 10, 20))
```

    Value of total :  None
    Value of total :  None
    

### Return Statement


```python
# Function definition is here
def sum(arg1, arg2):
  #Add both the parameters and return them
  total = arg1 + arg2
  total2 = total + arg1
  print("Inside the function : ", total)
  return total2

sum(10,20)
```

    Inside the function :  30
    




    40




```python
# Function definition is here
def sum(arg1, arg2):
  #Add both the parameters and return them
  total = arg1 + arg2
  total2 = total + arg1
  print("Inside the function : ", total)
  return total2

# Now you can call sum as a funtion
total = sum(10,20)
print("Outside the function : ", total)
```

    Inside the function :  30
    Outside the function :  40
    


```python
jumlahKucing = 20

def jumlahHewan():
  jumlahAnjing = 30
  return jumlahKucing + jumlahAnjing

def jumlahKelinci():
  return jumlahKucing + jumlahKucing

jumlahHewan()
jumlahKelinci()
```




    40


