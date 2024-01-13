# NumPy


```python
import numpy as np
```


```python
a = np.array([1, 2, 3])

print(a)
```

    [1 2 3]
    


```python
b = [1, 2, 3]

print(b)
```

    [1, 2, 3]
    


```python
type(a)
```




    numpy.ndarray




```python
type(b)
```




    list




```python
np.zeros(6)
```




    array([0., 0., 0., 0., 0., 0.])




```python
np.empty(6)
```




    array([0., 0., 0., 0., 0., 0.])




```python
print(np.arange(4))
print(np.arange(0,10,2))
```

    [0 1 2 3]
    [0 2 4 6 8]
    


```python
np.arange(2,29,5)
```




    array([ 2,  7, 12, 17, 22, 27])




```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
```


```python
np.append(arr, [1,2])
```




    array([1, 2, 3, 4, 5, 6, 7, 8, 1, 2])




```python
np.delete(arr, 1)
```




    array([1, 3, 4, 5, 6, 7, 8])




```python
np.sort(arr)
```




    array([1, 2, 3, 4, 5, 6, 7, 8])




```python
array_example = np.array([[[0, 1, 2, 3],
                            [4, 5, 6, 7]],
                          
                            [[0, 1, 2, 4],
                             [4, 5, 6, 7]],
                          
                          [[0, 1, 2, 3],
                           [4, 5, 6, 7]]])

print(array_example)
```

    [[[0 1 2 3]
      [4 5 6 7]]
    
     [[0 1 2 4]
      [4 5 6 7]]
    
     [[0 1 2 3]
      [4 5 6 7]]]
    


```python
array_example.ndim
```




    3




```python
array_example.size
```




    24




```python
array_example.shape
```




    (3, 2, 4)




```python
arr_one = np.array([[1, 2, 3, 4, 5]])
```


```python
arr_one.ndim
```




    2




```python
arr_one.size
```




    5




```python
arr_one.shape
```




    (1, 5)




```python
a = np.arange(6)

print(a)
```

    [0 1 2 3 4 5]
    


```python
b = a.reshape(3,2)

print(b)
```

    [[0 1]
     [2 3]
     [4 5]]
    


```python
a.reshape(6,1)
```




    array([[0],
           [1],
           [2],
           [3],
           [4],
           [5]])




```python
a = np.array([1, 2, 3, 4, 5, 6])
a.shape
```




    (6,)




```python
a2 = a[np.newaxis]
print(a2.shape)
print(a2)
```

    (1, 6)
    [[1 2 3 4 5 6]]
    


```python
row_vector = a[np.newaxis, :]
print(row_vector.shape)
print(row_vector)
```

    (1, 6)
    [[1 2 3 4 5 6]]
    


```python
col_vector = a[:, np.newaxis]
print(col_vector.shape)
print(col_vector)
```

    (6, 1)
    [[1]
     [2]
     [3]
     [4]
     [5]
     [6]]
    


```python
a = np.array([1, 2, 3, 4, 5, 6])
a.shape
```




    (6,)




```python
b = np.expand_dims(a, axis=1)
b.shape
```




    (6, 1)




```python
c = np.expand_dims(a, axis=0)
c.shape
```




    (1, 6)




```python
data = np.array([1,2,3])

print(data)
print(data[0])
print(data[1])
print(data[0:2])
print(data[1:])
print(data[-2:])
```

    [1 2 3]
    1
    2
    [1 2]
    [2 3]
    [2 3]
    


```python
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
a
```




    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])




```python
print(a[a>=5])
```

    [ 5  6  7  8  9 10 11 12]
    


```python
five_up = (a >= 5)

print(a[five_up])
print(a[a>=5])
```

    [ 5  6  7  8  9 10 11 12]
    [ 5  6  7  8  9 10 11 12]
    


```python
divisible_by_2 = a[a%2==0]

print(divisible_by_2)
```

    [ 2  4  6  8 10 12]
    


```python
c = a[(a > 2) & (a < 11)]

print(c)
```

    [ 3  4  5  6  7  8  9 10]
    


```python
arr = np.array([ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

arr1 = arr[3:8]
arr1
```




    array([4, 5, 6, 7, 8])




```python
a_1 = np.array([[1, 1],
        [2, 2]])
```


```python
a_2 = np.array([[3, 3],
        [4, 4]])
```


```python
np.vstack((a_1, a_2))
```




    array([[1, 1],
           [2, 2],
           [3, 3],
           [4, 4]])




```python
np.hstack((a_1, a_2))
```




    array([[1, 1, 3, 3],
           [2, 2, 4, 4]])




```python
a = np.arange(1, 25, 1)

arrsplit = a.reshape(2, 12)
print(arrsplit)
```

    [[ 1  2  3  4  5  6  7  8  9 10 11 12]
     [13 14 15 16 17 18 19 20 21 22 23 24]]
    


```python
np.hsplit(arrsplit, 3)
```




    [array([[ 1,  2,  3,  4],
            [13, 14, 15, 16]]),
     array([[ 5,  6,  7,  8],
            [17, 18, 19, 20]]),
     array([[ 9, 10, 11, 12],
            [21, 22, 23, 24]])]




```python
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
a
```




    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])




```python
b = a.view()
b
```




    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])




```python
 c = a.copy()
 c
```




    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])




```python
a = np.array([1, 2, 3, 4])

a.sum()
```




    10




```python
b = np.array([[1, 1], [2, 2]])
b
```




    array([[1, 1],
           [2, 2]])




```python
b.sum(axis=0)
```




    array([3, 3])




```python
b.sum(axis=1)
```




    array([2, 4])




```python
data = np.array([1, 2])
data
```




    array([1, 2])




```python
ones = np.ones(2)
ones
```




    array([1., 1.])




```python
data * data
```




    array([1, 4])




```python
data / data
```




    array([1., 1.])




```python
data * 2
```




    array([2, 4])




```python
data + ones
```




    array([2., 3.])



# Numpy Vector


```python
import numpy as np
a = np.random.rand(5)
```


```python
print(a)
```

    [0.58879191 0.15161498 0.77609351 0.08861043 0.27267114]
    


```python
a.shape
```




    (5,)




```python
print(a.T)
```

    [0.58879191 0.15161498 0.77609351 0.08861043 0.27267114]
    


```python
print(np.dot(a, a.T))
```

    1.0541855054564548
    


```python
a = np.random.rand(5,1)

print(a)
```

    [[0.3456531 ]
     [0.73698062]
     [0.6975975 ]
     [0.1515631 ]
     [0.4871205 ]]
    


```python
print(a.T)
```

    [[0.3456531  0.73698062 0.6975975  0.1515631  0.4871205 ]]
    


```python
print(np.dot(a, a.T))
```

    [[0.11947606 0.25473963 0.24112674 0.05238826 0.16837471]
     [0.25473963 0.54314043 0.51411584 0.11169907 0.35899837]
     [0.24112674 0.51411584 0.48664228 0.10573004 0.33981405]
     [0.05238826 0.11169907 0.10573004 0.02297137 0.07382949]
     [0.16837471 0.35899837 0.33981405 0.07382949 0.23728639]]
    


```python
a = np.random.rand(1,5)

print(a)
```

    [[0.46786496 0.62264232 0.73158119 0.77089578 0.70695814]]
    

# Exercise

## No.1


```python
# 1 Dimensi
import numpy as np
print(np.arange(1,28,1))
```

    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25 26 27]
    


```python
# 2 Dimensi
import numpy as np
a = np.arange(1,28,1)

b = a.reshape(3,9)
print(b)
```

    [[ 1  2  3  4  5  6  7  8  9]
     [10 11 12 13 14 15 16 17 18]
     [19 20 21 22 23 24 25 26 27]]
    


```python
# 3 Dimensi
array_example = np.array([[[ 1, 2, 3],
                           [ 4, 5, 6],
                            [ 7, 8, 9]],
                          
                            [[10, 11, 12,],
                             [13, 14, 15],
                             [16, 17, 18]],
                          
                          [[19, 20, 21],
                           [22, 23, 24],
                           [25, 26, 27]]])

print(array_example)
```

    [[[ 1  2  3]
      [ 4  5  6]
      [ 7  8  9]]
    
     [[10 11 12]
      [13 14 15]
      [16 17 18]]
    
     [[19 20 21]
      [22 23 24]
      [25 26 27]]]
    


```python
# Another way of no.1
mulai = int(input('Mulai: '))
selesai = int(input('Selesai: '))

soal1 = np.arange(mulai,selesai,1)
print(soal1)

dimensi2 = soal1.reshape((3,9))
print(dimensi2)

dimensi3 = soal1.reshape((3,3,3))
print(dimensi3)
```

    Mulai: 1
    Selesai: 28
    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
     25 26 27]
    [[ 1  2  3  4  5  6  7  8  9]
     [10 11 12 13 14 15 16 17 18]
     [19 20 21 22 23 24 25 26 27]]
    [[[ 1  2  3]
      [ 4  5  6]
      [ 7  8  9]]
    
     [[10 11 12]
      [13 14 15]
      [16 17 18]]
    
     [[19 20 21]
      [22 23 24]
      [25 26 27]]]
    

## No.2


```python
import numpy as np
batas = int(input('Batas data = '))
mulai = int(input('Mulai dari = '))
selesai = int(input('Sampai dengan ='))

d = np.random.randint(mulai, selesai, size=(1,batas))
print(d)
```

    Batas data = 16
    Mulai dari = 1
    Sampai dengan =50
    [[32 49 11 39 40 41 35 32 11 37 15 37 31 29 23  6]]
    


```python
a = d.reshape(4,4)
print(a)
```

    [[32 49 11 39]
     [40 41 35 32]
     [11 37 15 37]
     [31 29 23  6]]
    


```python
b = a[:, ::-1]
print(b)
```

    [[39 11 49 32]
     [32 35 41 40]
     [37 15 37 11]
     [ 6 23 29 31]]
    

## No.3


```python
np.ones(10)
```




    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])




```python
np.zeros(10)
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])




```python
arr = np.ones((10,10))*4

arr[1:9,1:9]=0

print(arr)
```

    [[4. 4. 4. 4. 4. 4. 4. 4. 4. 4.]
     [4. 0. 0. 0. 0. 0. 0. 0. 0. 4.]
     [4. 0. 0. 0. 0. 0. 0. 0. 0. 4.]
     [4. 0. 0. 0. 0. 0. 0. 0. 0. 4.]
     [4. 0. 0. 0. 0. 0. 0. 0. 0. 4.]
     [4. 0. 0. 0. 0. 0. 0. 0. 0. 4.]
     [4. 0. 0. 0. 0. 0. 0. 0. 0. 4.]
     [4. 0. 0. 0. 0. 0. 0. 0. 0. 4.]
     [4. 0. 0. 0. 0. 0. 0. 0. 0. 4.]
     [4. 4. 4. 4. 4. 4. 4. 4. 4. 4.]]
    
