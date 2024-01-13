#!/usr/bin/env python
# coding: utf-8

# # NumPy

# In[2]:


import numpy as np


# In[3]:


a = np.array([1, 2, 3])

print(a)


# In[4]:


b = [1, 2, 3]

print(b)


# In[6]:


type(a)


# In[7]:


type(b)


# In[8]:


np.zeros(6)


# In[9]:


np.empty(6)


# In[10]:


print(np.arange(4))
print(np.arange(0,10,2))


# In[11]:


np.arange(2,29,5)


# In[17]:


arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])


# In[18]:


np.append(arr, [1,2])


# In[19]:


np.delete(arr, 1)


# In[20]:


np.sort(arr)


# In[21]:


array_example = np.array([[[0, 1, 2, 3],
                            [4, 5, 6, 7]],
                          
                            [[0, 1, 2, 4],
                             [4, 5, 6, 7]],
                          
                          [[0, 1, 2, 3],
                           [4, 5, 6, 7]]])

print(array_example)


# In[22]:


array_example.ndim


# In[23]:


array_example.size


# In[24]:


array_example.shape


# In[25]:


arr_one = np.array([[1, 2, 3, 4, 5]])


# In[26]:


arr_one.ndim


# In[27]:


arr_one.size


# In[28]:


arr_one.shape


# In[29]:


a = np.arange(6)

print(a)


# In[30]:


b = a.reshape(3,2)

print(b)


# In[31]:


a.reshape(6,1)


# In[32]:


a = np.array([1, 2, 3, 4, 5, 6])
a.shape


# In[33]:


a2 = a[np.newaxis]
print(a2.shape)
print(a2)


# In[34]:


row_vector = a[np.newaxis, :]
print(row_vector.shape)
print(row_vector)


# In[35]:


col_vector = a[:, np.newaxis]
print(col_vector.shape)
print(col_vector)


# In[36]:


a = np.array([1, 2, 3, 4, 5, 6])
a.shape


# In[37]:


b = np.expand_dims(a, axis=1)
b.shape


# In[38]:


c = np.expand_dims(a, axis=0)
c.shape


# In[39]:


data = np.array([1,2,3])

print(data)
print(data[0])
print(data[1])
print(data[0:2])
print(data[1:])
print(data[-2:])


# In[40]:


a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
a


# In[41]:


print(a[a>=5])


# In[42]:


five_up = (a >= 5)

print(a[five_up])
print(a[a>=5])


# In[43]:


divisible_by_2 = a[a%2==0]

print(divisible_by_2)


# In[44]:


c = a[(a > 2) & (a < 11)]

print(c)


# In[45]:


arr = np.array([ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

arr1 = arr[3:8]
arr1


# In[46]:


a_1 = np.array([[1, 1],
        [2, 2]])


# In[47]:


a_2 = np.array([[3, 3],
        [4, 4]])


# In[48]:


np.vstack((a_1, a_2))


# In[49]:


np.hstack((a_1, a_2))


# In[50]:


a = np.arange(1, 25, 1)

arrsplit = a.reshape(2, 12)
print(arrsplit)


# In[51]:


np.hsplit(arrsplit, 3)


# In[52]:


a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
a


# In[53]:


b = a.view()
b


# In[54]:


c = a.copy()
c


# In[55]:


a = np.array([1, 2, 3, 4])

a.sum()


# In[56]:


b = np.array([[1, 1], [2, 2]])
b


# In[57]:


b.sum(axis=0)


# In[58]:


b.sum(axis=1)


# In[59]:


data = np.array([1, 2])
data


# In[60]:


ones = np.ones(2)
ones


# In[61]:


data * data


# In[62]:


data / data


# In[63]:


data * 2


# In[64]:


data + ones


# # Numpy Vector

# In[65]:


import numpy as np
a = np.random.rand(5)


# In[66]:


print(a)


# In[67]:


a.shape


# In[68]:


print(a.T)


# In[69]:


print(np.dot(a, a.T))


# In[70]:


a = np.random.rand(5,1)

print(a)


# In[71]:


print(a.T)


# In[72]:


print(np.dot(a, a.T))


# In[73]:


a = np.random.rand(1,5)

print(a)


# # Exercise

# ## No.1

# In[74]:


# 1 Dimensi
import numpy as np
print(np.arange(1,28,1))


# In[75]:


# 2 Dimensi
import numpy as np
a = np.arange(1,28,1)

b = a.reshape(3,9)
print(b)


# In[76]:


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


# In[77]:


# Another way of no.1
mulai = int(input('Mulai: '))
selesai = int(input('Selesai: '))

soal1 = np.arange(mulai,selesai,1)
print(soal1)

dimensi2 = soal1.reshape((3,9))
print(dimensi2)

dimensi3 = soal1.reshape((3,3,3))
print(dimensi3)


# ## No.2

# In[78]:


import numpy as np
batas = int(input('Batas data = '))
mulai = int(input('Mulai dari = '))
selesai = int(input('Sampai dengan ='))

d = np.random.randint(mulai, selesai, size=(1,batas))
print(d)


# In[79]:


a = d.reshape(4,4)
print(a)


# In[80]:


b = a[:, ::-1]
print(b)


# ## No.3

# In[81]:


np.ones(10)


# In[82]:


np.zeros(10)


# In[83]:


arr = np.ones((10,10))*4

arr[1:9,1:9]=0

print(arr)

