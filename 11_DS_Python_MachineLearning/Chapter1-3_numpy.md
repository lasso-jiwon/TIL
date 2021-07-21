# 넘파이


```python
import numpy
import pandas
import matplotlib.pyplot
import seaborn
from sklearn.model_selection import train_test_split
```

___
## 넘파이 ndarray 개요


```python
import numpy as np
```

* as np를 추가해 약어로 모듈을 표현해주는 게 관례임  
* 넘파이의 기반 데이터 타입은 **`ndarray`**임  
* ndarray를 이용해 넘파이에서 다차원 배열을 쉽게 생성하고 다양한 연산을 수행할 수 있음  

넘파이 `array()` 함수는 파이썬의 리스트와 같은 다양한 인자를 입력받아 `ndarray`로 변환하는 기능을 수행함  
생성된 `ndarray` 배열의 **`shape` 변수**는 `ndarray`의 크기, 즉 행과 열의 수를 튜플 형태로 가지고 있으며  
이를 통해 `ndarray` 배열의 차원까지 알 수 있음  

### ndarray.shape
ndarray의 차원과 크기를 튜플 형태로 나타내줌


```python
# 1차원 array : 3개의 데이터
array1 = np.array([1, 2, 3])
print('array1 type', type(array1))
print('array1 array 형태', array1.shape)
```

    array1 type <class 'numpy.ndarray'>
    array1 array 형태 (3,)



```python
# 2차원 array : 2개의 로우와 3개의 칼럼
array2 = np.array([[1, 2, 3],
                   [2, 3, 4]])
print('array2 type', type(array2))
print('array2 array 형태', array2.shape)
```

    array2 type <class 'numpy.ndarray'>
    array2 array 형태 (2, 3)



```python
# 2차원 데이터 : 1개의 로우와 3개의 칼럼
array3 = np.array([[1, 2, 3]])
print('array3 type', type(array3))
print('array3 array 형태', array3.shape)
```

    array3 type <class 'numpy.ndarray'>
    array3 array 형태 (1, 3)


>array1은 1차원 데이터임을 (3,)로 표현함  
array3은 2차원 데이터임을 (1, 3)으로 표현함

### ndarray.ndim
array의 차원 확인하기


```python
print('array1 : {:0}차원, array2 : {:1}차원, array3 : {:2}차원'.format(array1.ndim, array2.ndim, array3.ndim))
```

    array1 : 1차원, array2 : 2차원, array3 :  2차원


___
## ndarray의 데이터 타입

* ndarray 내의 데이터 값은 숫자 값, 문자열 값, 불 값 등이 모두 가능함  
* 숫자형의 경우 int형, unsigned int형, float형 그리고 complex타입도 제공함  
* ndarray 내의 데이터 타입은 그 연산의 특성상 같은 데이터 타입만 가능함  
  즉 한 개의 ndarray객체에 int 와 float가 함께 있을 수 없음  
* ndarray내의 데이터 타입은 dtype 속성으로 확인 가능


```python
list1 = [1, 2, 3]
print(type(list1))
array1 = np.array(list1)
print(type(array1))
print(array1, array1.dtype)
```

    <class 'list'>
    <class 'numpy.ndarray'>
    [1 2 3] int64


리스트는 서로 다른 데이터 타입을 가질 수 있음  
ndarray는 같은 데이터 타입만 가능함  
만약 다른 데이터 유형이 섞여있는 리스트를 ndarray로 변경하면 데이터 크기가 더 큰 데이터 타입으로 형 변환 일괄 적용함


```python
# int와 str이 함께 있는 리스트를 ndarray로 변경
list2 = [1, 2, '뇽바라기']
array2 = np.array(list2)
print(array2, array2.dtype)

list3 = [1, 2, 3.0]
array3 = np.array(list3)
print(array3, array3.dtype)
```

    ['1' '2' '뇽바라기'] <U21
    [1. 2. 3.] float64


array2는 정수 1, 2가 모두 문자로 변환됨  
array3은 정수 1, 2가 실수형으로 변환됨 

### astype() : ndarray 데이터 값의 타입 변경

astype()에 인자로 원하는 타입을 문자열로 지정함  
이 경우는 대용량 데이터의 ndarray를 만들 때 많은 메모리가 사용되는데, 메모리를 절약할 때 이용함  
예) int형으로 충분한 경우인데, 데이터 타입이 float라면 int 형으로 바꿔서 메모리를 절약


```python
array_int = np.array([1, 2, 3])
array_float = array_int.astype('float64')
print(array_int, array_int.dtype, '->', array_float, array_float.dtype)

array_int1 = array_float.astype('int32')
print(array_float, array_float.dtype, '->', array_int1, array_int1.dtype)

array_float1 = np.array([1.1, 2.1, 3.1])
array_int2 = array_float1.astype('int32')
print(array_float1, array_float1.dtype, '->', array_int2, array_int2.dtype)
```

    [1 2 3] int64 -> [1. 2. 3.] float64
    [1. 2. 3.] float64 -> [1 2 3] int32
    [1.1 2.1 3.1] float64 -> [1 2 3] int32


___
## ndarray를 편리하게 생성하기 : arange, zeros, ones

특정 크기와 차원을 가진 ndarray를 연속값이나 0또는 1로 초기화해 쉽게 생성해야 할 필요가 있는 경우가 발생함  
이 경우 `arange()`, `zeros()`, `ones()`를 이용해 쉽게 ndarray를 생성할 수 있음  
주로 테스트용 데이터를 만들거나 대규모의 데이터를 일괄 초기화 할 때 사용  

### arange()

`arrange()`는 함수 이름에서 알 수 있듯이 파이썬 표준 함수인 `range()`와 유사한 기능을 함  
0부터 함수 인자 값 -1까지의 값을 순차적으로 ndarray의 데이터 값으로 변환함


```python
sequence_array = np.arange(10)
print(sequence_array)
print(sequence_array.dtype, sequence_array.shape)
```

    [0 1 2 3 4 5 6 7 8 9]
    int64 (10,)


default 함수 인자는 stop 값이며,  
0부터 stop값(10-1) 9까지의 연속 숫자 값으로 구성된 1차원의 ndarray를 만들어 줌  
start 값도 부여해서 0이 아닌 다른 값부터 시작할 수도 있음  

### zeros() 와 ones()

`zeros()`는 함수 인자로 튜플 형태의 shape 값을 입력하면  
**모든 값을 0**으로 채운 해당 shape를 가진 ndarray를 반환함  

유사하게 `ones()`는 함수 인자로 튜플 형태의 shape값을 입력하면  
**모든 값을 1**로 채운 해당 shape를 가진 ndarray로 반환함  

함수 인자로 dtype를 정해주지 않으면 디폴트로 float64형의 데이터로 ndarray를 채움


```python
zero_array = np.zeros((3, 2), dtype = 'int32')
print(zero_array)
print(zero_array.dtype, zero_array.shape)
print()
one_array = np.ones((3, 2))
print(one_array)
print(one_array.dtype, one_array.shape)
```

    [[0 0]
     [0 0]
     [0 0]]
    int32 (3, 2)
    
    [[1. 1.]
     [1. 1.]
     [1. 1.]]
    float64 (3, 2)


___
## ndarray의 차원과 크기를 변경하는 reshape()

`reshape()` 에서 메서드는 ndarray를 특정 차원 및 크기로 변환함  
다음 예제는 0~9까지의 1차원 ndarray를 2로우x5칼럼과 5로우x2칼럼 형태로 2차원 ndarray로 변환함


```python
array1 = np.arange(10)
print('array1:\n', array1)

array2 = array1.reshape(2, 5)
print('array2:\n', array2)

array3 = array1.reshape(5, 2)
print('array3:\n', array3)
```

    array1:
     [0 1 2 3 4 5 6 7 8 9]
    array2:
     [[0 1 2 3 4]
     [5 6 7 8 9]]
    array3:
     [[0 1]
     [2 3]
     [4 5]
     [6 7]
     [8 9]]


`reshape()`는 지정된 사이즈로 변경 불가능하면 오류를 발생함  
가령 (10,)인 데이터를 (4, 3)형태로 변경할 수 없음!


```python
array1.reshape(3, 4)
```


    ---------------------------------------------------------------------------
    
    ValueError                                Traceback (most recent call last)
    
    <ipython-input-18-8f800396e055> in <module>
    ----> 1 array1.reshape(3, 4)


    ValueError: cannot reshape array of size 10 into shape (3,4)


`reshape()`를 실전에서 더욱 효율적으로 사용하는 경우는 아마도 인자를 -1로 적용하는 경우임  
**-1을 인자로 사용하면 원래 ndarray와 호환되는 새로운 shape로 변환해줌**  

아래 예제에 reshape()에 -1값을 인자로 적용한 경우에 어떻게 ndarray의 size를 변경할 수 있는지 알아보자


```python
array1 = np.arange(10)
print(array1)

array2 = array1.reshape(-1, 5)
print('array2.shape :', array2.shape)

array3 = array1.reshape(5, -1)
print('array3.shape :', array3.shape)
```

    [0 1 2 3 4 5 6 7 8 9]
    array2.shape : (2, 5)
    array3.shape : (5, 2)



```python
array1 = np.arange(10)
array4 = array1.reshape(-1, 4)
```


    ---------------------------------------------------------------------------
    
    ValueError                                Traceback (most recent call last)
    
    <ipython-input-21-474c3d9b15cb> in <module>
          1 array1 = np.arange(10)
    ----> 2 array4 = array1.reshape(-1, 4)


    ValueError: cannot reshape array of size 10 into shape (4)


-1 인자는 reshape(-1, 1)와 같은 형태로 자주 사용됨  
**reshape(-1, 1)은 원본 ndarray가 어떤 형태라도 2차원이고,  
여러 개의 로우를 가지되 반드시 1개의 컬럼을 가진 ndarray로 변환됨을 보장함**  

여러 개의 넘파이 ndarray는 stack이나 concat으로 결합할 때 각각의 ndarray의 형태를 통일해 유용하게 사용됨  
다음 예제는 reshape(-1, 1)을 이용해 3차원을 2차원으로, 1차원을 2차원으로 변경함


```python
array1 = np.arange(8)
array3d = array1.reshape((2, 2, 2))
print('array1 :\n', array1.tolist())
print()
print('array3d :\n', array3d.tolist())
print()

# 3차원 ndarray를 2차원 ndarray로 변환
array5 = array3d.reshape(-1, 1)
print('array5 : array3d.reshape(-1, 1)\n', array5.tolist())
print('array5 shape:', array5.shape)
print()

# 1차원 ndarray를 2차원 ndarray로 변환
array6 = array1.reshape(-1, 1)
print('array6 : array1.reshape(-1, 1)\n', array6.tolist())
print('array6 shape:', array6.shape)
```

    array1 :
     [0, 1, 2, 3, 4, 5, 6, 7]
    
    array3d :
     [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    
    array5 : array3d.reshape(-1, 1)
     [[0], [1], [2], [3], [4], [5], [6], [7]]
    array5 shape: (8, 1)
    
    array6 : array1.reshape(-1, 1)
     [[0], [1], [2], [3], [4], [5], [6], [7]]
    array6 shape: (8, 1)


___
## 넘파이의 ndarray의 데이터 세트 선택하기 - 인덱싱(Indexing)

넘파이에서 ndarray 내의 일부 데이터 세트나 특정 데이터만을 선택할 수 있도록 하는 인덱싱에 대해 알아보자  

1. **특정한 데이터만 추출** : 원하는 위치의 인덱스 값을 지정하면 해당 위치의 데이터가 반환됨  
2. **슬라이싱(Slicing)** : 슬라이싱은 연속된 인덱스상의 ndarray를 추출하는 방식임 [1:5]는 1과 4까지 반환  
3. **팬시 인덱싱(Fancy Indexing)** : 일정한 인덱싱 집합을 리스트 또는 ndarray 형태로 지정해 해당 위치에 있는 데이터의 ndarray를 반환함  
4. **불린 인덱싱(Boolean Indexind)** : 특정 조건에 해당하는지 여부인 T/F 값 인덱싱 집합을 기반으로  
   T에 해당하는 인덱스 위치에 있는 데이터의 ndarray를 반환함

### 단일 값 추출


```python
# 1부터 9까지의 1차원 ndarray 생성
array1 = np.arange(start=1, stop=10)
print('array1:', array1)

#index는 0부터 시작하므로 array1[2]는 3번째 index 위치의 데이터값을 의미
value = array1[2]
print('value:', value)
print(type(value))
```

    array1: [1 2 3 4 5 6 7 8 9]
    value: 3
    <class 'numpy.int64'>


인덱스는 0부터 시작하므로 array1[2]는 3번째 인덱스를 의미함  
인덱스 -1은 맨 뒤의 데이터 값을 의미함  


```python
print('맨 뒤의 값:', array1[-1], '맨 뒤에서 두 번째 값:', array1[-2])
```

    맨 뒤의 값: 9 맨 뒤에서 두 번째 값: 8


단일 인덱스를 이용해 ndarray 내의 데이터도 간단히 수정 가능


```python
array1[0] = 9
array1[8] = 0
print('array1:', array1)
```

    array1: [9 2 3 4 5 6 7 8 0]


**다차원 ndarray에서 단일값 추출하기**  
3차원 이상과 2차원 추출법의 큰 차이는 없음


```python
array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3, 3)
print(array2d)

print('(row=0, col=0) index 가리키는 값 :', array2d[0, 0])
print('(row=0, col=1) index 가리키는 값 :', array2d[0, 1])
print('(row=1, col=0) index 가리키는 값 :', array2d[1, 0])
print('(row=2, col=2) index 가리키는 값 :', array2d[2, 2])
```

    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    (row=0, col=0) index 가리키는 값 : 1
    (row=0, col=1) index 가리키는 값 : 2
    (row=1, col=0) index 가리키는 값 : 4
    (row=2, col=2) index 가리키는 값 : 9


### 슬라이싱

':'기호를 이용해 연속한 데이터를 슬라이싱 해 추출할 수 있음  
단일 데이터값 추출을 제외하고 슬라이싱, 팬시 인덱싱, 불린 인덱싱으로 추출된 데이터 세트 모두 ndarray타입임  


```python
array1 = np.arange(1, 10)
array3 = array1[0:3]
print(array3)
print(type(array3))
```

    [1 2 3]
    <class 'numpy.ndarray'>


슬라이싱 기호인 ':' 사이의 시작, 종료 인덱스는 생략이 가능함  

1. ':' 기호 앞 인덱스 생략하면 자동으로 맨 처음 인덱스인 0으로 간주함  
2. ':' 기호 뒤 인덱스 생략하면 자동으로 맨 마지막 인덱스로 간주함  
3. ':' 기호 앞/뒤 모두 생략하면 맨 처음/맨 마지막 인덱스로 간주함


```python
array1 = np.arange(1, 10)
array4 = array1[:3]
print(array4)

array5 = array1[3:]
print(array5)

array6 = array1[:]
print(array6)
```

    [1 2 3]
    [4 5 6 7 8 9]
    [1 2 3 4 5 6 7 8 9]


2차원 ndarray에서 슬라이싱으로 데이터 접근하는 방법  


```python
array1d = np.arange(1, 10)
array2d = array1.reshape(3, 3)
print('array2d:\n', array2d)

print('array2d[0:2, 0:2]:\n', array2d[0:2, 0:2])
print('array2d[1:3, 0:3]:\n', array2d[1:3, 0:3])
print('array2d[1:3, :]:\n', array2d[1:3, :])
print('array2d[:, :]:\n', array2d[:, :])
print('array2d[:2, 1:]:\n', array2d[:2, 1:])
print('array2d[:2, 0]:\n', array2d[:2, 0])
```

    array2d:
     [[1 2 3]
     [4 5 6]
     [7 8 9]]
    array2d[0:2, 0:2]:
     [[1 2]
     [4 5]]
    array2d[1:3, 0:3]:
     [[4 5 6]
     [7 8 9]]
    array2d[1:3, :]:
     [[4 5 6]
     [7 8 9]]
    array2d[:, :]:
     [[1 2 3]
     [4 5 6]
     [7 8 9]]
    array2d[:2, 1:]:
     [[2 3]
     [5 6]]
    array2d[:2, 0]:
     [1 4]


2차원 ndarray에서 뒤에 오는 인덱스를 없애면 1차원 ndarray를 반환함  
즉, array2d[0]과 같이 2차원에서 뒤에 오는 인덱스를 없애면 로우 축(axis 0)의 첫 번째 로우 ndarray를 반환함  
반환되는 ndarray는 1차원임 3차원 ndarray에서 뒤에 오는 인덱스를 없애면 2차원 ndarray를 반환함  


```python
print(array2d[0])
print(array2d[1])
print('array2d[0].shape:', array2d[0].shape, 'array2d[1].shape:', array2d[1].shape)
```

    [1 2 3]
    [4 5 6]
    array2d[0].shape: (3,) array2d[1].shape: (3,)


### 팬시 인덱싱

펜시 인덱싱은 리스트나 ndarray로 인덱스 집합을 지정하면 해당 위치의 인덱스에 해당하는 ndarray를 반환하는 인덱싱 방식임  


```python
array1d = np.arange(1, 10)
array2d = array1d.reshape(3, 3)

array3 = array2d[[0, 1], 2]
print('array2d[[0, 1], 2] =>', array3.tolist())

array4 = array2d[[0, 1], 0:2]
print('array2d[[0, 1], 0:2] =>', array4.tolist())

array5 = array2d[[0, 1]]
print('array2d[[0, 1]] =>', array5.tolist())
```

    array2d[[0, 1], 2] => [3, 6]
    array2d[[0, 1], 0:2] => [[1, 2], [4, 5]]
    array2d[[0, 1]] => [[1, 2, 3], [4, 5, 6]]


### 불린 인덱싱

불린 인덱싱은 조건 필터링과 검색을 동시에 할 수 있기 때문에 매우 자주 사용되는 인덱싱 방식임  
데이터 값 >5 추출하기가 문제가 있으면 불린 인덱싱은 for loop/if else 훨씬 간단하게 구현 가능함  
ndarray의 인덱스를 지정하는 []내에 조건문을 그대로 기재하면 됨  


```python
array1d = np.arange(1, 10)
# []안에 array1d >5 Boolean indexing을 적용
array3 = array1d[array1d>5]
print('array1d > 5 불린 인덱싱 결과값 :', array3)
```

    array1d > 5 불린 인덱싱 결과값 : [6 7 8 9]


___
## 행렬의 정렬 - sort()와 argsort()

넘파이에서 행렬을 정렬하는 대표적인 방법인 np.sort()와 ndarray.sort()  
그리고 정렬된 행렬의 인덱스를 반환하는 argsort()에 대해 알아보겠음  

### 행렬 정렬 - np.sort()와 ndarray.sort()

넘파이의 행렬 정렬은 np.sort()와 같이 넘파이에서 sort()를 호출하는 방식과  
ndarray.sort()와 같이 행렬 자체에서 sort()를 호출하는 방식이 있음  
두 방식의 차이는 np.sort()의 경우 원 행렬은 그대로 유지한 채 행렬의 정렬된 행렬을 반환함  
ndarray.sort()는 원 행렬 자체를 정렬한 형태로 변환하며 변환된 값은 None임


```python
org_array = np.array([3, 1, 9, 5])
print('원본 행렬 :', org_array)

# np.sort()로 정렬
sort_array1 = np.sort(org_array)
print('np.sort() 호출 후 반환된 정렬 행렬 :', sort_array1)
print('np.sort() 호출 후 원본 행렬 :', org_array)

# ndarray.sort()로 정렬
sort_array2 = org_array.sort()
print('org_array.sort() 호출 후 반환된 행렬 :', sort_array2)
print('org_array.sort() 호출 후 원본 행렬 :', org_array)
```

    원본 행렬 : [3 1 9 5]
    np.sort() 호출 후 반환된 정렬 행렬 : [1 3 5 9]
    np.sort() 호출 후 원본 행렬 : [3 1 9 5]
    org_array.sort() 호출 후 반환된 행렬 : None
    org_array.sort() 호출 후 원본 행렬 : [1 3 5 9]


원본 행렬 [3, 1, 9, 5]에 대해서 np.sort()는 원본 행렬을 변경하지 않고 정렬된 형태로 반환함  
ndarray.sort()는 원본 행렬 자체를 정렬한 값으로 변환함  
내림차순으로 정렬하기 위해서는 [::-1]을 적용함


```python
sort_array1_desc = np.sort(org_array)[::-1]
print('내림차순으로 정렬 :', sort_array1_desc)
```

    내림차순으로 정렬 : [9 5 3 1]


행렬이 2차원 이상일 경우에 axis 축 값 설정을 통해 로우 방향, 또는 칼럼 방향으로 정렬 수행함  


```python
array2d = np.array([[8, 12],
                    [7, 1]])

sort_array2d_axis0 = np.sort(array2d, axis=0)
print('로우 방향으로 정렬 :\n', sort_array2d_axis0)

sort_array2d_axis1 = np.sort(array2d, axis=1)
print('칼럼 방향으로 정렬 :\n', sort_array2d_axis1)
```

    로우 방향으로 정렬 :
     [[ 7  1]
     [ 8 12]]
    칼럼 방향으로 정렬 :
     [[ 8 12]
     [ 1  7]]


### 정렬된 행렬의 인덱스를 반환하기 - argsort()

원본 행렬이 정렬되었을 때 기존 원본 행렬의 원소에 대한 인덱스를 필요로 할 때 np.argsort()를 이용함  
np.argsort()는 정렬 행렬의 원본 행렬 인덱스를 ndarray 형으로 반환함  


```python
org_array = np.array([3, 1, 9, 5])
sort_indices = np.argsort(org_array)
print(type(sort_indices))
print('행렬 정렬 시 원본 행렬의 인덱스 :', sort_indices)
```

    <class 'numpy.ndarray'>
    행렬 정렬 시 원본 행렬의 인덱스 : [1 0 3 2]


오름차순이 아닌 내림차순으로 정렬 시에 원본 행렬의 인덱스를 구하는 것도 np.argsort()[::-1]을 적용함  


```python
org_array = np.array([3, 1, 9, 5])
sort_indices_desc = np.argsort(org_array)[::-1]
print('행렬 내림차순 정렬 시 원본 행렬의 인덱스 :', sort_indices_desc)
```

    행렬 내림차순 정렬 시 원본 행렬의 인덱스 : [2 3 0 1]


argsort()는 넘파이에서 매우 활용도가 높음  
아래 예제로 시험 성적 순으로 이름 출력하고자 한다면 np.argsort(score_array)를 이용해  
반환된 인덱스를 name_array에 팬시 인덱스로 적용해 추출한 수 있음


```python
import numpy as np

name_array = np.array(['라이언', '어피치', '죠르디', '춘식이', '앙몬드'])
score_array = np.array([78, 95, 84, 98, 88])

sort_indices_asc = np.argsort(score_array)
print('성적 오름차순 정렬 시 score_array의 인덱스 :', sort_indices_asc)
print('성적 오름차순 정렬 시 name_array의 이름 출력 :', name_array[sort_indices_asc])
```

    성적 오름차순 정렬 시 score_array의 인덱스 : [0 2 4 1 3]
    성적 오름차순 정렬 시 name_array의 이름 출력 : ['라이언' '죠르디' '앙몬드' '어피치' '춘식이']


___
## 선형대수 연산 - 행렬 내적과 전치 행렬 구하기

넘파이는 매우 다양한 선형대수 연산을 지원함  
그 중 가장 많이 사용되면서도 기본 연산인 행렬 내적과 전치 행렬을 구한는 방법을 알아보자

### 행렬 내적 (행렬 곱)

행렬 내적은 행렬 곱이며, 두 행렬 A와 B의 내적은 np.dot()을 이용해 계산이 가능함  
행렬 내적을 넘파이 dot()을 이용해 구해보자


```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10], 
              [11, 12]])
dot_product = np.dot(A, B)
print('행렬 내적 결과 :\n', dot_product)
```

    행렬 내적 결과 :
     [[ 58  64]
     [139 154]]


### 전치 행렬

원 행렬에서 행과 열 위치를 교환한 원소로 구성한 행렬을 그 행렬의 전치행렬이라고 함  
넘파이의 transpose()를 이용해 전치 행렬을 쉽게 구할 수 있음  


```python
A = np.array([[1, 2],
              [3, 4]])
transpose_mat = np.transpose(A)
print('A의 전치 행렬 :\n', transpose_mat)
```

    A의 전치 행렬 :
     [[1 3]
     [2 4]]

