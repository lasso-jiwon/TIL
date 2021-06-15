# 요약
* min(리스트) or min(숫자, 숫자, 숫자, ... )
* max(리스트) or max(숫자, 숫자, 숫자, ... )
* sum(리스트)
* `for i in reversed(리스트) :`
* enumerate(리스트) : `for i, element in enumerate(리스트) :`
* 딕셔너리.items() : `for key, value in 딕셔너리.items() :`
___

# 리스트에 적용할 수 있는 기본 함수

* **min()** : 리스트 내부에서 최솟값을 찾습니다.
* **max()** : 리스트 내부에서 최댓값을 찾습니다.
* **sum()** : 리스트 내부에서 값을 모두 더합니다.

___

# reversed() 함수로 뒤집기

**`for i in reversed(리스트) :`**
`reversed()` 함수의 결과가 제너레이터이기 때문에 여러번 활용되지 않음

```python
#변수 선언
ex = ["요소1", "요소2", "요소3"]

# 그냥 출력
print('# 단순 출력')
print(ex)
print()

# enumerate 함수
print('# enumerate 함수 사용')
print(enumerate(ex))
print()

# list 함수 강제 변환
print('# list 함수 강제 변환')
print(list(enumerate(ex)))
print()

# for 반복문 사용
print('# for 반복문 사용')
for i, value in enumerate(ex) :
    print("{}번째 요소는 {}입니다.".format(i, value))
```
결과
```python
# 단순 출력
['요소1', '요소2', '요소3']

# enumerate 함수 사용
<enumerate object at 0x7f9be17fbac0>

# list 함수 강제 변환
[(0, '요소1'), (1, '요소2'), (2, '요소3')]

# for 반복문 사용
0번째 요소는 요소1입니다.
1번째 요소는 요소2입니다.
2번째 요소는 요소3입니다.
```

> 확장 슬라이싱
> 리스트를 뒤집는 추가적 방법
```python
>>> "안녕하세요"[::-1]
'요세하녕안'
```

___

# enumerate() 함수와 반복문 조합하기

딱 한가지 형태로만 사용됨
**`for i, element in enumerate(리스트) :`**

예제
```python
ex = ["요소1", "요소2", "요소3"]

for i, value in enumerate(ex) :
    print("{}번째 요소는 {}입니다.".format(i, value))
```
결과
```python
0번째 요소는 요소1입니다.
1번째 요소는 요소2입니다.
2번째 요소는 요소3입니다.
```
___

# 딕셔너리의 items() 함수와 반복문 조합하기

딱 한가지 형태로만 사용됨
**`for key, value in 리스트.items() :`**

예제
```python
a = {"key_1" : "value_1", "key_2" : "value_2", "key_3" : "value_3"}
for key, value in a.items() :
    print("{}키의 값은 {}입니다.".format(key, value))
```
결과
```python
key_1키의 값은 value_1입니다.
key_2키의 값은 value_2입니다.
key_3키의 값은 value_3입니다.
```
___

# 리스트 내포

* 많이 사용되는 코드 한줄로 작성하는 방법
**`리스트 이름 = [표현식 for 반복자 in 반복할 수 있는 것]`**


* 반복문을 이용해서 리스트 내포를 사용하는 방법
**`리스트 이름 = [표현식 for 반복자 in 반복할 수 있는 것 if 조건문]`**


#### ☑️ 반복문을 이용한 리스트 생성

```python
array = []

for i in range(0, 20, 2) :
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    
    array.append(i*i)
    # [1, 4, 16, 36, 64, 100, 144 ....]
    
print(array)
```
결과
```python
[0, 4, 16, 36, 64, 100, 144, 196, 256, 324]
```


#### ☑️ 위 반복문을 리스트 안에 넣어서 한줄로 작성하기
```python
array = [i * i for i in range(0, 20, 2)]

print(array)
```
결과
```python
[0, 4, 16, 36, 64, 100, 144, 196, 256, 324]
```

#### ☑️ 리스트 내포 기본 예제
```python
array_1 = [i for i in range(10)]
array_2 = [i for i in range(0, 10, 2)]
array_3 = [1 for i in range(10)]

print(array_1)
print(array_2)
print(array_3)
```
결과
```python
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[0, 2, 4, 6, 8]
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

#### ☑️ 조건문을 추가한 리스트 내포

**`리스트 이름 = [표현식 for 반복자 in 반복할 수 있는 것 if 조건문]`**

```python
array_1 = [i for i in range(10) if i % 3 == 0]

print(array_1)
```

결과

```python
[0, 3, 6, 9]
```

___

# 확인 문제

2진수, 8진수, 16진수로 변환하는 코드는 많이 사용됩니다. 다음과 같은 형태로 10진수를 변환할 수 있습니다.

 ```python
 # 10진수 - 2진수 변환
 >>> "{:b}".format(10)
 '1010'
 
 # 2진수 - 10진수 변환
 >>> int("1010", 2)
 10
 
 # 10진수 - 8진수 변환
 >>> "{:o}".format(10)
 '12'
 
 # 8진수 - 10진수 변환
>>> int("12", 8)
 10
 
# 10진수 - 16진수 변환
>>> "{:x}".format(10)
'a'

# 16진수 - 10진수 변환
>>> int("10", 16)
16
 ```


추가로 반복 가능한 객체 (문자열, 리스트, 범위 등)의 count() 함수는 다음과 같이 사용합니다.
```python
>>> "안녕안녕하세요".count("안")
2
```

이를 활용해서 1~100 사이에 있는 숫자 중 2진수로 변환했을 때 0이 하나만 포함된 숫자를 찾고, 그 숫자들의 합을 구하는 코드를 만들어 보세요.

**기본 구현**
```python
output = 0

for i in range(1, 100+1) :
# i를 1부터 100까지 반복한다
    if "{:b}".format(i).count("0") == 1 :
    # 만약 이진수로 바꾼 i의 값에 0이 1개만 포함되어 있다면
        print("{} : {:b}".format(i, i))
        # 10진수와 2진수 값을 각각 출력해라
        output += i
print("합계 : {}".format(output))
```

**리스트 내포를 활용한 구현**
```python
output = [i for i in range(1, 100+1) if "{:b}".format(i).count("0") == 1]
for i in output :
    print("{} : {}".format(i, "{:b}".format(i)))
print("합계 : {}".format(sum(output)))
```

결과
```python
2 : 10
5 : 101
6 : 110
11 : 1011
13 : 1101
14 : 1110
23 : 10111
27 : 11011
29 : 11101
30 : 11110
47 : 101111
55 : 110111
59 : 111011
61 : 111101
62 : 111110
95 : 1011111
합계 : 539
```