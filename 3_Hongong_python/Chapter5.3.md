# 요약
* **튜플**은 리스트와 비슷하지만, 요소를 수정할 수 없는 파이썬의 특별한 문법입니다. 괄호를 생략해서 다양하게 활용할 수 있습니다.

* **람다**는 함수를 짧게 쓸 수 있는 파이썬의 특별한 문법입니다.

* **with 구문**은 블록을 벗어날 때 close()함수를 자동으로 호출해 주는 구문입니다.

___

# 튜플

튜플이 리스트와 다른 기본적인 부분
* 대괄호가 아니라 소괄호로 선언
* 한 번 선언하면 값을 바꿀 수 없음
* 괄호가 없어도 튜플로 인식될 수 있다면 튜플

여러 개의 값을 넣는 경우
`(데이터, 데이터, 데이터, ...)`

하나의 값을 넣는 경우 (,가 붙는다!)
`(데이터,)`

___

# 튜플을 사용하는 경우

## (1) 복합 할당
```python
[a, b] = [10, 20]
(c, d) = (30, 40)

print(a, b, c, d)
10, 20, 30, 40
```

#### 괄호가 없는 튜플
괄호를 생략해도 튜플로 인식할 수 있는 경우는 괄호를 생략해도 됨 아래 예제로 확인하기
```python
tuple_test = 1, 2, 3, 4
print(tuple_test)

print()

a, b, c = 5, 6, 7
print(a)
print(b)
print(c)

## 결과
(1, 2, 3, 4)

5
6
7
```


## (2) 스왑

a 와 b 에 들어간 값 교체하기

```python
a, b = 10, 20

print(a, b)  # 10 20
a, b = b, a
print(a, b)  # 20 10

## 결과
10 20
20 10
```

## (3) 튜플을 리턴하는 함수

몫과 나머지를 리턴하는 방법
```python
a, b = 97, 40
print(a // b)
print(a%b)
2
17
```

#### divmod 함수를 이용하여 몫과 나머지를 리턴할 수 있다
divmod 함수는 튜플을 이용한 함수임


```python
a, b = 97, 40
print(divmod(a, b))
print(type(divmod(a, b)))
(2, 17)
<class 'tuple'>
```

#### 튜플을 리턴하는 함수 만들기

```python
def test() :
    return 10, 20

a, b = test()
print(a, b)
10 20
```

___

> 더 알아보기

딕셔너리의 키로 올 수 있는것 : 숫자, 문자열, 불, 튜플!!! (리스트는 안됨)

```python
a = {
    숫자 : O
    문자열 : O
    불 : O
    튜플 : O
    리스트 : X
}
```

예를들어 좌표를 나타낸다고 할 때
```python
a = {
    (0, 0) : 10,
    (0, 1) : 20,
    (1, 0) : 30,
    (1, 1) : 40
}

print(a[(0,0)])
print(a[0,0])   # 튜플이라 ()생략 가능
10
10
```
___

# 콜백함수와 람다 filter()/map()

## 함수의 매개변수로 함수 전달하기
```python
# 이 함수는 매개변수를 함수로 받을 것을 가정하고 있다. 여기서 func가 함수임!
def call_10_times(func) :   # func가 함수로 들어올 것을 가정함
    for i in range(10) :
        func()
        
def print_hello() :
    print("안녕하세요")
    
call_10_times(print_hello)

## 결과
안녕하세요
안녕하세요
안녕하세요
안녕하세요
안녕하세요
안녕하세요
안녕하세요
안녕하세요
안녕하세요
안녕하세요
```

## 콜백함수의 매개변수
**콜백함수** : 내가 함수를 호출하는 것이 아니라 다른 어떤 함수에서 호출되는 함수를 콜백함수라 부른다.


```python
def call_10_times(func) :  # 2) 이 func에 print_hello가 들어오니까
    for i in range(10) :
        func(i)   # 1) 매개변수로 나는 i를 전달하겠다 하면
        
def print_hello(number) :  # 3) number을 매개변수로 받게 되면 number로 i 가 전달되어 들어오기 때문에 
    print("안녕하세요", number)  # 4) 이와 같이 함께 활용할 수 있게 됨
    
call_10_times(print_hello)

## 결과
안녕하세요 0
안녕하세요 1
안녕하세요 2
안녕하세요 3
안녕하세요 4
안녕하세요 5
안녕하세요 6
안녕하세요 7
안녕하세요 8
안녕하세요 9
```

## 람다

**람다** : 매개변수로 함수를 전달하기 위해 함수 구문을 작성하는 것이 번거롭고, 코드 낭비라는 생각이 들 때 함수를 간단하고 쉽게 선언하는 방법 = 짧게 쓸 수 있는 문법

'콜백함수명 :' 뒤에 한줄에 코드가 오게 되며, 이 코드는 자동으로 리턴해주기 때문에 return을 쓸 필요 없음

`함수명(lambda 콜백함수명 : print(“안녕하세요", 콜백함수명))`
ex) call_10_times(lambda number : print(“안녕하세요”, number))


```python
def call_10_times(func) : 
    for i in range(10) :
        # 콜백함수(callback)
        func(i)
    
call_10_times(lambda number : print("안녕하세요", number))


## 결과
안녕하세요 0
안녕하세요 1
안녕하세요 2
안녕하세요 3
안녕하세요 4
안녕하세요 5
안녕하세요 6
안녕하세요 7
안녕하세요 8
안녕하세요 9
```

## filter() 함수와 map() 함수

함수를 매개변수로 전달하는 대표적인 **표준함수**로 map()함수와 filter() 함수가 있습니다.

map()함수는 리스트의 요소를 함수에 넣고 리턴된 값으로 새로운 리스트를 구성해주는 함수입니다.
`map(함수, 리스트)`

filter() 함수는 리스트의 요소를 함수에 넣고 리턴된 값이 True인 것으로, 새로운 리스트를 구성해주는 함수입니다.
`filter(함수, 리스트)`

#### filter() 함수

**`filter(함수, 리스트 (=반복요소))`**
리스트의 요소를 함수에 넣고 리턴된 값이 True 인 것으로 새로운 리스트를 구성해주는 함수

```python
def 짝수만(number) :
    return number % 2 == 0   # 리턴 값이 불인 것을 사용해야함
     
a = list(range(100))
b = filter(짝수만, a)

print(list(b))
    
## 결과
[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98]
```

위 코드를 **람다**를 이용해서 바꾸기

`함수명(lambda 콜백함수명 : 한줄의 코드, 리스트 )`
filter(lambda number : number % 2 == 0, a)

```python
a = list(range(100))
b = filter(lambda number : number % 2 == 0, a)
print(list(b))


# 이 함수를 리스트 내포로 설정했을 때
a = list(range(100))
print([i*i for i in a if i % 2 == 0])  #이렇게 쓸 수 있음!
print(list(b))

## 결과
[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98]
```

다른 방법!

```python
짝수만 = lambda number : number % 2 == 0

a = list(range(100))
b = filter(짝수만, a)
print(list(b))

## 결과
[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98]
```


#### map() 함수

**`map(함수, 리스트(=반복요소))`**
기존의 리스트를 기반으로 새로운 리스트를 만들 때 사용한다.

```python
def 제곱(number) :
    return number * number

a = list(range(10))
print(list(map(제곱, a)))


## 결과
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

**람다**를 이용해서 한줄로 간결하게 작성하기

```python
a = list(range(10))
print(list(map(lambda number : number * number, a)))
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

#### 인라인 람다 작성하기

* 람다
```python
# 함수 선언
power = lambda x : x*x
under_3 = lambda x : x<3

# 변수 선언
list_a = [1, 2, 3, 4, 5]

# map 함수 사용
output_a = map(power, list_a)
print(list(output_a))

# filter 함수 사용
output_b = filter(under_3, list_a)
print(list(output_b))

## 결과
[1, 4, 9, 16, 25]
[1, 2]
```


* 인라인 람다
```python
list_a = [1, 2, 3, 4, 5]

output_a = map(lambda x : x*x, list_a)
print(list(output_a))

output_b = filter(lambda x : x<3, list_a)
print(list(output_b))

## 결과
[1, 4, 9, 16, 25]
[1, 2]
```
___

## 리스트 내포와 비교하기

많이 사용되는 코드 한줄로 작성하는 방법
`리스트 이름 = [표현식 for 반복자 in 반복할 수 있는 것]`

반복문을 이용해서 리스트 내포를 사용하는 방법
`리스트 이름 = [표현식 for 반복자 in 반복할 수 있는 것 if 조건문]`

```python
a = list(range(100))
print(list(map(lambda number : number * number, a)))
print([i * i for i in a])  # 리스트 내포 이용! 위 map() 함수와 결과가 같음


print([i*i for i in a if i % 2 == 0])  #짝수만이랑 같음
```

리스트 내포를 사용하는 경우, 결과로 리스트가 나옴
즉 그만큼의 리스트가 하나 더 복제되어 메모리를 찾이 한다는 의미

map(), filter() 함수 등은 제너레이터(generator) 함수라서 내부의 데이터가 실제로 메모리에 용량을 차지하는 것들이 아님
호출되기 전까지는 가상의 값만 가지고 있기 때문임
최근에는 리스트 내포를 더 많이 사용 하는 편임!

___

# 파일처리

## 다루는 파일의 큰 구분

* 어떤 대상을 처리할 것인가?
  - 텍스트 파일 : 텍스트에디터로 열 수 있는 파일
  - 바이너리 파일 : 텍스트에디터로 열 수 없는 파일 (이미지, 동영상, 워드, 엑셀, pdf 등)


* 어떻게 다룰 것인가?
  - 쓰기
  새로 작성 (write) : w
  있는 파일 뒤에 (append) : a
  - 읽기 (read) : r
  
  | 모드 | 설명                                |
  | ---- | ----------------------------------- |
  | w    | write 모드 (새로 쓰기 모드)         |
  | a    | append 모드 (뒤에 이어서 쓰기 모드) |
  | r    | read 모드 (읽기 모드)               |

## 파일 열고 닫기

파일을 열 때는 **open()** 함수를 사용합니다.
`파일 객체 = open(문자열 : 파일 경로, 문자열 : 읽기 모드)`

파일을 닫을 때는 **close()** 함수를 사용합니다.
`파일 객체.close()`

* 파일 새로 작성하기 (w)
```python
file = open("test.txt", "w")
file.write("안녕하세요.")
file.close()

안녕하세요.
```

* 있는 파일 뒤에 추가하기 (a)
```python
file = open("test.txt", "a")
file.write("안녕하세요.")
file.close()

안녕하세요.안녕하세요.
```

* 파일 읽어오기 (r)
```python
file = open("test.txt", "r")
print(file.read())
file.close()

안녕하세요.안녕하세요.
```


## with 구문
open과 close가 with 라는 구문에서 자동으로 호출하게 됨


* with 구문 사용 예시 1
```python
with open("test.txt","a") as file :
    file.write("안녕하세요.")
```
위 코드와 아래 코드는 완전히 동일한 코드이다.
```python
file = open("test.txt", "a")
file.write("안녕하세요.")
file.close()
```


* with 구문 사용 예시 2
```python
with open("test.txt","r") as file :
    print(file.read())
```

```python
file = open("test.txt", "a")
print(file.read())
file.close()
```
마찬가지로 위 두 코드도 동일한 코드이다.


## 텍스트 한 줄씩 읽기
이름, 키, 몸무게 데이터가 있다고 가정하고 만든 후, 파일을 한 줄씩 읽어 들여 처리하는 방법을 알아보자

#### 랜덤하게 1000명의 키와 몸무게 만들기
```python
# 랜덤한 숫자를 만들기 위해 가져옵니다.
import random

# 간단한 한글 리스트를 만듭니다.
hanguls = list("가나다라마바사아자차카타파하")

# 파일을 쓰기 모드로 엽니다.
with open("info.txt", "w") as file :
    for i in range(1000) :
        # 랜덤한 값으로 변수를 생성합니다.
        name = random.choice(hanguls) + random.choice(hanguls)
        weight = random.randrange(40, 100)
        height = random.randrange(140, 200)
        # 텍스트를 씁니다.
        file.write("{},{},{}/n".format(name,weight,height))
```

데이터를 한 줄씩 읽어들이는 때는 for 반복문을 다음과 같은 형태로 사용합니다.
```python
for 한줄을 나타내는 문자열 in 파일 객체 :
    처리
```

위 info.txt 데이터를 한줄씩 읽으면서 키와 몸무게로 BMI(비만도)를 계산해 봅시다.


#### 반복문으로 파일 한 줄씩 읽기
```python
with open("info.txt", "r") as file:
    for line in file :
        # 변수를 선언합니다.
        (name, weight, height) = line.strip().split(" , ")
        
        # 데이터에 문제 없는지 확인합니다. 문제가 있으면 지나감
        if (not name) or (not weight) or (not height) :
            continue
        # 결과를 계산합니다.
        bmi = int(weight) / ((int(height)/100) ** 2)
        result = ""
        if 25 <= bmi :
            result = "과체중"
        elif 18.5 <= bmi :
            result = "정상 체중"
        else :
            result = "저체중"
            
        # 출력합니다.
        print('\n'.join([
            "이름 : {}",
            "몸무게 : {}",
            "키 : {}",
            "bmi : {}",
            "결과 : {}"
        ]).format(name, weight, height, bmi, result))
        print()
```

___

# 제너레이터

간단하게 "이터레이터를 직접 만들 때 사용하는 구문"
함수 내부에 yield라는 키워드가 포함되면 해당 함수는 제너레이터가 됩니다.

* 일반적인 함수
```python
def 함수() :
    print("출력A")
    print("출력B")
    
함수()
출력A
출력B
```

* 제너레이터 함수 만들기
```python
def 함수() :
    print("출력A")
    print("출력B")
    yield
    
제너레이터 = 함수()
print(제너레이터)
<generator object 함수 at 0x7f8361938580>
```

* 제너레이터 함수를 출력하는 방법
```python
def 함수() :
    print("출력A")
    print("출력B")
    yield
    
제너레이터 = 함수()
next(제너레이터)
출력A
출력B
```

* yield 키워드로 값 전달하기
```python
def 함수() :
    print("출력A")
    print("출력B")
    yield 100   # 리턴처럼 어떤 값을 리턴함
    
제너레이터 = 함수()

값 = next(제너레이터)
print(값)
출력A
출력B
100
```

* yield 키워드로 양보하기

yield 키워드는 함수 내부에 여러번을 넣을 수 있음!

```python
def 함수() :
    print("출력A")
    yield 100       # 멈춤!
    print("출력B")
    yield 200
    print("출력C")
    yield 300
    print("출력D")
    yield 400   # 리턴처럼 어떤 값을 리턴함
    
제너레이터 = 함수()

값 = next(제너레이터)
print(값)
```
next를 실행하는 순간 함수의 위쪽부터 아래쪽으로 코드가 진행되게 됨
print("출력A") 를 만나면 출력을 하게 되는데 아래
yield를 만나는 순간 '내 이후의 것들은 실행 안하고 양보할게!' 라는 의미

* for 반복문과 조합하기

```python
def 함수() :
    print("출력A")
    yield 100
    print("출력B")
    yield 200
    print("출력C")
    yield 300
    print("출력D")
    yield 400   # 리턴처럼 어떤 값을 리턴함
    
제너레이터 = 함수()
for i in 제너레이터 :
    print(i)
    
제너레이터 = 함수()
for i in 제너레이터 :
    print(i)

## 결과 - 한번만 출력 됨 = 1회용 함수
출력A
100
출력B
200
출력C
300
출력D
400
```

* reversed() 함수 구현
```python
def 반전(리스트):
    for i in range(len(리스트)):
        yield 리스트[- i - 1]

for i in 반전([1, 2, 3, 4, 5]):
    print(i)
```

리스트가 만들어지지 않고 제너레이터만 됨
함수를 호출하는 순간에 내부 코드가 실행되는 것이 아니고 반복문 등에서 활용할 때 내부 코드가 실행되므로 지연 로드(Lazy Load) 등을 할 때도 활용되기도 함

예를 들어 Scrapy 프레임워크에서 yield 키워드를 쓰라고 하는데, 몰라도 사용 잘함 = 프레임 워크가 쓰라고 하면 쓰면 됨

___

# 확인문제

#### 1번 실행결과처럼 출력하시오

```python
# 실행 결과
# 1::2::3::4::5::6
numbers = [1, 2, 3, 4, 5, 6]
print("::".join(numbers))   # str을 예상했는데 int가 와서 오류

# 결과
TypeError: sequence item 0: expected str instance, int found


# map 함수를 사용해서 str로 새로운 리스트 만들어주기!
numbers = [1, 2, 3, 4, 5, 6]
print("::".join(map(str, numbers))) 

# 결과
1::2::3::4::5::6
```

#### 2번 다음 코드의 빈칸을 채워서 실행결과처럼 출력하시오

```python
numbers = list(range(1, 10+1))

print("# 홀수만 추출하기")
print(list(filter(lambda x : x % 2 == 1, numbers)))
print()

print("# 3 이상, 7 미만 추출하기")
print(list(filter(lambda y : 3 <= y < 7, numbers)))
print()

print("# 제곱해서 50 미만 추출하기")
print(list(filter(lambda z : z**2 < 50 , numbers)))

## 결과

# 홀수만 추출하기
[1, 3, 5, 7, 9]

# 3 이상, 7 미만 추출하기
[3, 4, 5, 6]

# 제곱해서 50 미만 추출하기
[1, 2, 3, 4, 5, 6, 7]

```