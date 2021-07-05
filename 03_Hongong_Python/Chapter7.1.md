# 요약
* **표준 모듈**은 파이썬이 기본적으로 제공하는 모듈입니다.
* **import 구문**은 모듈을 읽어 들일 때 사용하는 구문입니다.
* **as키워드**는 모듈을 읽어 들이고 별칭을 붙일 때 사용하는 구문입니다.
* **파이썬 문서**는 모듈의 자세한 사용 방법이 들어있는 문서입니다.


# 모듈이란
다른 사람이 만들어 둔 변수와 함수를 읽어들여서 사용할 수 있는 기능
* 표준(내장)모듈 : 파이썬에 기본적으로 내장되어 있는 모듈
* 외부(외장)모듈 : 내장되어 있지 않아서, 별도로 다운 받아서 사용하는 모듈

___

# 모듈 읽어 들이기(1) : __import__() 함수
math 모듈 사용하기
```python
math = __import__('math')

print(math.pi)
print(math.sin(10))

## 실행 결과
3.141592653589793
-0.5440211108893699
```

모듈을 굉장히 많이 사용하기때문에 별도의 구문으로 제공한다!

# 모듈 읽어 들이기(2) : import 구문
```python
# math = __import__('math')  이 코드와 아래 코드는 동일한다
import math

print(math.pi)
print(math.sin(10))

## 실행 결과
3.141592653589793
-0.5440211108893699
```

근데 math 라는 이름을 별도로 사용하고 있어서 이름 충돌이 발생하는 경우가 있을 수 있음. 그럴 땐 이름을 변경해서 사용하기!

# 모듈 읽어 들이기(3) : import as 구문
```python
# math = __import__('math') 
import math

# 수학 = __import__('math')
import math as 수학

print(수학.pi)
print(수학.sin(10))

## 실행 결과
3.141592653589793
-0.5440211108893699
```

# 모듈 읽어 들이기(4) : from import 구문
`from math import pi, sin` 구문을 사용하면 math.pi 가 아닌 그냥 pi 로 사용할 수 있게 된다!

```python
from math import pi, sin
print(pi)
print(sin(10))

## 실행 결과
3.141592653589793
-0.5440211108893699
```

나는 pi, sin 이렇게 하나하나 들고오는거 말고 모든 기능을 들고오면 좋겠어!

# 모듈 읽어 들이기(5) : form import * 구문
```python
from math import *
print(pi)
print(sin(10))
print(tan(10))

## 실행 결과
3.141592653589793
-0.5440211108893699
0.6483608274590867
```
___
# 모듈 읽어 들이기 정리
모듈을 읽는 다섯가지 방법
```python
# 1. 자주 사용되지 않음
math = __import__('math')

# 2. math 모듈 읽기
import math

# 3. math 모듈을 수학으로 읽기
import math as 수학

# 4. math 모듈에서 pi, sin 코드를 가져온다.
# math.pi가 아니라 그냥 pi로 사용 가능
from math import pi, sin

# 5. math 에서 모든 기능 들고온다.
# math.pi가 아니라 그냥 pi로 사용 가능
from math import *
```

#### 간단 퀴즈 : 다음 중 모듈을 읽어들이는 방법으로 알맞은 것은?
```python
from math import pi, sin (O)
import pi, sin from math (X)
import math from pi, sin (X)
import pi, sin from math (X)
```
___
# 파이썬 표준 라이브러리
https://docs.python.org/ko/3/library/index.html
___

# sys 모듈
sys.argv란 argument value의 약자로 한국어로는 "명령 매개 변수"라고 부르는 경우가 많습니다.


```python
> import sys
> print(sys.argv)
['/Users/jiwon/opt/anaconda3/lib/python3.8/site-packages/ipykernel_launcher.py', '-f', '/Users/jiwon/Library/Jupyter/runtime/kernel-8523e3c4-045e-4004-9195-dc16986913fe.json']
```

 # datetime 모듈
 ```python
> import datetime
> datetime.datetime.now()
datetime.datetime(2021, 6, 22, 17, 37, 54, 862519)

> from datetime import datetime
> now = datetime.now()
> print(now.year)
> print(now.month)
> print(now.day)
> print(now.hour)
> print(now.minute)
> print(now.second)
2021
6
22
17
39
27

> now = datetime(2000, 1, 1, 1, 1, 1)
> print(now.year)
> print(now.month)
> print(now.day)
> print(now.hour)
> print(now.minute)
> print(now.second)
2000
1
1
1
1
1
 ```

 # time 모듈
인터넷에서 데이터 수집을 하면 제약이 없이 1초에 1만번 수집이 됨.
상대방 사이트에 부담이 걸리기 때문에
1초에 1번만 요청한다던지 잠시 대기를 걸어둔다.

 ```python
# A를 출력하고 2초 뒤에 B를 출력함
import time
print("A")
time.sleep(2)
print("B")

A
B
 ```

 # urllib 모듈
 모듈과 모듈을 조합해서 사용하는 방법 urllib에서 request 모듈을 알아보자
 request.urlopen() 특정 웹 사이트를 긁어오는 코드

 ```python
 from urllib import request

target = request.urlopen("http://naver.com")
content = target.read()
print(content[:100])

## 결과 : 바이너리 문자열 출력
b'\n<!doctype html>                 <html lang="ko" data-dark="false"> <head> <meta charset="utf-8"> <t'
 ```
___

 # 확인문제
 #### 1. 다음 중 math 모듈의 함수를 제대로 읽어 들이지 못하는 코드를 고르세요.
 1) import math
 2) import sin, cos, tan from math
 3) import math as m
 4) from math import *

 정답 : 2번 (from math import sin, cos, tan)

 #### 2. 파이썬 문서를 보면서 본문에서 살펴보지 않았던 모듈의 이름을 다섯 개 적어 보세요. 그리고 해당 모듈에 어떠한 기능들이 들어 있는지도 간단하게 적어보세요.

| 번호 | 모듈 이름       | 모듈 기능                  |
| ---- | --------------- | -------------------------- |
| 0    | wave 모듈       | wav음악 형식과 관련된 처리 |
| 1    | array 모듈      | 효율적인 숫자 배열         |
| 2    | random 모듈     | 의사 난수 생성             |
| 3    | statistics 모듈 | 수학 통계 함수             |
| 4    | runpy 모듈      | 파이썬 모듈 찾기와 실행    |
| 5    | csv 모듈        | csv 파일 읽기와 쓰기       |

 #### 3. os모듈의 os.listdir() 함수와 os.path.isdir() 함수를 사용하면 특정 디렉터리를 읽어 파일 디렉터리인지를 확인할 수 있습니다. 직접 코드를 작성하고 실행해 보세요. 실행하는 위치에 따라서 출력 결과가 달라집니다.

 **_현재 디렉터리를 읽어 들이고 파일인지 디렉터리인지 구분하기_**
 `os.listdir(폴더 경로)` : 폴더 내부의 모든 것을 리스트로 리턴함
 `os.path.isdir(경로)` : 경로에 있는 것이 디렉터리면 True 아니면 False

 ```python
 # 모듈을 읽어 들입니다.
import os

# 현재 폴더의 파일/폴더를 출력합니다.
output = os.listdir(".")
print("os.listdir():", output)
print()

# 현재 폴더의 파일/폴더를 구분합니다.
print("# 폴더와 파일 구분하기")
for path in output:
    if os.path.isdir(path):
        print("폴더:", path)
    else:
        print("파일:", path)
        
## 결과
os.listdir(): ['string_operator02.py', 'chapter8-2 클래스의 추가적인 구문.ipynb', 'variable.ipynb', 'chapter4-3 반복문과 while 반복문.ipynb', 'chapter5-2 함수의 활용.ipynb'
폴더와 파일 구분하기
파일: string_operator02.py
파일: chapter8-2 클래스의 추가적인 구문.ipynb
파일: variable.ipynb
파일: chapter4-3 반복문과 while 반복문.ipynb
파일: chapter5-2 함수의 활용.ipynb
파일: Untitled1.ipynb

# 폴더와 파일 구분하기
파일: string_operator02.py
파일: chapter8-2 클래스의 추가적인 구문.ipynb
파일: variable.ipynb
파일: chapter4-3 반복문과 while 반복문.ipynb
파일: chapter5-2 함수의 활용.ipynb
 ```

 그리고 이를 활용해서 '폴더 라면 또 탐색하기'라는 재귀 구성으로 현재 폴더 내부에 있는 모든 파일을 탐색하도록 코드를 작성해 보세요.

> #### 확인문제 3번의 두 번째 자료 : 243페이지 (5-2 함수의 활용)
> 다음 빈칸을 재귀함수로 만들어거 리스트를 평탄화하는 함수를 만들어보세요. 리스트 평탄화는 중첩된 리스트가 있을 때 중첩을 모두 제거하고 풀어서 1차원 리스트로 만드는 것을 의미합니다.
```python
def flatten(data) :
    output = []
    for item in data :
        if type(item) == list :
            output += flatten(item)
        else :
            output += [item]
    return output
example = [[1,2,3], [4,[5,6]], 7,[8,9]]
print("원본 :", example)
print("변환 :", flatten(example))
원본 : [[1, 2, 3], [4, [5, 6]], 7, [8, 9]]
변환 : [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

```python
# 모듈을 읽어들입니다.
import os

# 폴더를 읽어 들이는 함수
def read_folder(path):
    # 폴더의 요소 읽어 들이기
    output = os.listdir(path)
    # 폴더의 요소 구분하기
    for item in output:
        if os.path.isdir(path+"/"+item):
            # 폴더라면 계속 읽어 들이기
            read_folder(path+"/"+item)
        else:
            # 파일이라면 출력하기
            print("파일:", item)
            
# 현재 폴더의 파일/폴더를 출력합니다.
read_folder(".")

## 실행 결과
파일: string_operator02.py
파일: chapter8-2 클래스의 추가적인 구문.ipynb
파일: variable.ipynb
파일: chapter4-3 반복문과 while 반복문.ipynb
파일: chapter5-2 함수의 활용.ipynb
파일: Untitled1.ipynb
파일: .DS_Store
파일: string_operator02.ipynb
파일: Untitled3.ipynb
파일: Untitled.ipynb
파일: chapter7-1 표준 모듈.ipynb
```