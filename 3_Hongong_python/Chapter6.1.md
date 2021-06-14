# 요약

* **구문 오류** 는 프로그램의 문법적인 오류로 프로그램이 실행조차 되지 않게 만드는 오류입니다.
* **예외**는 프로그램 실행 중에 발생하는 오류입니다. try, catch 구문 등으로 처리할 수 있습니다. 반대로 구문 오류는 실행 자체가 안 되므로 try catch 구문으로 처리할 수 없습니다.
* **기본 예외 처리**는 조건문 등을 사용해 예외를 처리하는 기본적인 방법입니다.
* **try except 구문**은 예외처리에 특화된 구문입니다.
___

# 오류의 종류

* **구문 오류** : 코드에 문제가 있어서 실행되지 않음
* **예외**(런타임 오류) : 코드 자체 문법적인 오류 X, 실행과 관련된 문제, 실행은 되지만 오류가 발생해서 죽어버림

구문 오류 예시
```python
print("실행되었습니다.")   # 아래 코드가 오류가 나서 윗줄 코드가 실행조차 안됨
print("ㅇㅂㅇ)

## 결과 - 실행되었습니다 글자 출력 X
SyntaxError: EOL while scanning string literal
```

예외 예시
```python
print("실행되었습니다")
list_a = [1, 2, 3]
print(list_a[100])

## 결과 - 실행되었습니다 글자 출력 O
실행되었습니다.
IndexError: list index out of range
```

# 기본 예외 처리
```python
# 숫자를 입력받습니다.
number_input_a = int(input("정수 입력 > "))

# 출력합니다.
print("원의 반지름 :", number_input_a)
print("원의 둘레 :", 2 * 3.14 * number_input_a)
print("원의 넓이 :", 3.14 * number_input_a * number_input_a)
```

여기서 실수를 입력하거나 정수가 아닌 ㅇㅂㅇ 같은걸 입력하면 중간에 죽어버림 x_x
정수를 입력하지 않으면 정수를 입력하라고 다시 요청!!!

`isdigit()` 함수를 사용해서 숫자로만 구성된 글인지 확인하기

```python
# 숫자를 입력받습니다.
string_input = input("정수 입력 > ")
if string_input.isdigit():
    number_input_a=int(string_input)
    # 출력합니다.
    print("원의 반지름 :", number_input_a)
    print("원의 둘레 :", 2 * 3.14 * number_input_a)
    print("원의 넓이 :", 3.14 * number_input_a * number_input_a)
else :
    print("정수를 입력해주세요!")
    
## 결과 입력
정수 입력 > ㅇㅂㅇ
정수를 입력해주세요!
```


# try except 구문

#### try except 구문의 기본적인 구조
```
try :
    # 예외가 발생할 수 있는 가능성이 있는 코드
except :
    # 예외가 발생했을 때 실행할 코드
```
예제
사용자로부터 문자열을 입력받고, 숫자로 변환하고, 제곱해서 출력하는 코드
```python
try :
    print(float(input("숫자를 입력해주세요 : "))**2)
except :
    # 예외가 발생했을 때 실행할 코드
    print("숫자를 입력해주세요!!!!!")
    
## 결과
숫자를 입력해주세요 : ㅇㅂㅇ
숫자를 입력해주세요!!!!!
```

## while 반복문을 이용해서 계속해서 입력받기
```python
while True :
    try :
        print(float(input("숫자를 입력해주세요 : "))**2)
        break  ## 오류가 발생했을 경우 여기서 코드를 중지
    except :
        # 예외가 발생했을 때 실행할 코드
        print("숫자를 입력해주세요!!!!!")
        
## 결과
숫자를 입력해주세요 : sd
숫자를 입력해주세요!!!!!
숫자를 입력해주세요 : sd
숫자를 입력해주세요!!!!!
숫자를 입력해주세요 : ㅇㅂㅇ
숫자를 입력해주세요!!!!!
숫자를 입력해주세요 : ㅇㅂㅇ
숫자를 입력해주세요!!!!!
숫자를 입력해주세요 : 34
1156.0
```

## 예외로 죽지 않게만 하기 except + pass


 ```python
while True :
    try :
        print(float(input("숫자를 입력해주세요 : "))**2)
        break
    except :
        pass

## 결과
숫자를 입력해주세요 : ㅇㅂㅇ
숫자를 입력해주세요 : ㅇㅂㅇ
숫자를 입력해주세요 : ㅇㅅㅇ
숫자를 입력해주세요 : ㅇㅅㅇ
숫자를 입력해주세요 : ㅇㅂㅇ
숫자를 입력해주세요 : ㅇㅅㅇ
 ```

**예제로 알아보기**

```python
 # 변수로 선언합니다.
list_input_a = ["52", "273", "43", "스파이", "123"]

# 반복을 적용합니다.
list_number = []
for item in list_input_a:
    # 숫자로 변환하여 리스트에 추가합니다.
    try:
        float(item)  # 예외가 발생하면 알아서 다음으로 진행은 안 되겠지?
        list_number.append(item)  # 예외 없이 통과했으면 리스트에 넣어줘
    except:
        pass
    
# 출력합니다.
print("{}내부에 있는 숫자는".format(list_input_a))
print("{}입니다".format(list_number))

## 결과
['52', '273', '43', '스파이', '103'] 내부에 있는 숫자는
['52', '273', '32', '103'] 입니다.
```

 # try except else 구문
else에 예외가 발생하지 않았을 때 실행할 코드 넣기!

#### try except else 구문의 기본적인 구조
```python
try:
	예외가 발생할 가능성이 있는 코드
except:
	예외가 발생했을 때 실행할 코드
else:
	예외가 발생하지 않았을 때 실행할 코드
```

try except else 구문을 사용할 때는 예외가 발생할 가능성이 있는 코드만 try 구문 내부에 넣고 나머지를 모두 else 구문으로 빼는 경우가 많습니다. 다음 코드를 살펴보겠습니다.

```python
try:
    number_input_a = int(input("정수 입력 >"))
except:
    print("정수를 입력하라능!!!! ㅇㅂㅇ!!")
else:
    print("원의 반지름 :", number_input_a)
    print("원의 둘레 :", 2 * 3.14 * number_input_a)
    print("원의 넓이 :", 3.14 * number_input_a * number_input_a)
    
## 결과
정수 입력 >asdf
정수를 입력하라능!!!! ㅇㅂㅇ!!
```

# finally 구문

finally 구문은 예외처리 구문에서 가장 마지막에 사용할 수 있는 구문으로 예외가 발생하든 발생하지 않든 무조건 실행할 때 사용하는 코드이다.

#### try except else finally 구문의 기본적인 구조
```python
try:
	예외가 발생할 가능성이 있는 코드
except:
	예외가 발생했을 때 실행할 코드
else:
	예외가 발생하지 않았을 때 실행할 코드
finally:
    무조건 실행할 코드
```

**예제로 알아보기**
```python
# try except 구문으로 예외를 처리합니다.
try:
    #숫자로 변환
    number_input_a = int(input("숫자를 입력하세요 >"))
    # 출력합니다.
    print("원의 반지름 :", number_input_a)
    print("원의 둘레 :", 2 * 3.14 * number_input_a)
    print("원의 넓이 :", 3.14 * number_input_a * number_input_a)
except:
    print("정수를 입력하하고 했늬 안했니....?")
else:
    print("예외가 발생하지 않았다능!!!")
finally:
    print("일단 프로그램이 어떻게든 끝났습니다만...?")
    
## 결과
숫자를 입력하세요 >ㄱㄷ
정수를 입력하하고 했늬 안했니....?
일단 프로그램이 어떻게든 끝났습니다만...?
```

## try, except, finally 구문의 조합
예외처리 구문은 다음과 같은 규칙을 지켜야 한다.
* try 구문은 단독으로 사용할 수 없으며, 반드시 except 구문 또는 finally 구문과 함께 사용해야한다.
* else 구문은 except 구문 뒤에 사용해야 합니다.

**조합 알아보기**
* try + except 구문
* try + except + else 구문
* try + except + finally rnans
* try + finally 구문

## finally 구문의 활용
#### 파일이 제대로 닫혔는지 확인하기
```python
try:
    file = open("info.txt", "w")
    file.close()
except Exception as e:
    print(e)
    
print("#파일이 제대로 닫혔는지 확인하기")
print("file.closed:", file.closed)

## 결과
#파일이 제대로 닫혔는지 확인하기
file.closed: True
```

#### finally 구문 사용해 파일 닫기
파일 중간에 예외가 발생할 경우 파일이 닫히지 않습니다. 따라서 반드시 finally 구문을 사용해서 파일을 닫아야 함!!!!
```python
try:
    file = open("info.txt")
    예외.발생해라()
except Exception as e:
    print(e)
finally:
    file.close()
print("#파일이 제대로 닫혔는지 확인하기")
print("file.closed:", file.closed)

## 결과
name '예외' is not defined
#파일이 제대로 닫혔는지 확인하기
file.closed: True
```

하지만!!!!
무조건 finally 구문만을 이용해서 파일을 닫는것은 아님! 코드가 더 깔끔해 보일 때만 사용하는것
그냥 try except file.close() 해도 잘만 닫힌다

# finally 구문 이렇게 씁니다!
## try 구문 내에서 return 키워드를 사용하는 경우
finally 구문은 반복문 또는 함수 내부에 있을 때 위력을 발휘한다!
```python
# test() 함수를 선언합니다.
def test():
    print("test() 함수의 첫 줄입니다.")
    try:
        print("try 구문이 실행되었습니다.")
        return
        print("try 구문의 return 키워드 뒤 입니다.")
    except:
        print("except 구문이 실행되었습니다.")
    else:
        print("else 구문이 실행되었습니다.")
    finally:  # try 구문 내부에 finally가있음!!!!!
        print("finally 구문이 실행되었습니다.")
    print("test() 함수의 마지막 줄입니다.")
    
# test() 함수를 호출합니다.
test()

## 결과
test() 함수의 첫 줄입니다.
try 구문이 실행되었습니다.
finally 구문이 실행되었습니다.
```
try 구문 내부에 return 키워드가 있다는 것이 포인드입니다. try 구문 중간에 탈출해도 finally 구문은 무조건 실행됩니다. try 구문에서 원할 때 return 키워드로 빠져나가도 파일이 무조건 닫히기 때문입니다.

___

# 확인 문제
#### 1. 구문오류와 예외의 차이를 설명해 보세요.
* 구문 오류 : 프로그램이 실행되기도 전에 발생하는 오류, 해결하지 않으면 프로그램 자체가 실행되지 않음.
문법적인 문제를 해결해서 오류를 해결한다.
* 예외 : 프로그램 실행 중에 발생하는 오류, 프로그램이 일단 실행되고 해당 지점에서 오류를 발생.
예외처리를 통해 해결할 수 있다.

#### 2. 다음 코드의 빈칸을 조건문을 사용한 코드, try except 구문을 사용한 코드로 채워서 예외가 발생하지 않고 코드가 실행결과처럼 출력되게 하세요.

* 리스트 내부의 특정 값 확인하기 `index()` 함수 사용
```python
numbers = [12, 546, 126, 232, 45, 66]
numbers.index(232)
3
```
* 숫자가 많으면 가장 첫번째 있는 위치를 리턴
```python
numbers = [1, 1, 1, 1, 1, 1]
numbers.index(1)
0
```
* 없는 값을 찾으면 value error 발생
```python
numbers = [12, 546, 126, 232, 45, 66]
numbers.index(1000000)
ValueError: 1000000 is not in list
```
* 풀이 1 if else 사용
```python
numbers = [12, 324, 125, 68, 95, 775, 52, 45]

print("# (1) 요소 내부에 있는 값 찾기")
print("- {}는 {}위치에 있습니다.".format(52, numbers.index(52)))
print()

print("# (2) 요소 내부에 없는 값 찾기")
number = 10000

if number in numbers:
    print("- {}는 {}위치에 있습니다.".format(number, numbers.index(number)))
else:
    print("- 리스트 내부에는 없는 값입니다.")
print()

print("--- 정상적으로 종료되었습니다. ---")
```

* 풀이 2 try except 사용

```python
numbers = [12, 324, 125, 68, 95, 775, 52, 45]

print("# (1) 요소 내부에 있는 값 찾기")
print("- {}는 {}위치에 있습니다.".format(52, numbers.index(52)))
print()

print("# (2) 요소 내부에 없는 값 찾기")
number = 10000

try:
    print("- {}는 {}위치에 있습니다.".format(number, numbers.index(number)))
except:
    print("- 리스트 내부에는 없는 값입니다.")
print()

print("--- 정상적으로 종료되었습니다. ---")
```
* 실행 결과
```
# (1) 요소 내부에 있는 값 찾기
- 52는 6위치에 있습니다.

# (2) 요소 내부에 없는 값 찾기
- 리스트 내부에는 없는 값입니다.

--- 정상적으로 종료되었습니다. ---
```


#### 3. 다음 중 구문 오류 발생이 예상되면 '구문 오류'에, 예외 발생이 예상되면 '예외'에 체크 표시를 한 후, 예상되는 에러명도 적어보세요.

```python
output = 10 + "개"      # (1)
int("안녕하세요")         # (2)
cursor.close)          # (3)
[1, 2, 3, 4, 5][10]    # (4)
```

(1) 예외 (Type Error) 
(2) 예외 (ValueError)
(3) 구문 오류 (SyntaxError)
(4) 예외 (IndexError)