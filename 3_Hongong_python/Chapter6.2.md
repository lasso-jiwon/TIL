# 요약
* **예외 객체**는 예외와 관련된 정보를 담고 있는 객체입니다.
* **raise 구문**은 예외를 강제로 발생시킬 때 사용하는 구문입니다.
* **GitHub 검색**은 많은 사람이 함께 개발하는 소셜 코딩 사이트 깃허브를 이용하는 것으로 유능한 개발자들의 정제된 코드를 살펴볼 수 있습니다.

# 예외 객체
```except 예외의 종류 as 변수로 사용할 이름 :```

* 예외의 종류에 무엇을 넣어야 할지 모르겠으면 모든 예외들의 어머니인 `Exception` 을 넣어준다. 근데 여기서 Exception 이 캐멀 케이스네? -> 클래스 이름이네? 즉 Exception 은 클래스임.

* 변수로 사용할 이름은 일반적으로 e 나 exception 소문자로 넣음.

```python
try :
    #숫자로 변환합니다.
    number = int(input("정수 입력 >"))
    #출력합니다.
    print("원의 반지름 :", number)
    print("원의 둘레 :", 2 * 3.14 * number)
    print("원의 넓이 :", 3.14 * number * number)
# except 예외의 종류 as 변수로 사용할 이름 :
except Exception as exception:
    print(type(exception))
    print(exception)
    
## 실행 결과
정수 입력 >ㅇㅂㅇ
<class 'ValueError'>
invalid literal for int() with base 10: 'ㅇㅂㅇ'
```
실행결과에서 exception의 타입과 내용을 확인할 수 있다.
근데 사용자 입장에서는 저렇게 코드로 보여주면 모름.. 어쩌라는겨? 🤔
그래서 아래와 같이 수정한다.

* if type(exception) == ValueError:

```python
try :
    #숫자로 변환합니다.
    number = int(input("정수 입력 >"))
    #출력합니다.
    print("원의 반지름 :", number)
    print("원의 둘레 :", 2 * 3.14 * number)
    print("원의 넓이 :", 3.14 * number * number)
# except 예외의 종류 as 변수로 사용할 이름 :
except Exception as exception:
    if type(exception) == ValueError:
        print("값에 문제가 있습니다...!")
        
## 실행 결과
정수 입력 >ㅇㅂㅇ
값에 문제가 있습니다...!
```
___
# if 조건문으로 예외 구분하기

0~4까지 인덱스 번호를 입력하면 해당 값을 출력해주는 함수
* if type(exception) == ValueError:
근데 5를 입력함..-_- IndexError 발생
```python
try :
    a = [123, 121, 454, 657, 100]
    number = int(input("정수 입력(0~4까지 입혁해주세요) >"))
    print(a[number])
except Exception as exception:
    print(type(exception))
    if type(exception) == ValueError:
        print("값에 문제가 있습니다...!")
        
## 실행 결과
정수 입력(0~4까지 입혁해주세요) >5
<class 'IndexError'>
```

* elif type(exception) == IndexError 추가
```python
try :
    a = [123, 121, 454, 657, 100]
    number = int(input("정수 입력(0~4까지 입혁해주세요) >"))
    print(a[number])
except Exception as exception:
    print(type(exception))
    if type(exception) == ValueError:
        print("값에 문제가 있습니다...!")
    elif type(exception) == IndexError:
        print("0~4까지 숫자를 입력해주세요.")
        
## 실행 결과
정수 입력(0~4까지 입혁해주세요) >5
<class 'IndexError'>
0~4까지 숫자를 입력해주세요.
```
이러한 형태로 예외를 구분하는 코드는 굉장히 굉장히 많이 사용이 되기 때문에...파이썬 개발자들은 들여쓰기를 한번 더 넣지 않고 예외를 구분할 수 있는 방법을 사용합니다. 바로 except 구문을 여러개 사용하는 것!
___
# except 구문으로 예외 구분하기
마지막 `except Exception as exception` 의 Exception 은 모든 예외의 부모이기 때문에  여기서는 무조건 걸리기 때문에 마지막에 필수적으로 사용하는 경우가 많다!!!
```python
try :
    a = [123, 121, 454, 657, 100]
    number = int(input("정수 입력(0~4까지 입혁해주세요) >"))
    print(a[number])
except ValueError as exception:
    print("값에 문제가 있습니다.")
except IndexError as exception:
    print("0부터 4까지 입력해주세요.")
except Exception as exception:  # 필수 사용
    print("알 수 없는 예외가 발생했습니다.")
    # 개발자에게 메일을 보낸다.
    
## 실행 결과
정수 입력(0~4까지 입혁해주세요) >//
값에 문제가 있습니다.
```
___
# raise 키워드

raise 뒤에 '예외 객체' 를 넣으면 예외가 강제적으로 발생하게 됩니다.  
뭘 넣어야 할지 모르겠다면 역시나 Exception()
raise Exception~~
```python
raise Exception ("안녕하세요")

## 실행 결과
Exception: 안녕하세요
```
그래서 이걸 언제 써....?
예외를 강제적으로 발생하는것은 어떤 문제가 있어서 프로그램을 긴급하게 종료하려고 할 때 사용합니다! raise 개발자들을 위한 라이브러리를 개발할 때 사용함. 개발자가 어떤 실수를 할 때 프로그램이 강제 종료되게 해서 '여기에서 문제가 있어서 더 프로그램을 실행하면 안될 거 같으니까 그냥 종료할게' 라는 느낌으로 쓰임!

___
# 확인 문제
#### 1. 예외를 강제로 발생시킬 때 사용하는 키워드로 맞는 것은 무엇일까요?
답) 2번 raise