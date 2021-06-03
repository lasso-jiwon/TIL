#!/usr/bin/env python
# coding: utf-8

# # bytes
# 따옴표(or 쌍따옴표) 앞에 b를 붙이면 bytes 타입의 리터널이 표현법이다.  
# str과 byte는 거의 비슷한데, 차이점은 다음과 같다.
# 
# * str은 유니코드로만 구성된 문자열
# * bytes는 아스키코드로만 구성된 문자열  
# 
# bytes는 str과 마찬가지로 sequence, homogeneous, immutable 한 성질을 가진 container이다.

# In[5]:


# 바이트 처리
a = b'k'
a


# In[8]:


# 한글은 아스키코드 (바이트)로 바꿀 수 없어서 오류 뜸
# 아스키코드는 숫자, 영문자, 일부 특수문자로만 가능!
b = b'문'


# In[9]:


# 여러 글자도 가능함
c = b'python'
c


# # bytearray
# bytearray는 byte의 성질을 가지는데, mutable타입의 성질도 동시에 가지는 자료형이다.  
# byterattay는 리터널 표현법이 없으므로 생성자를 호출해서 객체를 생성해야 한다.

# In[11]:


bytearray(b'1234')


# In[12]:


print('고수\n배우이다')


# In[15]:


a = r'고수\n배우이다' 
print(a)


# # List(리스트)
# list의 리터널 표현법은 [] 대괄호이다.  
# heterogeneous(숫자+문자 혼합 사용 가능), sequence(인덱스 사용이 가능함), mutable 한 container이다.
# 

# In[18]:


a = [1,2,3]
print(a,type(a))


# In[20]:


# 리스트는 이종의 원소 타입을 가질 수 있다. (숫자+문자 조합 가능)
a=[1,2,3,'4']
a


# In[22]:


# index 사용이 가능하다.
a[0]


# In[23]:


# slice 사용 가능
a[0:3]


# ## id
# mutable에 대해서는 id를 먼저 이해해야 한다.  
# id 는 값이 저장된 메모리 주소이다.

# In[27]:


# a=[1,2,3,'4']
id(a)


# In[26]:


# a에 새로운 값 부여
a = 3  # a에 메모리 주소를 재할당
id(a)  # 값이 바뀌면 메모리 주소가 바뀐다.


# In[30]:


a=[1,2,3,'4']
print(a,id(a))


# In[31]:


a.append(5)  # append() 리스트에 값을 추가할 때 사용하는 함수, 가장 마지막에 해당 값이 추가됨
print(a,id(a))  # 값을 추가해도 원래 가지고 있는 주소 (id)의 값은 그대로 가지고 있다.


# ## mutable
# mutable의 실행 결과는 3가지 형태가 있다.
# 1. out은 없지만 자기 자신은 바뀌는 형태 `append()`
# 2. out도 있고 자기 자신도 변경이 되는 형태 `pop()`
# 3. out은 있고 자기 자신은 바뀌지 않는 형태 `count()`

# In[32]:


# 1. out은 없지만 자기 자신은 바뀌는 형태
a.append(6) #리턴되는 값 없음, 따로 입력해 줘야함
a


# In[35]:


# 2. out도 있고 자기 자신도 변경이 되는 형태
a.pop()  # 리턴 값도 있고, 자기 자신도 바뀜


# In[34]:


a


# In[37]:


# 3. out은 있고 자기 자신은 바뀌지 않는 형태
a.count(3)


# In[38]:


a


# 파이썬 키워드의 None은 다른 프로그래밍 언어에서 Null과 같다.  
# python이 out이 없으면 None과 관련이 있다.

# In[40]:


None # 키워드이다.
import keyword
type(keyword.kwlist)


# In[42]:


print(keyword.kwlist)


# ## list 생성
# 리스트를 생성하는 4가지 방법
# 1. 중괄호[] 를 이요한 생성
# 2. 함수를 이용한 생성
# 3. 요소 값을 이용한 생성
# 4. 함수와 요소 값을 이용한 생성

# In[48]:


# 1. 중괄호[] 를 이요한 생성
a = []
print(a)


# In[45]:


# 2. 함수를 이용한 생성
b = list()
print(b)


# In[50]:


# 3. 요소 값을 이용한 생성
c = [1,2,3]
c


# In[49]:


# 4. 함수와 요소 값을 이용한 생성
d = list([1,2,3])
d


# ## list 요소 추가

# ### append()
# 요소를 리스트 맨 끝에 추가한다. (파괴적 함수)  
# append(추가할 요소 값)  

# In[51]:


a = [1,2,3]

#요소를 리스트 맨 끝에 추가
a.append(5)
a


# ### insert() 
# 요소를 리스트의 특정 위치에 삽입한다. (파괴적 함수)  
# insert(인덱스 값, 추가할 요소 값) 

# In[54]:


a.insert(2,7)  # 2인덱스에 7요소를 삽입한다.
a


# **<span style="color:yellow">리스트 속 리스트 추가하기</span>**

# In[91]:


a = [10, 20, 30, 40, 50]
print(a[1:4])
a.insert(2,[1,2,3])
print(a)
print(a[0])
print(a[1])
print(a[2])
print(a[2][1])  #리스트 속 리스트의 요소 가져오기


# ## list 병합

# ### extend()
# * 기존의 리스트에 새로운 리스트를 병합시켜 기존 리스트의 원본 구조를 변경한다. (파괴적 함수)  
#     -> a.extend(b) (a 리스트에 b 리스트를 병합)
# * 연산자(+)는 원본 리스트에는 변화가 없고 연산이 완료된 새로운 리스트를 리턴한다. (비파괴적)  

# In[82]:


b = [1,3,5]
c = [2,4,6]
b.extend(c)  #b에 c리스트를 병합해라 (b리스트 뒤에 c리스트를 추가함)
print(b)


# In[60]:


d=[1,2,7]
b.extend(d)
print(b)


# In[64]:


b = [1,3,5]
c = [2,4,6]
d=[1,2,7]
print(b+c+d)  # 연산자는 비파괴적
print(b)  # b, c, d의 원본 값은 변하지 않음
print(c)
print(d)


# ## list 삭제

# ### remove
# 리스트의 요소 값을 지정하여 삭제한다. (list에서 제공하는 함수)  
# 리스트.remove(삭제할 값)

# In[81]:


a = [10, 20,30,40,50]
a.remove(20)  # remove(요소)
print(a)


# ### del
# 리스트의 인덱스 번호를 지정하여 삭제한다. (built-in)  
# del 리스트[시작 인덱스 : 끝 인덱스]

# In[69]:


a = [10, 20,30,40,50]
del(a[2:4])  #del a[2:4] 와 동일
print(a)


# ### clear()
# 리스트 안에 모든 요소를 삭제한다.  
# 리턴값 없음 

# In[70]:


a = [10, 20,30,40,50]
print(a.clear())
print(a)


# ## list 정렬

# ### sort()
# 리스트 내부 요소들을 정렬

# In[80]:


a = [5, 2, 1]

a.sort()  # 오름차순 정렬
print(a)

a.sort(reverse=True)  #내림차순 정렬
print(a)


# ### reverse()
# 리스트 요소들을 역순으로 뒤집어 준다.

# In[85]:


a = ['a','b','c']
a.reverse()
a


# 슬라이싱으로 인덱스 가져오기

# In[86]:


a = [10, 20, 30, 40, 50]
print(a[1:4])


# ## list 연산
# 리스트 더하기(+) : 리스트 + 리스트만 사용 가능   
# 리스트 반복하기 (*) : 리스트 * int 만 사용 가능

# In[93]:


a = [1,2,3]
b = [4,5,6]
print(a + b)
print(a * 3)


# In[94]:


# + 연산 주의 : 리스트에는 리스트만 연결 할 수 있다 다른 값 연결 X
print(a + 3)


# In[95]:


# * 연산 주의 : int 값만 곱할 수 있다.
print(a * b)


# # Tuple (튜플)
# * **튜플과 리스트의 공통점**  
#  임의의 객체를 저장할 수 있다는 것과 순서(sequence) 자료형이다.  
#  튜플과 리스트 모두 이종의 요소를 가질 수 있다.  
# 
#  
# * **튜플과 리스트의 차이점**  
# 튜플은 변경 불가능한 순서(sequence) 자료형이다.  
# 튜플은 함수의 가변 인수를 지원한다.  
# 튜플의 리터널 표현법은 ()소괄호이다.

# In[97]:


b = (1,2,3)
print(b, type(b))


# In[99]:


b = (1, 2, 3, [4, 5]) # 이종 요소 사용
b


# ## 연산자와 튜플 구분

# In[100]:


# 연산자의 우선순위를 정하기 위해 ()를 사용함!
# 튜플이랑 구분해야함
(1 + 3) * 4


# In[102]:


c = (1)
type(c)   # 요소 1개를 튜플로 지정하려 했지만 int로 나옴 이럴땐?


# In[115]:


c = (1,)   # 쉼표를 넣으면 튜플이 됨
type(c)


# ## ()를 생략해도 튜플

# In[104]:


a = 1, 2, 3
type(a)


# In[105]:


b = 1,
type(b)


# In[107]:


a, b = 1, 2
print(a, type(a))
print(b, type(b))


# In[109]:


c = 3, 4
a, b = c
print(a, type(a))
print(b, type(b))


# ## 튜플의 인덱싱과 슬라이싱

# In[113]:


a = 1, 2, 3
a[2]  #인덱스로 가져오면 int 형태로 리턴


# In[116]:


a[2:]  # 슬라이싱으로 가져오면 튜플의 형태로 리턴


# In[121]:


list_a = [1, 2, 3, 4]
list_a[0:2]=[100,0]


# In[122]:


list_a


# In[119]:


list_b = [1, 2, 3, 4]
list_b[1:1]=[100]


# In[120]:


list_b


# ## 튜플을 활용한 언팩킹

# In[130]:


a = ((1, 2), (3, 4))
b, c = a


# In[133]:


print(b)
print(c)


# In[134]:


(x, y),(z, k) = a


# In[137]:


x, y


# In[144]:


a, *b = (1, 2, 3, 4, 5)
# a = 1, 나머지 2, 3, 4, 5는 list 형태로 b로 가져옴

print(a, type(a))
print(b, type(b))


# In[145]:


*a, b = (1, 2, 3, 4, 5)
# a = 1, 2, 3, 4로 지정하고 5는 b로 가져옴

print(a, type(a))
print(b, type(b))


# In[146]:


def calc(a, b):
  return a+b, a * b  # 튜플을 반환한다.


# In[147]:


x, y =calc(5, 4)


# In[148]:


print(x, y)


# # Range (범위)
# homogenuous, sequence, immutable한 container이다.

# In[157]:


# 0부터 10 미만까지 1씩 증가해서 생성
a = range(10)
print(a)
list(a)


# In[160]:


# 1부터 10미만까지 2씩 증가해서 생성
b = range(1,10, 2)
print(b)
list(b)


# In[159]:


# range 함수는 최소 1개 이상의 인자를 넣어 생성해야 한다
c = range()  # type error


# In[164]:


d = range(1, 10)
d[0]  # 인덱스로 가져오기 


# In[165]:


d[3:9]  # 슬라이싱으로 가져오기


# # Set (집합)

# ## 집합의 특징
# * set은 집합이다.  
# * 집합의 특징은 중복을 허용하지 않고, 순서가 유지되지 않는다.  
# * set은 이종 데이터를 가질 수 있고(heterogenuous), 순서형이 아니며(non-sequence), 변경가능한(mutable) container이다.  
# * 리터널 표기법으로 {} 중괄호를 사용한다.

# In[166]:


# 집합의 중복을 허용하지 않는 특징
a = {1, 2, 3, 3, 3}
a


# In[167]:


# 순서가 유지되지 않는 특징
a = {2, 1, 5, 3, 4}
a


# In[171]:


# 이종데이터를 가질 수 있는 특징
a = {'가', 1, True, 0, False, '집합'}  #True = 1, False = 0을 의미한다. 둘 중 먼저 있는 값이 출력됨
a


# In[226]:


a = {'가', True, 1, False, 0, '집합'}  #True = 1, False = 0을 의미한다. 둘 중 먼저 있는 값이 출력됨
a


# In[227]:


a = {True, 1, 2, '가', True, 0, False}
# 중복 제거 = {True, 2, '가', 0}
# 정렬 = {0, 2, True, '가'}
# 순서는 숫자 - 논리값 - 문자
a


# In[175]:


# set은 sequence가 아니므로 index와 slice를 사용할 수 없다.
a[0:1]


# * 에러 메세지에 not subscriptable라고 뜬다.  
#   'set' object is not subscriptable
# * subscriptable은 숫자 또는 문자로 값을 읽어오는 것을 의미한다.  

# * 가능한 메서드에는 clear, pop, remove 등이 있으며 mutable 이다.  
# * dir()로 확인해서 객체의 특성을 파악할 수 있다.

# In[176]:


dir(a)


# In[180]:


a = {2, 1, 5, 3, 4}
a.remove(2)  # 값 2가 지워짐
a


# In[181]:


a.clear()
a


# set은 특별한 자기만의 연산자가 있다.  
# 교집합(&), 합집합(|), 차집합(-), 대칭차집합(^)

# In[183]:


a = {1, 2}
b = {2, 3}


# In[184]:


# 교집합
a & b


# In[185]:


# 합집합
a | b


# In[186]:


# 차집합
a - b


# In[187]:


# 대칭차집합
a^b


# ## 집합의 데이터 추가와 삭제

# In[221]:


a = {1, 5, 6}
a


# In[223]:


# set의 데이터 추가
a.add(3)
a


# In[224]:


# set의 데이터 삭제
a.remove(3)
a


# ## 집합 연결하기 update()

# In[225]:


# set 2개의 결합
b = {7, 8, 9}
a.update(b)
a


# set은 mutable을 원소로 가질 수 없다.  
# 내부적으로 hash 기법으로 mapping되는 형태이기 때문이다.  
# mutabledms 자기 자신이 바뀌기 때문에 중복 검사하기가 어렵다.  

# In[188]:


c = {1, 'a'}
c


# In[189]:


c = {1, [2, 3]}  # 에러! 리스트를 원소로 가질 수 없음


# In[190]:


c = {1, {2, 3}}  # 에러! 집합을 원소로 가질 수 없음


#  > frozenset  
# 항상 mutable은 immutable과 짝을 이룬다.  
# set(mutable)의 짝은 frozenset(immutable)이다.  
# frozenset 리터널 표기법이 제공이 안되고 frozenset()을 이용해서 생성한다.

# In[191]:


f = frozenset([1, 2, 3])
f


# In[192]:


type(f)


# In[193]:


dir(f)  #frozenset은 immutable이므로 clear, pop, remove 등의 메소드가 제공되지 않는다.


# ## mutable과 immutable의 차이
# mutable과 immutable의 차이를 set을 이용해서 비교

# In[194]:


print(dir(1,))  # 튜플 


# In[196]:


print(dir([1,2]))  #리스트


# In[198]:


# immutable - mutable의 차이를 비교해서 immutabel에서만 제공되는 자원을 가져옴
set(dir((1,))) - set(dir([1,2]))   # 튜플 - 리스트 = immutable - mutable 차집합으로 나타냄


# In[199]:


# mutabel에서만 제공되는 자원을 가져옴
set(dir([1,])) - set(dir((1,)))


# ## 집합의 연산자
# set에서 +(더하기), *(곱하기) 연산자를 사용할 수 없다
# 
# * 같은 sequence 컨터이너끼리는 더해진다.
# * 같은 sequence 컨테이너 끼리는 곱하는 것은 안되지만 컨테이너와 숫자를 곱하면 반복이 된다.
# * non-sequence 컨테이너끼리는 +, * 연산자를 사용할 수 없다.

# In[200]:


'고수' + '배우'


# In[201]:


'고수' + 3 # 문자열 + 숫자는 불가능!


# In[203]:


'고수' * 3 # 문자열 * 숫자는 반복


# In[204]:


[1,2] + [3,4]  # 같은 컨테이너끼리 더하기가 가능함


# In[205]:


[1,2] * 3


# In[207]:


(1, 2) + (3, 4)


# In[208]:


(1, 2) * 3


# In[209]:


# set(집합)은 + 연산자를 제공하지 않는다
{1, 2} + {3, 4}


# In[210]:


# * 연산자도 사용 불가능하다.
{1, 2} * 3


# # Coercion 타입 변환
# python에서는 연산 결과 타입이 바뀌는 경우가 있다.  
# 타입 변환을 강제로 하는 coercion 때문이다.

# In[215]:


a = 10 / 4  # int 연산을 통해 float로 바뀜
print(a, type(a))


# In[214]:


10 // 4 # 몫 구하기


# # 비교 연산자
# container는 비교(관계) 연산자를 사용할 수 있다. 같은 타입끼리만 연산을 할 수 있다.  
# 문자열은 코드 (아스키/유니코드)로 바뀌어서 대소비교한다.  
# 아스키 코드 영문 대문자 A = 65, 소문자 a = 97, 숫자 '0; = 48 이다.  

# In[216]:


'A' > 'a'  #문자열은 코드 


# In[217]:


'aA' > 'a'


# In[218]:


# 다른 컨테이너하고는 비교 연산자를 사용할 수 없다.문자열과 int는 서로 비교 불가!
'A' < 1


# # Dictionary (딕셔너리)
# dictionary의 리터널 표현법은 중괄호{}와 콜론(:) 이다.  
# key : value 쌍을 이루는 구조를 mapping 시킨다고 한다.  
# key에는 고윳값을 넣는게 좋음!

# In[228]:


a = {1:10, 'b' : 2, 'a' : [1, 2, 3]}
a


# dictionary는 index를 key로 한다.  
# dictionary는 문자로 인덱싱할 수 있다.  
# 다른 container들은 숫자만 가능하다.

# In[231]:


a['b']  # 문자로 요소를 접근


# In[232]:


a[1]  # 숫자로 요소를 접근


# In[233]:


a['k']  # 없는 값을 넣으면 KeyError!


# In[234]:


# 딕셔너리에서는 슬라이싱이 제공되지 않는다.
a[1:'b']


# dictionary와 set은 비슷하다.  
# - 중괄호를 사용하는 literal 표현도 비슷하다.  
# 비어있는 중괄호는 비어있는 dictionary다.  
#   
# - python 내부 구조를 보면 dictionary를 기반으로 set을 만들었다.  
# 둘 다 hashable type만 가질 수 있다.  
# dict의 경우 key만 hashtable이면 된다.

# In[238]:


# 비어있는 dictionary 생성
a = {} 
type(a)


# In[239]:


# 비어있는 set 생성
b = set()
type(b)


# In[241]:


# key 값이 겹치면 에러를 발생시키지 않고 뒤에 값이 앞에 값을 덮어쓴다.
a = {'k' : 1, 'm':2, 'p':3, 'm':4}
a

