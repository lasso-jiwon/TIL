# 4명의 키와 몸무게 계산 (합계, 평균, 편차, 분산, 표준편차, 상관계수)
1. python을 이용한 4명의 키와 몸무게
2. numpy을 이용한 4명의 키와 몸무게

## python을 이용한 4명의 키와 몸무게


```python
height = [175, 165, 180, 160]
weight = [75, 70, 95, 72]
```

### 합계 구하기 height, weight

```python
h_sum = sum(height)
w_sum = sum(weight)
```


```python
print('키 합계:', h_sum)
print('몸무게 합계:', w_sum)
```

    키 합계: 680
    몸무게 합계: 312


### 평균 구하기 height, weight


```python
# 전체 데이터 수 구하기
length = len(height)
```


```python
h_mean = h_sum / length
w_mean = w_sum / length
```


```python
print('키 평균:', h_mean)
print('몸무게 평균:', w_mean)
```

    키 평균: 170.0
    몸무게 평균: 78.0


### 편차 구하기 height, weight


```python
h_deviation = [i-h_mean for i in height]
w_diviation = [i-w_mean for i in weight]
```


```python
print('키 편차:', h_deviation, sum(h_deviation))
print('몸무게 편차"', w_diviation, sum(w_diviation))
```

    키 편차: [5.0, -5.0, 10.0, -10.0] 0.0
    몸무게 편차" [-3.0, -8.0, 17.0, -6.0] 0.0


키 편차 : [5.0, -5.0, 10.0, -10.0]  # 총 편차 : 0.0
몸무게 편차 : [-3.0, -8.0, 17.0, -6.0]  # 총 편차 : 0.0

### 분산 구하기


```python
# h_data = [i**2 for i in h_deviation] 와 아래 식은 같음
h_data = [(i-h_mean)**2 for i in height]
w_data = [(i-w_mean)**2 for i in weight]

h_variance = sum(h_data)/(length)
w_variance = sum(w_data)/(length)
```


```python
print('키 분산:', h_variance)
print('몸무게 분산:', w_variance)
```

    키 분산: 62.5
    몸무게 분산: 99.5


### 표준편차 구하기


```python
import math
math.sqrt
h_standard_deviation = math.sqrt(h_variance)
w_standard_diviation = math.sqrt(w_variance)
```


```python
print('키 표준편차:', h_standard_deviation)
print('몸무게 표준편차', w_standard_diviation)
```

    키 표준편차: 7.905694150420948
    몸무게 표준편차 9.974968671630002


### 공분산 구하기
편차를 곱하고 갯수로 나눔
![image.png](attachment:image.png)


```python
h_w_covariance = [i*j for i, j in zip(h_deviation, w_diviation)]
print(h_w_covariance)
h_w_covariance = sum(h_w_covariance) / length
print(h_w_covariance)
```

    [-15.0, 40.0, 170.0, 60.0]
    63.75


### 상관계수


```python
h_w_coef = h_w_covariance / (h_standard_deviation)
h_w_coef
```




    8.063808033429368



## numpy를 이용한 4명의 키와 몸무게


```python
import numpy as np
```


```python
height = [175, 165, 180, 160]
weight = [75, 70, 95, 72]
```

### 합계


```python
h_sum = np.sum(height)
w_sum = np.sum(weight)
```


```python
print('키 합계:', h_sum)
print('몸무게 합계', w_sum)
```

    키 합계: 680
    몸무게 합계 312


### 평균


```python
h_mean = np.mean(height)
w_mean = np.mean(weight)
```


```python
print('키 평균:', h_mean)
print('몸무게 평균:', w_mean)
```

    키 평균: 170.0
    몸무게 평균: 78.0


### 편차


```python
h_deviation = [i-h_mean for i in height]
w_diviation = [i-w_mean for i in weight]
```


```python
print('키 편차:', h_deviation, sum(h_deviation))
print('몸무게 편차"', w_diviation, sum(w_diviation))
```

    키 편차: [5.0, -5.0, 10.0, -10.0] 0.0
    몸무게 편차" [-3.0, -8.0, 17.0, -6.0] 0.0


### 분산
ddof = 0 이 기본값임  
모수와 통계값을 구할 때 각각 달라짐


```python
h_variance = np.var(height)
w_variance = np.var(weight)

# h_variance = np.var(height, ddof=1)
# w_variance = np.var(weight, ddof=1)
```


```python
print('키 분산:', h_variance)
print('몸무게 분산:', w_variance)
```

    키 분산: 62.5
    몸무게 분산: 99.5


### 표준편차


```python
h_standard_deviation = np.std(height)
w_standard_diviation = np.std(weight)
```


```python
print('키 표준편차:', h_standard_deviation)
print('몸무게 표준편차', w_standard_diviation)
```

    키 표준편차: 7.905694150420948
    몸무게 표준편차 9.974968671630002


### 공분산


```python
# 공분산 매트릭스로 출격
h_w_covariance = np.cov(height, weight, ddof = 0) #default ddof = 0
h_w_covariance
```




    array([[62.5 , 63.75],
           [63.75, 99.5 ]])



### 상관계수


```python
# 상관계수 매트리스로 출력
np.corrcoef(height, weight)
```




    array([[1.        , 0.80840435],
           [0.80840435, 1.        ]])




```python

```


```python

```
