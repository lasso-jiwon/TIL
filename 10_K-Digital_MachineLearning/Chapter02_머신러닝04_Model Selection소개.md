# 학습 데이터 세트 / 테스트 데이터 세트 분리

사이킷런 model_selection 모듈의 주요 기능
- 학습 데이터와 테스트 데이터 세트 분리
- 교차 검증 분할 및 평가
- Estimator의 하이퍼 파라미터 튜닝

학습 데이터와 테스트 데이터 세트 분리
- train_test_split() 함수 사용

학습 데이터 세트
- 머신러닝 알고리즘의 학습을 위해 사용
- 데이터의 피처(속성)과 레이블(결정값) 모두 포함
- 학습 데이터를 기반으로 머신러닝 알고리즘이 데이터 속성과 (피처)과 결정값을 (레이블)의 패턴을 인지하고 학습

테스트 데이터 세트
- 학습된 머신러닝 알고리즘 테스트용
- 머신러능 알고리즘은 제공된 속성 데이터를 기반으로 결정값 예측
- 학습 데이터와 별도의 세트로 제공

train_test_split() 함수 사용
- train_test_split(iris_data, iris_label, test_size=0.3, random_state=11)
- train_test_split(피처 데이터 세트, 레이블 데이터 세트, 테스트 데이터 세트 비율, 난수 발생값)
- 피처 데이터 세트 : 피처(feature)만으로 된 데이터(numpy) [5.1, 3.5, 1.4, 0.2],...
- 레이블 데이터 세트 : 레이블(결정 값) 데이터(numpy) [0 0 0 ... 1 1 1 .... 2 2 2]
- 테스트 데이터 세트 비율 : 전체 데이터 세트 중 테스트 데이터 세트 비율 (0.3)
- 난수 발생값 : 수행할 때마다 동일한 데이터 세트로 분리하기 위해 시드값 고정 (실습용)

train_test_split() 반환값
- X_train : 학습용 피처 데이터 세트 (feature)
- X_test : 테스트용 피처 데이터 세트 (feature)
- y_train : 학습용 레이블 데이터 세트 (target)
- y_test : 테스트용 레이블 데이터 세트 (target)
- feature : 대문자 X_
- label(target) : 소문자 y_

___
## 학습 데이터/테스트 데이터 세트 분리 예제

붓꽃 데이터 품종 예측
1. 학습/테스트 데이터 세트로 분리하지 않고 예측
2. 학습/테스트 데이터 세트로 분리하고 예측

### 1. 학습/테스트 데이터 세트로 분리하지 않고 예측


```python
# 1. 학습/테스트 데이터 세트로 분리하지 않고 예측
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
dt_clf = DecisionTreeClassifier()

train_data = iris.data # 피처(속성)만으로 된 데이터
train_label = iris.target # target 값(결정값(정답), 레이블 값)

# 학습 수행 : 테스트 데이터 세트 분리하고 않고 사용
# fit(학습용 피처 데이터, 학습용 target 데이터)
dt_clf.fit(train_data, train_label)

# 학습된 데이터 셋으로 예측 수행 : 테스트 데이터 세트 분리하지 않고 사용
pred = dt_clf.predict(train_data)

print('예측 정확도 : ', accuracy_score(train_label, pred))

# 결과
# 예측을 별도로 분리하지 않고 학습된 train_data로 했기 때문에
# 결과가 1.0 (100%)로 출력됨 (잘못된 예측 방법!!!!!)
```

    예측 정확도 :  1.0


학습 데이터와 유사한 데이터로 테스트를 했다면  
알고리즘이 안좋더라도 예측 정확도가 높을 수 있음 (신뢰성 떨어짐)  
  
알고리즘을 얼마나 잘 학습했는냐는  
기존의 학습 데이터에는 포함되어 있지 않은 데이터 대해  
얼마나 잘 예측할 수 있는냐와 밀접한 관계가 있음  

### 2. 학습/테스트 데이터 세트로 분리하고 예측 : sklearn


```python
# 2. 학습/테스트 데이터 세트로 분리하고 예측
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dt_clf = DecisionTreeClassifier()
iris_data = load_iris()

# 학습/테스트 데이터 세트 분리
# 학습 데이터와 테스트 데이터 세트로 분리
X_train, X_test, y_train, y_test = train_test_split(iris_data.data,
                                                    iris_data.target,
                                                    test_size=0.3,
                                                    random_state=121)

# 학습 수행
dt_clf.fit(X_train, y_train)

# 예측 수행
pred = dt_clf.predict(X_test)

# 예측 정확도 출력
print('예측 정확도 : {0:.4f}'.format(accuracy_score(y_test, pred)))
```

    예측 정확도 : 0.9556


___
### DataFrame/Series로 분할하기
Numpy 뿐만 아니라 Pandas DataFrame/Series도 train_test_split() 사용해서 분할 가능  
위에서 했던 2. 학습/테스트 데이터 세트로 분리하고 예측 : sklearn 과 결과는 같음


```python
import pandas as pd
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df['target'] = iris_data.target  # 타겟값 넣어주기
iris_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>
</div>




```python
# 피처 데이터 세트
ftr_df = iris_df.iloc[:, :-1] # 마지막 컬럼 -1 까지 추출
ftr_df

# 레이블 데이터 세트 (target 값)
tgt_df = iris_df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(ftr_df,
                                                    tgt_df,
                                                    test_size=0.3,
                                                    random_state=121)


```


```python
print(type(X_train),'\n', type(X_test),'\n', type(y_train),'\n', type(y_test))

# X_train : 학습용 피처 데이터
# X_test : 테스트용 피처 데이터
```

    <class 'pandas.core.frame.DataFrame'> 
     <class 'pandas.core.frame.DataFrame'> 
     <class 'pandas.core.series.Series'> 
     <class 'pandas.core.series.Series'>



```python
# 학습 / 예측 / 예측 정확도 출력

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)

# 예측 정확도 출력
print('예측 정확도 : {0:.4f}'.format(accuracy_score(y_test, pred)))
```

    예측 정확도 : 0.9556


___
# 교차 검증 : K-Fold와 Stratified K-Fold

#### 학습/테스트 데이터 세트 분리 시 문제점
- 부적합 데이터 선별로 인한 알고리즘 성능 저하
- 과적합 문제 발생 (overfitting)

#### 과적합 (overfitting)
모델이 학습 데이터에 과도하게 최적화되어 다른 데이터로 실제 예측을 수행할 경우 예측 성능이 과도하게 떨어지는 것

#### 과적합 문제 발생
고정된 학습 데이터와 테스트 데이터에만 최적의 성능을 발휘할 수 있도록 편향되게 모델을 유도하는 경향 발생  
결국, 해당 테스트데이터에만 과적합되는 학습 모델이 만들어져서 다른 테스트용 데이터가 들어올 경우 성능 저하 발생

ML은 데이터에 기반하고   데이터는 이상치, 분포도, 다양한 속성 값, 피처 중요도 등 ML 에 영향을 미치는 다양한 요소를 가지고 있음  
특정 ML 알고리즘에 최적으로 동작할 수 있도록 데이터를 선별해서 학습한다면 실제 데이터 양식과 많은 차이가 있을 것이고  
결국 성능 저하로 이어짐  
  
문제점 개성 ---> 교차 검증을 이용해서 다양한 학습 평가 수행

#### 교차 검증
데이터 편중을 막기 위해 별도의 여러 세트로 구성된 학습 데이터 세트와 검증 데이터 세트에서 학습과 평가를 수행하는 것  
학습 후, 학습된 데이터로 여러 번 검증 수행  
각 세트에서 수행한 평가 결과에 따라 하이퍼 파라미터 튜닝 등의 모델 최적화 쉽게 가능

#### ML 모델의 성능 평가 프로세스
- 교차 검증 기반으로 1차 평가 수행 후 최종적으로 테스트 데이터 세트에 적용해 평가
- ML에 사용되는 데이터 세트를 세분화 해서 학습, 검증, 테스트 세트로 분리
- 테스트 데이터 세트 외에 별도의 검증 데이터 세트를 둬서 최종 평가 이전에 학습된 모델을 다양하게 평가하는데 사용

#### 교차 검증 방법
1. K-폴드 교차 검증
2. Stratified K 폴드 교차 검증

___
# K-폴드 교차 검증
K개의 데이터 폴드 세트를 만들어서 K번만큼 각 폴드 세트에 학습과 검증 평가를 반복적으로 수행  
가장 보편적으로 사용되는 교차 검증 방법임

K = 5 라면  
5개의 폴드된 데이터 세트를 학습용과 검증용으로 변경하면서 5번 평가 수행 후  
5개의 평균한 결과로 예측 성능 평가  

K 폴드 교차 검증 프로세스 구현을 위한 사이킷런 클래스
1. K-Fold 클래스 : 폴드 세트로 분리하는 객체 생성
    - kfold = KFold(n_splits=5)
2. split() 메소드 : 폴드 데이터 세트로 분리
    - kfold.split(features)
    - 각 폴드마다 학습용, 검증용 데이터 추출하고 학습 및 예측 수행, 정확도 측정
3. 최종 평균 정확도 계산

## K-폴드 교차 검증 예제


```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target

print('붓꽃 데이터 세트 크기 :', features.shape[0])
```

    붓꽃 데이터 세트 크기 : 150



```python
features.shape # 150개 데이터, 피처 4개
```




    (150, 4)



### k-fold 교차검증으로 5개 폴드 세트로 분리하기


```python
# DecisionTreeClassifier 객체 생성
dt_clf = DecisionTreeClassifier(random_state=156)

# 5개의 폴드 세트로 분리하는 KFold 객체 생성
kfold = KFold(n_splits = 5)

# 폴드 세트별로 정확도를 저장할 리스트 객체 생성
cv_accuracy = []
```

features : 150개 데이터  
5개로 나누므로 학습용 데이터는 120, 검증용 데이터는 30  
KFold 객체의 split() 메소드를 사용해서 폴드 데이터 세트로 분리    
폴드별 학습용, 검증용 데이터 세트의 행 인덱스 반환 출력 확인하기  


```python
# 폴드별 학습용, 검증용 데이터 세트의 행 인덱스 확인하기
for train_index, test_index in kfold.split(features):
    print('train_index :\n', train_index, '\n', 'test_index :\n', test_index)
```

    train_index :
     [ 30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
      48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65
      66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83
      84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 101
     102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119
     120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137
     138 139 140 141 142 143 144 145 146 147 148 149] 
     test_index :
     [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24 25 26 27 28 29]
    train_index :
     [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
      18  19  20  21  22  23  24  25  26  27  28  29  60  61  62  63  64  65
      66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83
      84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 101
     102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119
     120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137
     138 139 140 141 142 143 144 145 146 147 148 149] 
     test_index :
     [30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53
     54 55 56 57 58 59]
    train_index :
     [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
      18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
      36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
      54  55  56  57  58  59  90  91  92  93  94  95  96  97  98  99 100 101
     102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119
     120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137
     138 139 140 141 142 143 144 145 146 147 148 149] 
     test_index :
     [60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83
     84 85 86 87 88 89]
    train_index :
     [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
      18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
      36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
      54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71
      72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89
     120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137
     138 139 140 141 142 143 144 145 146 147 148 149] 
     test_index :
     [ 90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
     108 109 110 111 112 113 114 115 116 117 118 119]
    train_index :
     [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
      18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
      36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
      54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71
      72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89
      90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
     108 109 110 111 112 113 114 115 116 117 118 119] 
     test_index :
     [120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137
     138 139 140 141 142 143 144 145 146 147 148 149]


이 다음에 어떤 순서로 진행할 것인가...  
1. 각 폴드 별 학습용, 검증용 데이터 추출하기
2. 학습 및 예측하기
3. 정확도 측정하기 -> 리스트에 저장
4. 평균 검증 정확도 확인


```python
# 반복문이라서 단순 횟수 증가하는 생성값 만들어줌
n_iter = 0

# K 가 5 이므로 5번 반복
for train_index, test_index in kfold.split(features):
    # 1) 각 폴드 별 학습용, 검증용 데이터 확인하기
    X_train, X_test = features[train_index], features[test_index]  # 피처 데이터
    y_train, y_test = label[train_index], label[test_index]  # 레이블 (타깃) 데이터

    # 2) 학습 및 예측 수행
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)

    n_iter += 1

    # 3) 반복할 때마다 정확도 측정
    accracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]  # X_train.shpae : (120, 4) [0]이라 120만 나옴
    test_size = X_test.shape[0]  # X_test.shpae : (30, 4) [0]이라 30만 나옴

    print('{0}) 교차검증 정확도 : {1} \n 학습 데이터 크기 : {2} \n 검증 데이터 크기 : {3} \n'.format(n_iter, accracy, train_size, test_size))
    print('{0}) 검증 세트 인덱스 : {1}'.format(n_iter, test_index))
    print('----------------------------------------')

    # 리스트에 저장하기
    cv_accuracy.append(accracy)

# 4) 개별 정확도를 합하여 평균 정확도 계산
print('※ 평균 검증 정확도 :', np.mean(cv_accuracy))
```

    1) 교차검증 정확도 : 0.0 
     학습 데이터 크기 : 100 
     검증 데이터 크기 : 50 
    
    1) 검증 세트 인덱스 : [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
     48 49]
    ----------------------------------------
    2) 교차검증 정확도 : 0.0 
     학습 데이터 크기 : 100 
     검증 데이터 크기 : 50 
    
    2) 검증 세트 인덱스 : [50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73
     74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97
     98 99]
    ----------------------------------------
    3) 교차검증 정확도 : 0.0 
     학습 데이터 크기 : 100 
     검증 데이터 크기 : 50 
    
    3) 검증 세트 인덱스 : [100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117
     118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135
     136 137 138 139 140 141 142 143 144 145 146 147 148 149]
    ----------------------------------------
    ※ 평균 검증 정확도 : 0.3222222222222222


___
# Stratified K 폴드 교차 검증

**불균형한 분포도**를 가진 레이블 (결정 클래스) 데이터 집합을 위한 K폴드 방식임  
데이터 분포도가 불균형하게 퍼져있을 때, 분산되어 있을 때 사용함  
예를들어 특정 레이블 값이 특이하게 많거나 매우 적어서 값의 분포가 한 쪽으로 치우친 경우  
**학습 데이터와 검증 데이터 세트가 가지는 레이블 분포도가 유사하도록 검증 데이터 추출**  
원본 데이터의 레이블 분포도를 먼저 고려한 뒤, 이 분포와 동일하게 학습 데이터와 검증 데이터를 분배함  
KFold로 분할된 레이블 데이터 세트가 전체 레이블 값의 분포도를 반영하지 못하는 문제를 해결하는 방법임  

#### Stratified K 폴드 교차 검증 방법이 사용되는 예시
- 대출 사기 데이터 예측
- 데이터 세트 : 1억건
- feature 수십 개
- 대출 사기 여부를 뜻하는 레이블 : 대출사기 1, 정상 대출 0
- 대출 사기 건수가 매우 적음 약 1000건 (전체의 0.00001%)

이렇게 작은 비율로 1 레이블 값이 있다면 K폴드 랜덤하게 학습/테스트 데이터 세트의 인덱스를 고르더라도 레이블 값인 0과 1 비율을 제대로 반영하지 못하는 경우가 쉽게 발생함  
즉, 레이블 값으로 1이 특정 개별 반복별 학습/테스트 데이터 세트에는 상대적으로 많이 들어 있어도 다른 반복 학습/테스트 데이터 세트에는 적게 포함되어 있을 수 있음
  
그러나 대출 사기 레이블이 1인 레코드는 비록 건수는 적지만 알고리즘이 대출 사기를 예측하기 위한 중요한 피처값을 가지고 있음! 때문에 매우 중요한 데이터 세트임  
대출 사기 레이블 값의 분포를 원본 데이터의 분포와 유사하게 학습/테스트 데이터 세트에서도 유지하는 게 매우 중요함  
  
따라서 원본 데이터의 레이블 분포를 먼저 고려한 뒤 이 분포와 동일하게 학습과 검증 데이터 세트를 분배하는 방식인 Stratified K 폴드 교차 검증 방법을 사용해서 예측

## Stratified K 폴드 교차 검증 예제
먼저 K폴드 문제점을 확인하고, 사이킷런의 Stratified K 폴드 교차 검증 방법으로 개선  
붓꽃 데이터 세트를 데이터 프레임으로 생성하고 레이블 값의 분포도 확인

### Stratified K 폴드 교차 검증을 그냥 K폴드로 하는 경우


```python
import pandas as pd

iris = load_iris()

iris_df = pd.DataFrame(data = iris_data, columns=iris_data.feature_names)
iris_df['lable'] = iris_data.target
iris_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>lable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 각 레이블 값 개수 확인
iris_df['lable'].value_counts()

# 결과 : 레이블 값은 0, 1, 2 가 모두 50개로 동일함
```




    2    50
    1    50
    0    50
    Name: lable, dtype: int64




```python
# 3개의 폴드 세트 생성
kfold = KFold(n_splits=3)

# 반복문이라서 단순 횟수 증가하는 생성값 만들어줌
n_iter = 0

for train_index, test_index in kfold.split(iris_df):
    n_iter += 1
    
    label_train = iris_df['lable'].iloc[train_index]
    label_test = iris_df['lable'].iloc[test_index]
    
    print('교차 검증 : {0}'.format(n_iter))
    print('학습 레이블 데이터 분포 : \n', label_train.value_counts())
    print('검증 레이블 데이터 분포 : \n', label_test.value_counts())
    print('-----------------------------')
    
```

    교차 검증 : 1
    학습 레이블 데이터 분포 : 
     2    50
    1    50
    Name: lable, dtype: int64
    검증 레이블 데이터 분포 : 
     0    50
    Name: lable, dtype: int64
    -----------------------------
    교차 검증 : 2
    학습 레이블 데이터 분포 : 
     2    50
    0    50
    Name: lable, dtype: int64
    검증 레이블 데이터 분포 : 
     1    50
    Name: lable, dtype: int64
    -----------------------------
    교차 검증 : 3
    학습 레이블 데이터 분포 : 
     1    50
    0    50
    Name: lable, dtype: int64
    검증 레이블 데이터 분포 : 
     2    50
    Name: lable, dtype: int64
    -----------------------------


#### 결과
교차 검증 할 때마다 3개의 폴드 세트로 만들어지는 학습 레이블과 검증 레이블이 완전히 다른 값으로 추출되었음  

첫 번째 교차 검증  
학습 레이블의 1, 2 값이 각 50개 나옴  
검증 레이블에서는 0의 값이 50개 나옴  
-> 학습 레이블은 1, 2 밖에 없으므로 0의 경우를 전혀 학습하지 못함 검증 레이블은 0밖에 없으므로 학습 모델은 절대 0을 예측하지 못함  
  
이런 유형으로 교차 검증 데이터 세트를 분할하면 검증 예측 정확도는 0이 됨  

### Stratified K 폴드 교차 검증 방식으로 iris 데이터 교차 검증
Stratified KFold 클래스 사용  
동일한 데이터 분할을 StratifiedKFold로 수행하고 학습/검증 레이블 데이터 분포도 확인  
단 하나의 큰 차이는 레이블 데이터 분포도에 따라 학습/검증 데이터를 나누기 때문에 split() 메서드 인자로 피처 데이터와 함께 반드시 레이블 데이터 반드시 필요하다는 것


```python
# 동일한 데이터 분할을 StratifiedKFold로 수행하고 학습/검증 레이블 데이터 분포도 확인  
# 단 하나의 큰 차이는 레이블 데이터 분포도에 따라 학습/검증 데이터를 나누기 때문에 split() 메서드 인자로 피처 데이터와 함께 반드시 레이블 데이터 반드시 필요하다는 것

from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits = 3)  # 폴드 세트 3개
n_iter = 0

# 레이블 데이터 세트도 반드시 인자로 사용
for train_index, test_index in skfold.split(iris_df, iris_df['lable']):
    n_iter += 1
    label_train = iris_df['lable'].iloc[train_index]
    label_test = iris_df['lable'].iloc[test_index]
    
    print('교차 검증 : {0}'.format(n_iter))
    print('학습 레이블 데이터 분포 : \n', label_train.value_counts())
    print('검증 레이블 데이터 분포 : \n', label_test.value_counts())
    print('-----------------------------')
    
# 출력 결과
# 학습 레이블과 검증 레이블 데이터 값의 분포도가 동일하게 할당
# 학습 레이블 : 0, 1, 2 - 33, 33, 34
# 검증 레이블 : 0, 1, 2 - 17, 17, 16

# 이렇게 분할 되어야 레이블 값이 0, 1, 2 모두 학습할 수 있고 이에 기반해서 검증 수행 가능
```

    교차 검증 : 1
    학습 레이블 데이터 분포 : 
     2    34
    1    33
    0    33
    Name: lable, dtype: int64
    검증 레이블 데이터 분포 : 
     1    17
    0    17
    2    16
    Name: lable, dtype: int64
    -----------------------------
    교차 검증 : 2
    학습 레이블 데이터 분포 : 
     1    34
    2    33
    0    33
    Name: lable, dtype: int64
    검증 레이블 데이터 분포 : 
     2    17
    0    17
    1    16
    Name: lable, dtype: int64
    -----------------------------
    교차 검증 : 3
    학습 레이블 데이터 분포 : 
     0    34
    2    33
    1    33
    Name: lable, dtype: int64
    검증 레이블 데이터 분포 : 
     2    17
    1    17
    0    16
    Name: lable, dtype: int64
    -----------------------------


### 출력 결과
학습 레이블과 검증 레이블 데이터 값의 분포도가 동일하게 할당  
학습 레이블 : 0, 1, 2 - 33, 33, 34  
검증 레이블 : 0, 1, 2 - 17, 17, 16  
  
이렇게 분할 되어야 레이블 값이 0, 1, 2 모두 학습할 수 있고 이에 기반해서 검증 수행 가능  


```python
# Stratified KFold 방식으로 붓꽃 데이터 교차 검증
# Stratified KFold 클래스 사용
from sklearn.model_selection import StratifiedKFold

# DecisionTreeClassifier 객체 생성
dt_clf = DecisionTreeClassifier(random_state=156)

skfold = StratifiedKFold(n_splits = 3)
n_iter = 0
cv_accuracy = []

# StratifiedKFold의 split() 호출할 때 피처 데이터와 함께 반드시 레이블 데이터 세트도 추가 입력
for train_index, test_index in skfold.split(features, label):   # 여기서  label 데이터 셋을 넣는 점이 다르다
    # 1) 각 폴드 별 학습용, 검증용 데이터 확인하기
    X_train, X_test = features[train_index], features[test_index]  # 피처 데이터
    y_train, y_test = label[train_index], label[test_index]  # 레이블 (타깃) 데이터

    # 2) 학습 및 예측 수행
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)

    n_iter += 1

    # 3) 반복할 때마다 정확도 측정
    accracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]  # X_train.shpae : (120, 4) [0]이라 120만 나옴
    test_size = X_test.shape[0]  # X_test.shpae : (30, 4) [0]이라 30만 나옴

    print('{0}) 교차검증 정확도 : {1} \n 학습 데이터 크기 : {2} \n 검증 데이터 크기 : {3} \n'.format(n_iter, accracy, train_size, test_size))
    print('{0}) 검증 세트 인덱스 : \n {1}'.format(n_iter, test_index))
    print('--------------------------------------------')

    # 리스트에 저장하기
    cv_accuracy.append(accracy)

# 4) 개별 정확도를 합하여 평균 정확도 계산
print('※ 교차 검증별 정확도 :', np.round(cv_accuracy,4))
print('※ 평균 검증 정확도 :', np.mean(cv_accuracy))
```

    1) 교차검증 정확도 : 0.98 
     학습 데이터 크기 : 100 
     검증 데이터 크기 : 50 
    
    1) 검증 세트 인덱스 : 
     [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  50
      51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66 100 101
     102 103 104 105 106 107 108 109 110 111 112 113 114 115]
    --------------------------------------------
    2) 교차검증 정확도 : 0.94 
     학습 데이터 크기 : 100 
     검증 데이터 크기 : 50 
    
    2) 검증 세트 인덱스 : 
     [ 17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  67
      68  69  70  71  72  73  74  75  76  77  78  79  80  81  82 116 117 118
     119 120 121 122 123 124 125 126 127 128 129 130 131 132]
    --------------------------------------------
    3) 교차검증 정확도 : 0.98 
     학습 데이터 크기 : 100 
     검증 데이터 크기 : 50 
    
    3) 검증 세트 인덱스 : 
     [ 34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  83  84
      85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 133 134 135
     136 137 138 139 140 141 142 143 144 145 146 147 148 149]
    --------------------------------------------
    ※ 교차 검증별 정확도 : [0.98 0.94 0.98]
    ※ 평균 검증 정확도 : 0.9666666666666667


##  Stratified K 폴드 교차 검증 정리
원본 데이터의 레이블 분포도 특성을 반영한 학습 및 검증 데이터 세트를 만들 수 있으므로  
왜곡된 레이블 데이터 세트에서는 반드시  Stratified K 폴드를 이용해서 교차 검증해야 함  
일반적으로 분류(Classification)에의 교차 검증은 K폴드가 아니라  Stratified K 폴드로 분할되어야 함  
회귀(Regression)에서  Stratified K 폴드가 지원되지 않음  
이유는 회귀의 결정 값은 이산값 형태의 레이블이 아니라 연속된 숫자값이기 때문에 결정값별로 분포를 정하는 의미가 없기 때문

___
# 사이킷런 API를 사용해 교차검증

## 교차 검증 (Cross Validation) 과정
1. 폴드 세트 설정
2. for 문에서 반복적으로 학습 및 검증 데이터 추출하고 학습/예측 수행
3. 폴드 세트 별로 예측 성능을 평균하여 최종 성능 평가

## 교차 검증을 보다 간편하게 해준 사이킷런 API
- cross_val_score() 함수
- (1) ~ (3) 단계의 교차 검증 과정을 한꺼번에 수행
- 내부에서 Estimator를 학습(fit), 예측(predict), 평가(evaluation) 시켜주므로 간단하게 교차 검증 수행 가능

### cross_val_score() 주요 파라미터
- estimator : Classifier 또는 Regressor (분류 또는 회귀)
- X : 피처 데이터 세트
- y : 레이블 데이터 세트
- scoring : 예측 성능 평가 지표
- cv : 교차 검증 폴드 수
- cross_val_score(dt_clf, data, label, scoring='accuracy‘, cv=3)
- cv로 지정된 횟수 만큼 scoring 파라미터로 지정된 평가 지표로 평가 결과값을 배열로 반환
- 일반적으로 평가 결과값 평균을 평가 수치로 사용


```python
# cross_val_score()
# 교차 검증 폴드 수 : 3
# 성능 평가 지표 : accuracy (정확도)

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
import numpy as np

iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)

data = iris_data.data
labela = iris_data.target

# 성능 지표 : accuracy(정확도), 교차 검증 폴드 수 : 3
scores = cross_val_score(dt_clf, data, label, scoring = 'accuracy', cv = 3)

print('교차 검증별 정확도 :', scores)
print('평균 검증 정확도 :', np.round(np.mean(scores), 4))
```

    교차 검증별 정확도 : [0.98 0.94 0.98]
    평균 검증 정확도 : 0.9667


#### cross_val_score() 수행 결과
앞 예제 StratifiedKFold를 이용해 붓꽃 데이터 교차 검증 결과와 동일함  


___
# 하이퍼 파라미터 (Hyper Parameter)
모델링할 때 사용자가 직접 세팅해 주는 값  
여러 하이퍼 파라미터를 순차적으로 변경하면서 최고 성능을 가지는 파아미터 조합을 찾을 수 있음  
예) max_depth, min_samples_split, iteration 등  
머신러닝 알고리즘을 구성하는 주요 구성 요소임  
이 값들을 조정해서 알고리즘의 예측 성능을 개선할 수 있음

## GridSearchCV 클래스
**교차 검증과 최적 하이퍼 파라미터 튜닝을 한번에**  
사이킷런에서는 GridSearchCV 클래스를 이용해서 Classifier나 Regressor 같은 알고리즘에 사용되는 하이퍼 파라미터를 순차적으로 입력하면서 최적의 파라미터를 편리하게 도출할 수 있는 방법을 제공함  
Grid는 격자라는 의미임 촘촘하게 파라미터를 입력하면서 테스트 하는 방식
  
#### 최적의 하이퍼 파라미터를 찾는 방법  
머신러닝 알고리즘의 여러 하이퍼 파라미터를 순차적으로 변경하면서 최고 성능을 가지는 파라미터를 찾으려면 파아미터의 집합을 만들어서 순차적으로 적용하면서 최적화 수행  
성능이 최고일 때의 하이퍼 파라미터가 최적의 파라미터가 됨  

### GridSearchCV 클래스 생성자의 주요 파라미터
- estimator : classifier, regressor, peipeline
- param_grid : key + 리스트 값을 가지는 딕셔너리 (estimator 튜닝을 위한 하이퍼 파라미터 )
  - key: 파라미터명, 리스트값:파라미터 값
- scoring : 예측 성능을 측정할 평가 방법
  - 성능 평가 지표를 지정하는 문자열 (예: 정확도인 경우 'accuracy')
- cv : 교차 검증을 위해 분할되는 학습/테스트 세트의 개수
- refit : 최적의 하이퍼 파라미터를 찾은 뒤
  - 입력된 estimator 객체를 해당 하이퍼 파라미터로 재학습 여부
  - 디폴트 : True
