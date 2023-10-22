# 7주차 정리노트 및 과제

상태: 시작 전

-유성

- 유성
    - 사이킷런의 `export_graphviz()` 함수는 결정 트리 모델을 시각화하는 데 사용되는 함수입니다.
    - 클래스 확률 추정(Class Probability Estimation)은 분류 모델에서 각 클래스에 속할 확률을 추정하는 것을 의미합니다.
    
    *문제: moons 데이터셋에 결정 트리를 훈련시키고 세밀하게 튜닝해보세요.*
    
    # 결정트리 (Decision Tree)
    
    결정트리는 분류와 회귀 문제, 다중 출력 작업에 모두 사용할 수 있는 머신러닝 알고리즘입니다. 랜덤 포레스트의 기본 구성 요소이기도 합니다.
    
    ## 결정트리 결과해석 - 지니 (Gini)
    
    결정트리에서 각 노드의 불순도를 측정하는 값으로 지니 계수를 사용합니다. 지니 계수는 결정트리 알고리즘의 비용 함수에 사용됩니다.
    
    ## CART 훈련 알고리즘
    
    결정트리를 훈련시키기 위해 CART(Classification And Regression Trees) 훈련 알고리즘을 사용합니다. CART 알고리즘은 탐욕적 알고리즘(Greedy Algorithm)을 활용하여 특성과 임곗값을 선택합니다. 탐욕적 알고리즘은 해당 노드에 포함된 샘플을 가장 순수한 두 개의 부분집합으로 분할하는 결정을 내립니다.
    
    ## 계산 복잡도
    
    결정트리를 사용하여 예측을 수행할 때는 루트 노드에서부터 리프 노드까지 탐색해야 합니다. 이 과정의 계산 복잡도는 O(log2(m))을 가집니다. 따라서 특성의 수와는 무관하게 예측 속도가 매우 빠릅니다.
    
    ## 규제 매개변수
    
    결정트리를 훈련시킬 때는 모델 파라미터를 조정하여 규제할 수 있습니다. 규제를 함으로써 모델의 자유도가 제한되고 과대적합 위험이 줄어들지만, 과소적합 위험이 커질 수 있습니다.
    
    ## Moons 데이터셋에 결정트리 훈련하기
    
    1. `make_moons(n_samples=1000, noise=0.4)`를 사용하여 데이터셋을 생성합니다.
    2. `train_test_split()` 함수를 사용하여 훈련 세트와 테스트 세트로 나눕니다.
    3. `DecisionTreeClassifier`의 최적의 매개변수를 찾기 위해 교차 검증과 그리드 탐색을 수행합니다.
    
    a. `make_moons(n_samples=10000, noise=0.4)`를 사용해 데이터셋을 생성합니다.
    
    `random_state = 42`를 추가하여 이 노트북의 출력을 일정하게 만듭니다:
    
    ```python
    #7-a . make_moon ( n_sample=1000, noise=0.4)를 사용해 데이터셋을 생성합니다
    **from** sklearn.datasets **import** make_moons
    X_moons, y_moons **=** make_moons(n_samples**=a**10000, noise**=**0.4, random_state**=**42)
    ```
    
    <aside>
    💡 Out[3]:
    
    ```
    C:\Users\User\anaconda3\anaconda33\lib\site-packages\scipy\__init__.py:155: UserWarning: A NumPy version >=1.18.5 and <1.25.0 is required for this version of SciPy (detected version 1.26.0
      warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
    ```
    
    </aside>
    
    b. 이를 `train_test_split()`를 사용해 훈련 세트와 테스트 세트로 나눕니다.
    
    ```python
    #7-b. 이를 train_test_split( )을 사용해 훈련 세트와 테스트 세트로 나눈다
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons,
                                                        test_size=0.2,
                                                        random_state=42)
    ```
    
    c. `DecisionTreeClassifier`의 최적의 매개변수를 찾기 위해 교차 검증과 함께 그리드 탐색을 수행합니다(`GridSearchCV`를 사용하면 됩니다). 힌트: 여러 가지 `max_leaf_nodes` 값을 시도해보세요.
    
    ```python
    #37-c. DecisionTreeClassifier의 최적의 매개변수를 찾기 위해 교차 검증과 함께 그리드 탐색을 수행한다.(GridSearchCV) 힌트 : 여러가지 max_leaf_nodes 값을 시도해보세요!
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV
    
    tree_clf = DecisionTreeClassifier(random_state=42)
    
    params = {'max_leaf_nodes' : list(range(2,100)),'min_samples_split' : [2,3,4]}
    
    grid_search_cv = GridSearchCV(tree_clf,param_grid=params,cv=3,n_jobs=-1,verbose=1)
    
    grid_search_cv.fit(X_train,y_train)
    ```
    
    Out :
    
    <aside>
    💡 `Fitting 3 folds for each of 294 candidates, totalling 882 fits`
    
    Out[3]:
    
    `GridSearchCV(cv=3, estimator=DecisionTreeClassifier(random_state=42), n_jobs=-1,
                 param_grid={'max_leaf_nodes': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                31, ...],
                             'min_samples_split': [2, 3, 4]},
                 verbose=1)`
    
    </aside>
    
    ```python
    grid_search_cv.best_estimator_
    ```
    
    <aside>
    💡 `DecisionTreeClassifier(max_leaf_nodes=4, random_state=42)`
    
    </aside>
    
    ```python
    from sklearn.metrics import accuracy_score
    
    # 별도의 훈련이 필요없다. GridSearchCV는 최적의 모델로 다시 훈련시키기 때문이다.
    
    y_pred = grid_search_cv.best_estimator_.predict(X_test)
    accuracy_score(y_test,y_pred)
    ```
    
    <aside>
    💡 `0.855`
    
    </aside>
    

- 동현

### 결정트리

- 분류와 회귀, 다중 출력 작업 가능
- 랜덤 포레스트의 기본 구성 요소

### 결정트리 결과해석 - gini

- 해당 노드의 불순도 측정값
- 연속된 결정트리 학습 과정에 사용되는 알고리즘의 비용함수에 사용

### CART 훈련 알고리즘

- 결정트리를 훈련시키기 위해 CART 훈련 알고리즘을 사용함
    
    → 탐욕적 알고리즘 활용
    
    → 탐욕적 알고리즘: 여러 경우 중 하나를 결정해야 할 때마다 그 순간에 최적이라고 생각되는 것을 선택해 나가는 방식
    
- 비용함수를 최소화 하는 특성 𝑘 와 해당 특성의 임곗값 𝑡𝑘 을 결정
    
    → 𝑘 와 𝑡𝑘 를 고르는 방법
    
    - 𝐽(𝑘, 𝑡𝑘 )가 작을수록 불순도가 낮은 두 개의 부분집합으로 분할됨
    - 탐욕적 알고리즘은 해당 노드에 포함된 샘플을 지니 불순도가 가장 낮은, 즉, 가장 순수한두 개의 부분집합으로 분할
    - max_depth 깊이에 다다르거나 불순도를 줄이는 분할을 더 이상 찾을 수 없을 때, 또는 다른 규제의 한계에 다다를 때까지 반복

### 계산 복잡도

- 예측을 위해선 결정 트리를 루트 노드에서부터 리프 노드까지 탐색해야함
    
    → O(log2(m))의 복잡도를 가짐
    
    - 예측에 필요한 전체 복잡도는 특성 수와 무관
    - 큰 훈련 세트를 다룰 때도 예측 속도가 매우 빠름

### 규제 매개변수

- 결정트리를 훈련 데이터로 학습시킬때 상황마다 알맞은 정의된 모델 파라미터를 이용하여 규제할 수 있다
- 매개변수를 이용해 규제를 하게 되면 자유도가 제한되고, 과대적합 위험도 줄어들게 되지만 과소적합 위험은 커진다
- 

### 다음단계를 따라 moons 데이터셋에 결정트리를 훈련시키고 세밀하게 튜닝하라

1. make_moons(n_sample=1000, nosie=0.4)를 사용해 데이터셋을 생성한다

```python
# sklearn.datasets에서 moons을 불러온다
from sklearn.datasets import make_moons

# 그리고 x, y변수에 n_samples=1000, noise=0.4을 사용해 데이터 셋을 생성한다
# 추가적으로 random_state=42를 이용하여 항상 같은 값이 나오도록한다
x, y = make_moons(n_samples=1000, noise=0.4, random_state=42) 
```

1. 이를 train_test_split()을 사용해 훈련세트와 테스트 세트로 나눈다

```python
# train_test_split()을 사용하기 위해선 또 sklearn을 이용해 불러와줘야한다
from sklearn.model_selection import train_test_split

# 그리고 훈련세트와 데스트 세트로 나눠주는데 test_size는 보통 0.2, 0.3을 해주므로 여기선 0.2를 택하고
# 이것 역시 random_state=42를 해줌으로써 같은 값이 나오도록한다
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

1. DecisionTreeClassifier의 최적의 매개변수를 찾기 위해 교차 검증과 함께 그리드 탐색을 수행한다

```python
# 그리드 탐색을 수행을하고 DecisionTreeClassifier의 최적의 매개변수를 찾기위해 또 불러온다
from sklearn.model_selection import GridSearchCV

# 탐색하려는 매개변수의 그리드를 설정한다
# 여기서 'max_leaf_nodes'의 범위를 설정할 수 있는데 그리드 탐색은 이중에 최적의 값을 찾아준다
# 일단 최대한 많은 값을 포함하기 위해 range(2, 100)으로 설정하였다
params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}

# 설정한 params로 grid_search를 진행하려면 몇가지 지정을 해줘야하는데
# 먼저 random_state=42로 DecisionTreeClassifier가 일정하게 해주고
# 설정한 params를 넣으면서 verbose를 넣어주는데 
# verbose은 학습 과정 중에 출력할 메시지의 양을 결정해주고 1은 기본적인 정보만 출력한다
# cv는 교차검증에서 데이터를 몇개의 부분집합으로 분할할지 결정해주고 기본값이 5이기에 5로 정해주었다
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=5)

# 만들어진 grid_search_cv를 분할시킨 데이터에 적합시킨다
grid_search_cv.fit(x_train, y_train)

# 그리고 best_estimator_를 통해 가장 적합한 max_leaf_nodes를 알 수 있는데 아래와 같은 결과가 나왔다
grid_search_cv.best_estimator_
```

- 결과 - 이를통해 max_leaf_nodes값이 6일때 가장 적합하다는 것을 알 수 있다

![Untitled](7%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5%E1%84%82%E1%85%A9%E1%84%90%E1%85%B3%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%200ab094b060724721a2fb44d9c00daa17/Untitled.png)

1. 찾은 매개변수를 사용해 전체 훈련 세트에 대해 모델을 훈련시키고 테스트 세트에서 성능을 측정한다

```python
# 성능을 측정하기 위해서는 accuracy_score가 필요하므로 import해준다
from sklearn.metrics import accuracy_score

# 완성된 grid_search_cv 모델에 predict(x_test)를 통해 x_test값의 예측값을 변수 y_pred에 저장해준다 
y_pred = grid_search_cv.predict(x_test)

# accuracy_score를 이용해 y_test값을 넣었을때 y_pred값의 정확도가 얼만큼인지를 확인한다
accuracy_score(y_test, y_pred)
```

- 결과 - 훈련시킨 모델에서 테스트 세트에서의 성능은 85.5%정도가 나온다

![Untitled](7%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5%E1%84%82%E1%85%A9%E1%84%90%E1%85%B3%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%200ab094b060724721a2fb44d9c00daa17/Untitled%201.png)