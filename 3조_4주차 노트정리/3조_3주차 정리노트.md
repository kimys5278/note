# 3주차 정리노트

상태: 시작 전

# Q 1. 그냥 데이터를 머신러닝에 넣으면 되는데 왜 데이터를 한번씩 살펴보는 걸까?

### 유성

- 사용하고자 하는 데이터만 사용해야하는데, 굳이 필요없는 데이터까지 사용한다면 가독성이 떨어진다고 생각한다. 또한 학습을 위해 데이터 전처리를 해야하는데 전처리를 하려면 데이터가 어떤 형태로 있는지 어떠하게 바뀌는지 수시로 확인을 해야하기 때문이라고 생각한다.

### 동현

- 어떤 머신 러닝에 넣어서 사용할지 확인해봐야하기 떄문이라고 생각하기때문
예를 들어 데이터로 회귀를 하고 싶은데 데이터를 살펴봤더니 공간회귀적인 데이터라면 GWR같은 것을 사용할 수 있다

# Q 2. train & test 데이터를 왜 8대2로 나누는가 더 좋은건 없는가

### 동현

- 평균적으로 데이터 셋을 8대 2로 나누는거 같은데  일단 train데이터로 머신을 학습시키기에 당연히 train 데이터가 더 많아야하고 그렇다고 train 데이터가 너무 많으면 과적합될 수 있다고 생각해서 8:2에서 7:3중에 적정치를 고르면 된다고 생각함 데이터 ex) 0.75:0.25, 0.72:0.28 등등 소수점으로 내려가도됨다 해보고 적정치를 찾으면 될듯

### 유성

- 일반적인 데이터 셋 일 때는 8:2가 좋을 것 같지만, 적은 데이터 일 땐, Validate를 추가해 6:2:2로 나눠도 좋을 것 같다.
- 그 이유는 오버피팅을 피하기 위해서 인 것 같다.

---

# 질문

### 유성

1. 질문 

```jsx
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
```

→ HOUSING_PATH 값이 안 넘어옴.

해결 ⇒ 

```jsx
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
```

HOUSING_PATH를 지정을 안 해줬음.

2.*`# 소득 카테고리 개수를 제한하기 위해 1.5로 나눕니다.`* 

1.5로 나누는 이유는??

 → x축이 15개이고, 5까지만 그래프가 잘나와있어서.

 → 나눈 카테고리 5개를 잘 보이게 하기 위해 1.5로 나눈것 같다.

1. inplace = True ⇒ 기존 데이터프레임에 변경된 설정을 덮어 쓰겠는다는 의미.