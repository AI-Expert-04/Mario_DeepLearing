# Mario_DeepLearing (유전알고리즘과 인공신경망을 활용한 게임 학습)

## 환경 설정
```bash
>>> Window python=3.6
>>> Mac python=3.7
```

# 마리오 학습 정리
![tabel](image/image77.png) 
### 인공신경망

### 선택
- 엘리트 보존 선택 2개
- 룰렛 휠 생성으로 8개 생성

### 교배
- SBX(일점 교차)를 통해 교배

### 변이
- 정적 돌연 변이
- 가우시안 돌연 변이

### 결과
세대 : 7703세대,

시간 : 7일

느낀점 : 엘리트 보존을 사용하여 퇴화를 막았지만 계단식 성장을 하여 언젠간 성공하지만 언젠가가 오지 않을 까 두렵다.

### 핵심코드
##### Genentic_Algorithm.py
<pre><code>    relu = lambda X: np.maximum(0, X) # 단층, Hidden_layer
    sigmoid = lambda X: 1.0 / (1.0 + np.exp(-X)) # Output_layer
    
    class Chromosome:  # 염색체
    def __init__(self):
        # 4개의 유전자가 모여 하나의 염색체가 됨

        self.w1 = np.random.uniform(low=-1, high=1, size=(80, 9))
        self.b1 = np.random.uniform(low=-1, high=1, size=(9,))

        self.w2 = np.random.uniform(low=-1, high=1, size=(9, 6))
        self.b2 = np.random.uniform(low=-1, high=1, size=(6,))
        
        def predict(self, data): # 예측
        # data = Input_layer;
        self.l1 = relu(np.matmul(data, self.w1) + self.b1) # 행렬곱
        # sigmoid = Output_layer
        output = sigmoid(np.matmul(self.l1, self.w2) + self.b2) # 행렬곱
        # output = [-1 ~ 1, -1 ~ 1, -1 ~ 1, -1 ~ 1, -1 ~ 1, -1 ~ 1]
        # -1 ~ 1 사이의 값을 가진 6개의 output 중 0.5보다 크면 1로 바꾸고 아니면 0으로 출력
        result = (output > 0.5).astype(np.int)
        return result
        자세한 코드는 Genentic_Algorithm.py</code></pre>       
# 마리오 게임
![tabel](image/Mario_DeepLearning_Image.png)
# 마리오 게임 학습 완료 영상
YouTube [Link](https://www.youtube.com/watch?v=icxwqmojT18)
# Pyqt5로 그린 인공신경망 
![tabel](image/label_.png)
# 인공신경망 예시
![tabel](image/label5.png) 
# 활성화 함수 ReLU, Sigmoid
![tabel](image/label3.png)
# Pyqt5로 그린 게임 타일 정보
![tabel](image/label2.png)
# TSP알고리즘
![tabel](image/TSP_Image.png)




