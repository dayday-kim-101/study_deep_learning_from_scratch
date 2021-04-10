# 1-5 Backpropagation

시그모이드 사용 시 오버플로우 방(여러 element에 적용하는 것 확인중)

x >= 0 (1 / (1 + np.exp(-x)))

x < 0 (exp(x) / 1 + exp(x))

*이걸 합쳐서 구현한 max 어쩌구 있다고함 (찾아봐야됨)


