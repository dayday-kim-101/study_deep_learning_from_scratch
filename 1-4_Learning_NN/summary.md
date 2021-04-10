# Learning Neural Network

<br>

### Entropy
잘 일어나지 않는 사건(Unlikely Event)는 정보량이 크다(많다)!

즉, 확률이 낮을수록 큰 값이되는 지표!

ex)

$$
P(x) -> \frac{1}{P(x)}, \frac{1}{1000} -> 1000
$$

$$
log\frac{1}{P(x)} = -logP(x)
$$  

이렇게 log를 씌우는게 일반적!

데이터 분석 시 로그를 취하는 이유

- 정규성을 높이고, 정확한 값을 얻기 위함!
- 큰 값을 작게 만들어주고, 복잡한 계산을 쉽게해준다.
- 왜도와 첨도를 줄여서 분석 시 의미있는 결과를 도출한다.

(왜도: 데이터가 한쪽으로 치우친 정도, 첨도: 분포가 얼마나 뾰족한지를 나타내는 정도)

$$
log\frac{1}{P(x)} \left\{\begin{matrix}
\rightarrow {log_{2}}\frac{1}{P(x)} \\
\rightarrow {log_{e}}\frac{1}{P(x)}
\end{matrix}\right.
$$

로그 밑수를 2로 하는 것을 섀넌(shannon), 비트 방법

지수(e)를내트(Nert) 방법이라고 함

<br>

$$H(P) = H(x) = -\sum p(x)logp(x)$$

확률분포 P에 대한 Entropy!

x는 확률변수

<br>

### KL Divergence

sd

<br>

학습 시 미분가능여부를 봐야함! 미분이 되야 학습이 되지 (당연한얘기임...)

Logic -> 실수를 Log로 변환한다. (Log는 비율로 나타낼 수 있지. 0~1이 확률을 의미할 수 있지)

ex) 3 -> exp(3) 이 낫다 이거지.


<br>

softmax 보다 sigmoid 많이 씀
