# 1-2 Perceptron
Perceptron은기본적으로 다음과 같이 입력과 가중치의 계산이 theta를 넘으면 1, 넘지않으면 0을 출력한다.

<center>

![equation](http://latex.codecogs.com/gif.latex?y%3D%5Cbegin%7Bcases%7D%200%28w_%7B1%7Dx_%7B1%7D&plus;w_%7B2%7Dx_%7B2%7D%29%20%5Cleq%20%5Ctheta%20%5C%5C%201%28w_%7B1%7Dx_%7B1%7D&plus;w_%7B2%7Dx_%7B2%7D%29%20%3E%20%5Ctheta%20%5Cend%7Bcases%7D)
<!--
y=\begin{cases}
 0(w_{1}x_{1}+w_{2}x_{2}) \leq \theta   \\ 
 1(w_{1}x_{1}+w_{2}x_{2}) >  \theta   
\end{cases}
-->
</center>
<div style="text-align: right">
(1-2.1)
</div>
<br>
<br>


여기서 𝜃를 -b로 치환하면, 다음과 같다. 즉, bias, 입력, 가중치의 계산이 0을 넘으면 1, 넘지않으면 0을 출력한다.

bias가 뉴런이 얼마나 쉽게 활성화되는지 결정한다.

(ex, bias가 -2000이면 나머지 애들이 2000이나 넘어야 출력이 1이되어 활성화한다!)

<center>

![equation](http://latex.codecogs.com/gif.latex?y%3D%5Cbegin%7Bcases%7D%200%28b%20&plus;w_%7B1%7Dx_%7B1%7D&plus;w_%7B2%7Dx_%7B2%7D%29%20%5Cleq%200%20%5C%5C%201%28b%20&plus;w_%7B1%7Dx_%7B1%7D&plus;w_%7B2%7Dx_%7B2%7D%29%20%3E%200%20%5Cend%7Bcases%7D)
<!--
y=\begin{cases}
 0(b +w_{1}x_{1}+w_{2}x_{2}) \leq 0  \\
 1(b +w_{1}x_{1}+w_{2}x_{2}) >  0
\end{cases}
-->
</center>
<div style="text-align: right">
(1-2.2)
</div>




## The weakness


* 단층 퍼셉트론으론 XOR게이트 못 함

* Why? 선형이라 구체적인 구별이 안된다.

* 비선형 함수 필요 즉, 다차원으로 경계를 늘려야된다.

* But, 다층 퍼셉트론으로 처리는 가능하나 우리가 가중치를 설정해줘야 한다.
