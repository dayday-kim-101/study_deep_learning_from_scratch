import sys, os
sys.path.append("/Users/dayday/MyGithub/Deep-Learning-from-Scratch")      # 부모 디렉터리의 파일을 가져울 수 있도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 처음 한 번은 몇 분 정도 걸린다.
# 데이터 가져오기
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test


def init_network():
    with open("./1-3_Neural_Network/src/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    B1, B2, B3 = network['b1'], network['b2'], network['b3']
    W1, W2, W3 = network['W1'], network['W2'], network['W3']

    a1 = np.dot(x, W1) + B1
    z1 = sigmoid(a1)
#    print(z1.shape)
    
    a2 = np.dot(z1, W2) + B2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + B3
 
    y = softmax(a3)

    return y

'''
print(x_train.shape)
print(t_train.shape)

print(x_test.shape)
print(t_test.shape)
'''
'''
img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
'''
x, t = get_data()
print(x.shape)
print(x[0].shape)
network = init_network()

batch_size = 100                # for batch
accuracy_cnt = 0


for i in range(len(x)):
    x_batch = x[i:i+batch_size]         # for batch
    y_batch = predict(network, x_batch) # for batch
    '''
    y = predict(network, x[i])
    p = np.argmax(y)            # 확률이 가장 높은 원소의 인덱스 얻음
    if p == t[i]:
        accuracy_cnt += 1
    '''
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
print("Accuracy: " + str(float(accuracy_cnt) / len(x)))

sys.path.remove("/Users/dayday/MyGithub/Deep-Learning-from-Scratch")
