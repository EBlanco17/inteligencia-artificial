import numpy as np
class Sigmoide:
    def __init__(self, x):
        self.x = x
    def __call__(self):
        return 1/(1+np.exp(-self.x))
    def derivada(self):
        return self.__call__()*(1-self.__call__())

class RedNeuronal:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.w1 = np.random.rand(self.x.shape[1],4)
        self.w2 = np.random.rand(4,1)
    def feedforward(self):
        self.z = Sigmoide(np.dot(self.x, self.w1))
        self.z2 = Sigmoide(np.dot(self.z(), self.w2))
        return self.z2()
    def backprop(self):
        self.z2_error = self.y - self.z2()
        self.z2_delta = self.z2_error * self.z2.derivada()
        self.z_error = self.z2_delta.dot(self.w2.T)
        self.z_delta = self.z_error * self.z.derivada()
        self.w1 += self.x.T.dot(self.z_delta)
        self.w2 += self.z().T.dot(self.z2_delta)
    def train(self, X, y):
        self.x = X
        self.y = y
        self.feedforward()
        self.backprop()

if __name__ == "__main__":
    X = np.array(([2,9],[1,5],[3,6]), dtype=float)
    y = np.array(([92],[86],[89]), dtype=float)
    X = X/np.amax(X, axis=0)
    y = y/100
    nn = RedNeuronal(X,y)
    for i in range(600):
        print("Input: \n" + str(X))
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str(nn.feedforward()))
        print("Loss: \n" + str(np.mean(np.square(y - nn.feedforward()))))
        print("\n")
        nn.train(X, y)

    X = np.array(([4,8]), dtype=float)
    X = X/np.amax(X, axis=0)
    print("Input: \n" + str(X))
    print("Predicted Output: \n" + str(nn.feedforward()))

