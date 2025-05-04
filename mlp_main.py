import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mlp import MultiLayerPerceptron
import time


te = [10**4, 10**5, 10**6, 2*10**4, 2*10**5, 2*10**6]
baczsajz = [100, 10, 1, 100, 10, 1]
def fake_data(m):
    np.random.seed(0)
    X = np.random.rand(m, 2) * np.pi
    y = np.cos(X[:, 0] * X[:, 1]) * np.cos(2 * X[:, 0]) + np.random.randn(m) * 0.1
    return X, y

if __name__ == '__main__':
    print("START")
    X, y = fake_data(1000)
    for j in range(4):
        for i in range(len(te)):
            f = open('wyniki.txt', 'a')
            nn = MultiLayerPerceptron(K=128, T=te[i], learning_rate=0.01, batch_size=baczsajz[i])
            print("FIT...")
            t1 = time.time()
            if j == 0:
                nazwaTXT = "BACKPROPAGATION"
                nn.fit(X, y)
            elif j == 1:
                nazwaTXT = "EMA"
                nn.fitEMA(X, y)
            elif j == 2:
                nazwaTXT = "RPROP"
                nn.fitRPROP(X, y)
            elif j == 3:
                nazwaTXT = "ADAM"
                nn.fitADAM(X, y)
            t2 = time.time()
            czas = t2 - t1
            print(f"FIT DONE [TIME: {czas} s]")
            y_pred = nn.predict(X)
            mse = np.mean(0.5 * (y_pred - y)**2) # mean squared error
            print(f"TRAIN MSE: {mse}")

            lines = [nazwaTXT, "czas: "+str(czas)+"s", "mse: "+str(mse)]
            for line in lines:
                f.write(line)
                f.write('\n')
            f.close()
            #wykresy
            steps = 20       
            X1, X2 = np.meshgrid(np.linspace(0.0, np.pi, steps), np.linspace(0.0, np.pi, steps))
            X12 = np.array([X1.ravel(), X2.ravel()]).T    
            y_ref = np.cos(X12[:, 0] * X12[:, 1]) * np.cos(2 * X12[:, 0])    
            Y_ref = np.reshape(y_ref, (steps, steps))
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1, projection="3d")
            ax.plot_surface(X1, X2, Y_ref, cmap=cm.get_cmap("Spectral"))
            #ax.scatter(X[:, 0], X[:, 1], y)    
            Y_pred = np.reshape(nn.predict(X12), (steps, steps))
            ax = fig.add_subplot(1, 2, 2, projection="3d")
            ax.plot_surface(X1, X2, Y_pred, cmap=cm.get_cmap("Spectral"))   
            nazwaWykres = nazwaTXT+str(i) + '.png'
            plt.savefig(nazwaWykres)
            print("DONE")