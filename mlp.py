from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class MultiLayerPerceptron(BaseEstimator, RegressorMixin):

    def __init__(self, K=16, T=10**5, learning_rate=0.01, seed=0, batch_size=1, rozped=0.9, etaZero = 0.05, etaMin = 10**(-3), etaMax = 50, etaInc = 1.2, etaDec = 0.5):
        self.K_ = K # liczba neuronow w warstwie ukrytej
        self.T_ = T # liczba krokow uczenia on-line 
        self.learning_rate_ = learning_rate # wspolczynnik uczenia
        self.seed_ = seed
        self.V_ = None # macierz wag w warstwie ukrytej
        self.W_ = None # wektor (kolumnowy) wag w warstwie wyjsciowej (liniowej)
        self.batch_size_ = batch_size
        #EMA
        self.rozped_ = rozped
        #RPROP
        self.etaZero_ = etaZero
        self.etaMin_ = etaMin
        self.etaMax_ = etaMax
        self.etaInc_ = etaInc
        self.etaDec_ = etaDec

    
    def sigmoid_activation(self, s):
        return 1.0 / (1.0 + np.exp(-s))
    
    def sigmoid_activation_d(self, phi):
        return phi * (1.0 - phi)
    
    def relu_activation(self, s):
        return s * (s > 0.0)
    
    def relu_activation_d(self, phi):
        return 1.0 * (phi > 0.0)
    
    def activation(self, s):
        return self.sigmoid_activation(s)
        #return self.relu_activation(s)

    def activation_d(self, phi):
        return self.sigmoid_activation_d(phi)
        #return self.relu_activation_d(phi)
    

    def fitEMA(self, X, y):
        np.random.seed(self.seed_)
        m, n = X.shape
        self.V_ = (np.random.rand(self.K_, n + 1) * 2.0 - 1.0) * 1e-3
        self.W_ = (np.random.rand(self.K_ + 1, 1) * 2.0 - 1.0) * 1e-3
        X = np.c_[np.ones((m, 1)), X]
        M1 = np.empty(shape=self.W_.shape)
        M2 = np.empty(shape=self.V_.shape)
        M1[0] = 0
        M2[0] = 0
        for _ in range(self.T_):
            indexes = np.random.permutation(m)[:self.batch_size_]
            X_batch = X[indexes]
            s = self.V_.dot(X_batch.T) 
            phi = self.activation(s)
            one_phi = np.r_[np.ones((1, self.batch_size_)), phi]
            y_MLP = (self.W_.T).dot(one_phi)[0]
            err_d = y_MLP - y[indexes]
            dW = np.sum(err_d * one_phi, axis=1)
            dW = np.array([dW]).T
            dV = (err_d * self.W_[1:] * self.activation_d(phi)).dot(X_batch)#to jest ta suma z ∇ ze wzoru
            #m+ = B * m+-1 + (1-B) * ∇+
            M1 = self.rozped_ * M1 + (1 - self.rozped_) * dW
            M2 = self.rozped_ * M2 + (1 - self.rozped_) * dV
            #w++1 = w+ - learn? * m+
            self.W_ = self.W_ - self.learning_rate_ * M1
            self.V_ = self.V_ - self.learning_rate_ * M2 

    def fitRPROP(self, X, y):
        np.random.seed(self.seed_)
        m, n = X.shape
        self.V_ = (np.random.rand(self.K_, n + 1) * 2.0 - 1.0) * 1e-3
        self.W_ = (np.random.rand(self.K_ + 1, 1) * 2.0 - 1.0) * 1e-3
        kpnW = np.empty([len(self.W_),1])
        kpnW.fill(self.etaZero_)
        kpnV = np.empty([len(self.V_), len(self.V_[0])])
        kpnV.fill(self.etaZero_)
        #n+-1
        kpnWminus = np.zeros([len(self.W_),1])
        kpnVminus = np.zeros([len(self.V_), len(self.V_[0])])
        X = np.c_[np.ones((m, 1)), X]
        for _ in range(self.T_):
            indexes = np.random.permutation(m)[:self.batch_size_]
            X_batch = X[indexes]
            s = self.V_.dot(X_batch.T) 
            phi = self.activation(s)
            one_phi = np.r_[np.ones((1, self.batch_size_)), phi]
            y_MLP = (self.W_.T).dot(one_phi)[0]
            err_d = y_MLP - y[indexes]
            dW = np.sum(err_d * one_phi, axis=1)
            dW = np.array([dW]).T
            dV = (err_d * self.W_[1:] * self.activation_d(phi)).dot(X_batch)

            for i in range(len(kpnW)):
                if(kpnWminus[i] * dW[i] > 0):
                    kpnW[i] *= self.etaInc_
                elif(kpnWminus[i] * dW[i] < 0):
                    kpnW[i] *= self.etaDec_

                if(kpnW[i] > self.etaMax_):
                    kpnW[i] = self.etaMax_
                if(kpnW[i] < self.etaMin_):
                    kpnW[i] = self.etaMin_
            for i in range(len(self.V_)):
                for j in range(len(self.V_[0])):
                    if (kpnVminus[i][j] * dV[i][j]) > 0:
                        kpnV[i][j] = kpnV[i][j] * self.etaInc_
                    if kpnV[i][j] > self.etaMax_:
                        kpnV[i][j] = self.etaMax_
                    if (kpnVminus[i][j] * dV[i][j]) < 0:
                        kpnV[i][j] = kpnV[i][j] * self.etaDec_
                    if kpnV[i][j] < self.etaMin_:
                        kpnV[i][j] = self.etaMin_
      

            #przepisz na -1
            kpnWminus = dW
            kpnVminus = dV

            self.W_ = self.W_ - kpnW * (np.sign(dW) + (dW == 0))
            self.V_ = self.V_ - kpnV * (np.sign(dV) + (dV == 0))

    def fitADAM(self, X, y):
        np.random.seed(self.seed_)
        B1 = 0.9
        B2 = 0.999
        eps = 10**(-7)
        m, n = X.shape
        self.V_ = (np.random.rand(self.K_, n + 1) * 2.0 - 1.0) * 1e-3
        self.W_ = (np.random.rand(self.K_ + 1, 1) * 2.0 - 1.0) * 1e-3
        X = np.c_[np.ones((m, 1)), X]

        mplus1 = np.zeros([len(self.W_),1])
        mplus2 = np.zeros([len(self.V_), len(self.V_[0])])

        vplus1 = np.zeros([len(self.W_),1])
        vplus2 = np.zeros([len(self.V_), len(self.V_[0])])
        for i in range(self.T_):
            indexes = np.random.permutation(m)[:self.batch_size_]
            X_batch = X[indexes]
            s = self.V_.dot(X_batch.T) 
            phi = self.activation(s)
            one_phi = np.r_[np.ones((1, self.batch_size_)), phi]
            y_MLP = (self.W_.T).dot(one_phi)[0] # (1 x (K + 1)).dot((K + 1) x self.batch_size_)[0] -> shape: (self.batch_size_,)
            err_d = y_MLP - y[indexes]
            dW = np.sum(err_d * one_phi, axis=1)
            dW = np.array([dW]).T
            dV = (err_d * self.W_[1:] * self.activation_d(phi)).dot(X_batch)  
            #1 wzór
            mplus1 = B1 * mplus1 + (1 - B1) * dW
            mplus2 = B1 * mplus2 + (1 - B1) * dV
            #2 wzór
            vplus1 = B2 * vplus1 + (1 - B2) * dW**2
            vplus2 = B2 * vplus2 + (1 - B2) * dV**2
            #3 wzór 
            mdaszek1 = mplus1/(1 - B1 ** (i + 1))
            mdaszek2 = mplus2/(1 - B1 ** (i + 1))
            #4 wzór
            vdaszek1 = vplus1/(1 - B2 ** (i + 1))
            vdaszek2 = vplus2/(1 - B2 ** (i + 1))

            #w++1 = w+ - learn(ta falka) * mdaszek/(vdaszek^(-1) + eps)
            self.W_ = self.W_ - self.learning_rate_ * mdaszek1/(np.sqrt(vdaszek1) + eps)
            self.V_ = self.V_ - self.learning_rate_ * mdaszek2/(np.sqrt(vdaszek2) + eps)  

    def fit(self, X, y):
        np.random.seed(self.seed_)
        m, n = X.shape
        self.V_ = (np.random.rand(self.K_, n + 1) * 2.0 - 1.0) * 1e-3
        self.W_ = (np.random.rand(self.K_ + 1, 1) * 2.0 - 1.0) * 1e-3
        X = np.c_[np.ones((m, 1)), X]
        for _ in range(self.T_):
            indexes = np.random.permutation(m)[:self.batch_size_]
            X_batch = X[indexes]
            s = self.V_.dot(X_batch.T) 
            phi = self.activation(s)
            one_phi = np.r_[np.ones((1, self.batch_size_)), phi]
            y_MLP = (self.W_.T).dot(one_phi)[0] # (1 x (K + 1)).dot((K + 1) x self.batch_size_)[0] -> shape: (self.batch_size_,)
            err_d = y_MLP - y[indexes]
            dW = np.sum(err_d * one_phi, axis=1)
            dW = np.array([dW]).T
            dV = (err_d * self.W_[1:] * self.activation_d(phi)).dot(X_batch)                                                   
            self.W_ = self.W_ - self.learning_rate_ * dW
            self.V_ = self.V_ - self.learning_rate_ * dV      
     
    def predict(self, X):
        m = X.shape[0]
        X = np.c_[np.ones((m, 1)), X] 
        s = self.V_.dot(X.T) 
        phi = self.activation(s)
        one_phi = np.r_[np.ones((1, m)), phi]
        y_MLP = (self.W_.T).dot(one_phi)[0] # (1 x (K + 1)).dot((K + 1) x m)[0] -> shape: (m,)  
        return y_MLP