import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings


class GaussianProcessRegression:
    
    def __init__(self, cov_function):
        self.cov_function = cov_function
    
    
    def fit(self, T, X):
        ''' "Обучение" модели регрессии.
                T --- np.array, размерность (n, d): моменты времени, 
                      в которые проведены измерения
                X --- np.array, размерность n: полученные значения процесса
        '''
        self.T = T
        self.Cinv = [[self.cov_function(ti - tj) for tj in T] for ti in T]
        # для избежания сингулярной матрицы
        for i in range(len(self.Cinv)):
            self.Cinv[i][i] += 1e-9
        self.Cinv = np.matrix(self.Cinv).I
        self.X = np.matrix(X).T
        
        return self
        
        
    def predict(self, T):
        ''' Оценка значения процесса. 
                T --- np.array, размерность (n, d): моменты времени, 
                      в которые нужно оценить значения. 
                
            Возвращает:
                values --- np.array, размерность n: предсказанные 
                           значения процесса
                sigma --- np.array, размерность n: соответствующая дисперсия
        '''
        
        R = np.array([np.matrix([self.cov_function(t-ti) for ti in self.T]).T for t in T])
        values = np.array([float(r.T*self.Cinv*self.X) for r in R])
        sigma = np.array([float(self.cov_function(np.zeros_like(T[0])) - r.T*self.Cinv*r) for r in R])
        
        return values, sigma