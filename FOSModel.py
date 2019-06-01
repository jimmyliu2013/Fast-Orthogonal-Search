import numpy as np
import math
import itertools
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time

"""
 usage: 1.model = FOSModel(max_delay_in_input, max_delay_in_output, max_order, max_m, mse_reduction_threshold)
        2.model.fit(x, y)
        3.model.predict(x, y) # x is longer than y
"""


class FOSModel:
    
    DEBUG = False
    
    def __init__(self, max_delay_in_input, max_delay_in_output, max_order, max_m, mse_reduction_threshold, mse_threshold=0):
        
        if max_delay_in_input < 0 or max_delay_in_output < 1 or max_order < 1 or max_m < 1 or mse_reduction_threshold < 0 or mse_threshold < 0:
            raise ValueError("illegal parameter!")
        
        self.MAX_M = max_m
        self.MSE_THRESHOLD = mse_threshold
        self.MSE_REDUCTION_THRESHOLD = mse_reduction_threshold
        self.MAX_ORDER = max_order
        self.final_P = []
        self.final_P_in_polynomial_form = []
        self.final_a_m = []
        self.L = max_delay_in_input  # delay in x
        self.K = max_delay_in_output  # delay in y
        self.N_0 = max(max_delay_in_input, max_delay_in_output)
        self.candidate_list = []
        # generate candidate from order 1 to max_order
        for order in range(1, max_order + 1):
            self.candidate_list.extend(self.generate_candidate_list(order))
    
    def generate_candidate_list(self, sum_order):
        candidate_func_list = []
        for order in range(0, sum_order + 1):
            order_of_x_term = order
            order_of_y_term = sum_order - order
            comb_of_x = []
            comb_of_y = []
            together = []
            if order_of_x_term > 0:
                comb_of_x = itertools.combinations_with_replacement(range(0, self.L + 1), order_of_x_term)
            if order_of_y_term > 0:
                comb_of_y = itertools.combinations_with_replacement(range(1, self.K + 1), order_of_y_term)
            if order_of_x_term > 0 and order_of_y_term > 0:
                together = list(itertools.product(comb_of_x, comb_of_y))
            else:
                together = list(itertools.zip_longest(comb_of_x, comb_of_y))
            candidate_func_list.extend(together)
        return candidate_func_list
    
    def generate_P_func_from_delays(self, delay_of_x, delay_of_y, x, y):

        def P(n):
            if delay_of_x == None and delay_of_y == None:
                raise ValueError('order for P cannot be zero')
            p = 1
            if delay_of_x != None:
                for l in delay_of_x:
                    p = p * (x[n - l] if n >= l else 0)
            if delay_of_y != None:
                for k in delay_of_y:
                    p = p * (y[n - k] if n >= k else 0)
            return p

        return P
                
    def calculate_value_of_P(self, P):
        P_value = []
        for i in range(0, len(self.X)):
            if P == 1:
                P_value.append(1)
            else:
                P_value.append(P(i))
        return P_value
    
    def calculate_time_average_of_P(self, P_M, start, end):
        summ = 0
        for n in range(start, end):
            summ += P_M(n)
        return summ / (end - start)
    
    def calculate_time_average_of_P_square(self, P_M, start, end):
        summ = 0
        for n in range(start, end):
            summ += P_M(n) ** 2
        return summ / (end - start)
    
    def calculate_time_average_of_P_M_times_P_r(self, P_M, P_r, start, end):
        summ = 0
        for n in range(start, end):
            summ += P_M(n) * P_r(n)
        return summ / (end - start)
    
    def calculate_time_average_of_P_M_times_y(self, P_M, Y, start, end):
        summ = 0
        for n in range(start, end):
            summ += P_M(n) * Y[n]
        return summ / (end - start)
    
    def fit(self, X, Y):
        print("start fitting")
        start_time = time.time() 
        
        if len(Y) != len(X):
            raise ValueError("length of input and output data must be equal")
        
        Y_bar = np.mean(Y[self.N_0:])
        
        self.final_P = []
        self.final_P_in_polynomial_form = []
        self.final_a_m = []
        
        # init global storage
        P = [None] * (self.MAX_M + 1)
        P[0] = 1
        
        # another form of P
        P_in_polynomial_form = [None] * (self.MAX_M + 1)
        P_in_polynomial_form[0] = 1
        
        Alpha = [[0 for x in range(self.MAX_M + 1)] for y in range(self.MAX_M + 1)]
        
        D = [[0 for x in range(self.MAX_M + 1)] for y in range(self.MAX_M + 1)]
        D[0][0] = 1
        
        C = [None] * (self.MAX_M + 1)
        C[0] = Y_bar
        
        g = [None] * (self.MAX_M + 1)
        g[0] = Y_bar
        
        M = 1

        while (True):
            if FOSModel.DEBUG:
                print("\n################### choose candidate for M=", M, "###################")
            
            # init storage for current M
            MAX_Q_M = 0
            INDEX_OF_BEST_P_M = -1
            
            for index, candidate in enumerate(self.candidate_list):
                
                # init local storage for current candidate in current M 
                D_M_r = [0] * (M + 1)
                alpha_M_r = [0] * (M + 1)
                
                delay_of_x = candidate[0]
                delay_of_y = candidate[1]
                if FOSModel.DEBUG:
                    print("\n----------------- evaluate the %dth candidate in the list -----------------" % (index))

                P_M = self.generate_P_func_from_delays(delay_of_x, delay_of_y, X, Y)
            
                P_bar = self.calculate_time_average_of_P(P_M, self.N_0, len(X))
            
                P_square_bar = self.calculate_time_average_of_P_square(P_M, self.N_0, len(X))
                if FOSModel.DEBUG:
                    print("delay_of_x: ", delay_of_x)
                    print("delay_of_y: ", delay_of_y)
                    print("P_bar = ", P_bar)
                    print("P_square_bar = ", P_square_bar)
                
                for r in range(0, M + 1):
                    if r == 0:
                        D_M_r[r] = P_bar
                        alpha_M_r[r] = P_bar
                        if FOSModel.DEBUG:
                            print("D[%d][%d] = %10.5f" % (M, r, D_M_r[r]))
                            print("alpha[%d][%d] = %10.5f" % (M, r, alpha_M_r[r]))
                    elif r > 0 and r < M:
                        D_M_r[r] = self.calculate_time_average_of_P_M_times_P_r(P_M, P[r], self.N_0, len(X)) - np.sum([Alpha[r][i] * D_M_r[i] for i in range(0, r)])
                        alpha_M_r[r] = D_M_r[r] / D[r][r]
                        if FOSModel.DEBUG:
                            print("D[%d][%d] = %10.5f" % (M, r, D_M_r[r]))
                            print("alpha[%d][%d] = %10.5f" % (M, r, alpha_M_r[r]))
                    elif r == M:
                        D_M_r[r] = self.calculate_time_average_of_P_square(P_M, self.N_0, len(X)) - np.sum([alpha_M_r[i] * D_M_r[i] for i in range(0, r)])
                        if FOSModel.DEBUG:
                            print("D[%d][%d] = %10.5f" % (M, r, D_M_r[r]))
                    else:
                        raise ValueError("wrong r value!")
                
                C_M = self.calculate_time_average_of_P_M_times_y(P_M, Y, self.N_0, len(X)) - np.sum([alpha_M_r[r] * C[r] for r in range(0, M)])
                if FOSModel.DEBUG:
                    print("C(%d) for the %dth candidata = %10.5f" % (M, index, C_M))
                
                g_M = C_M / D_M_r[M]
                Q_M = g_M ** 2 * D_M_r[M]
                
                if FOSModel.DEBUG:
                    print("Q(%d) for the %dth candidata = %10.5f" % (M, index, Q_M))
                
                if Q_M > MAX_Q_M:
                    MAX_Q_M = Q_M
                    INDEX_OF_BEST_P_M = index
                    P[M] = P_M
                    P_in_polynomial_form[M] = self.candidate_list[index]
                    C[M] = C_M
                    D[M][M] = D_M_r[M]
                    g[M] = g_M
                    for r in range(0, M):
                        Alpha[M][r] = alpha_M_r[r]
            
            actual_len_of_P = M + 1
            
            if FOSModel.DEBUG:
                print("\n################### finished choosing candidate for M=%d, best Q_M is: %10.5f, corresponding index is: %d" % (M, MAX_Q_M, INDEX_OF_BEST_P_M), "###################")
            
            should_stop_early = (MAX_Q_M <= self.MSE_REDUCTION_THRESHOLD)
            if should_stop_early:
                if FOSModel.DEBUG:
                    print("\nthe last term contributes very little to the model, it will be removed")
                actual_len_of_P = actual_len_of_P - 1  # remove the last term since it contribute very little to the model
            
            MSE = np.mean(Y[self.N_0:] ** 2) - np.sum([ g[m] ** 2 * D[m][m] for m in range(0, actual_len_of_P) ])
            
            # calculate coefficient
            a_m = [0] * (actual_len_of_P)
            v = [0] * (actual_len_of_P)
            for m in range(0, actual_len_of_P):
                v[m] = 1
                for i in range(m + 1, actual_len_of_P):
                    v[i] = -1 * np.sum([ Alpha[i][r] * v[r] for r in range(m, actual_len_of_P - 1) ])
                a_m[m] = np.sum([np.asarray(g[i]) * v[i] for i in range(m, actual_len_of_P)])
            
            if should_stop_early or M >= self.MAX_M or MSE <= self.MSE_THRESHOLD:
                if FOSModel.DEBUG:
                    print("\nstop adding terms.MSE: ", MSE)
                    print("by MSE threshold reduction? : ", should_stop_early)
                    print("by MSE threshold? : ", MSE < self.MSE_THRESHOLD)
                self.final_P = P[:actual_len_of_P]
                self.final_P_in_polynomial_form = P_in_polynomial_form[:actual_len_of_P]
                self.final_a_m = a_m
                print("end fitting, time used: ", time.time() - start_time)
                return (self.final_P_in_polynomial_form, self.final_a_m)
            
            # got to add next term
            self.candidate_list.pop(INDEX_OF_BEST_P_M)
            M += 1
    
    def predict(self, X, Y):
        
        if len(self.final_P_in_polynomial_form) == 0:
            raise Exception("model is empty, fit first!")
         
        if(len(Y) > len(X)):
            raise ValueError("output is longer than input!")
        
        Y_predicted = np.array([])
         
        for n in range(len(Y), len(X)):
            y = 0
            for i in range(0, len(self.final_P_in_polynomial_form)):
                if i == 0:
                    y += self.final_a_m[i] * 1  # constant term
                else:
                    delay_of_x = self.final_P_in_polynomial_form[i][0]
                    delay_of_y = self.final_P_in_polynomial_form[i][1]
                    if delay_of_x == None and delay_of_y == None:
                        raise ValueError('this cannot be a constant term')
                    p = 1
                    if delay_of_x != None:
                        for l in delay_of_x:
                            p = p * (X[n - l] if n >= l else 0)
                    if delay_of_y != None:
                        for k in delay_of_y:
                            p = p * (Y[n - k] if n >= k else 0)
                    y += self.final_a_m[i] * p
            Y_predicted = np.append(Y_predicted, y)
            Y = np.append(Y, y)
        return Y, Y_predicted
    
    def get_printable_function(self):
        f = "y = " + str(self.final_a_m[0])
        for i in range(1, len(self.final_P_in_polynomial_form)):
            delay_of_x = self.final_P_in_polynomial_form[i][0]
            delay_of_y = self.final_P_in_polynomial_form[i][1]
            if delay_of_x == None and delay_of_y == None:
                        raise ValueError('this cannot be a constant term')
            current_term = str(self.final_a_m[i])
            if delay_of_x != None:
                for l in delay_of_x:
                    current_term = current_term + "*x[n-" + str(l) + "]"
            if delay_of_y != None:
                for k in delay_of_y:
                    current_term = current_term + "*y[n-" + str(k) + "]"
            f = f + " + " + current_term
        return f
    
