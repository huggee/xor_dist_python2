# -*- coding: utf-8 -*-
import numpy as np
from function import *

class MultiLayer:
    def __init__(self, struct, coeff):
        self.layers = struct.size
        self.str = np.array(struct)

        #重み初期値の生成
        weight1 = np.zeros(0,dtype=np.float128)
        self.weight = []
        for var1 in range(self.layers-1):
            weight1 = bit(coeff * np.random.uniform(-1.0, 1.0, (struct[var1]+1, struct[var1+1] )))
            self.weight.append(weight1)

        # 隠れ層
        self.u = np.zeros(self.str[1], dtype = np.float128)
        # self.u.append(np.zeros(self.str[1], dtype = np.float128)

        # 隠れ層の活性の生成
        self.hid = []
        for var1 in range(self.layers-2):
            hid1 = np.zeros(self.str[var1+1],dtype=np.float128)
            self.hid.append(hid1)

        # 修正情報の生成
        self.PHI = []
        for var1 in range(self.layers-1):
            phi1 = np.zeros( struct[var1+1] + 0 ,dtype=np.float128)
            self.PHI.append(phi1)


    def overwrite_weight(self, new_weight):
        self.weight = new_weight

    def fit(self, X, T, act, eta):
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        T = np.array(T)
        loss_acc     = 0

        # 入力データをランダムに1つ取り出す。
        i = np.random.randint(X.shape[0])
        x = np.array(X[i],dtype=np.float128)
        t = T[i]

        #　順伝播計算
        # INPUT to HIDDEN
        w = bit(np.array(self.weight[0]))
        self.u = multiply_inf(x, w)
        self.hid[0] = act_func(self.u, act[0])

        # HIDDEN to OUTPUT
        w = bit(np.array(self.weight[1]))
        self.hid[0] = np.hstack([1, self.hid[0] ])
        self.y = act_func(multiply_inf(self.hid[0], w),act[1])


        # 逆伝播計算 + 重み更新
        # OUTPUT to HIDDEN
        ph = np.array(self.PHI[1])
        w  = np.array(self.weight[1])
        ph_ = np.array(self.PHI[0])

        # 修正情報
        ph = self.y - t
        # ph = bit_16(dif(self.y,act[1]) * ( t - self.y ))
        loss_acc += 0.5 * (np.sum(np.dot(ph, ph)))
        # 重み更新
        for var2 in range(self.hid[0].size):
            w[var2] -= bit_16(eta * ph * self.hid[0][var2])
        # PHI逆伝播
        for val2 in range(self.u.size):
            ph_[val2] = bit_16(dif(self.u[val2], act[0]) * multiply_lea(ph , w[val2]))
        # Over write
        self.PHI[1] = np.array(ph)
        self.weight[1] = np.array(w)
        self.PHI[0] = np.array(ph_)

        # HIDDEN to INPUT
        ph2 = np.array(self.PHI[0]) # PHI算出用
        w = np.array(self.weight[0])

        # 重み算出
        for var2 in range(0, x.size):
            w[var2] -= bit_16(eta * ph2 * x[var2])
        # Over write
        self.weight[0] = np.array(w)

        return loss_acc

    def result(self, l, act):
        x = np.insert(l, 0, 1)
        # INPUT to HIDDEN
        w = bit(np.array(self.weight[0]))
        self.hid[0] = act_func( multiply_inf(x, w), act[0])

        # HIDDEN to OUTPUT
        w = bit(np.array(self.weight[1]))
        self.hid[0] = np.hstack([1, self.hid[0]])
        self.y = act_func(multiply_inf(self.hid[0], w), act[1])

        return step(self.y)


    def get_weight(self):
        return self.weight

        # return np.array(self.weight)
