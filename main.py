# -*- coding: utf-8 -*-
'''
Itaru HIDA
July, 2017

distribution learning for XOR with 3-layer perceptron

python 2.7.13
'''

import numpy as np
import argparse
from function import *
from multilayer_bit_const import *

##### OPTIONS #####
parser = argparse.ArgumentParser(
    prog = "xor_dist",
    usage = "distribution learning for XOR with 3-layer perceptron",
    add_help = True
    )

parser.add_argument("-l", "--learning_rate",
                    help = "learning rate (default: 0.25, right shift by 2 bits)",
                    default = 0.25,
                    type = float
                    )

parser.add_argument("-nh", "--n_hidden",
                    help = "number of neurons in hidden layer (default: 10)",
                    default = 10,
                    type = int
                    )

parser.add_argument("-c", "--coeff",
                    help = "coefficient for initial weight (default: random from -1.0 to 1.0)",
                    default = 3.0,
                    type = float
                    )

parser.add_argument("-i", "--iteration",
                    help = "learning iteration (default: 100)",
                    default = 100,
                    type = int
                    )

parser.add_argument("-v", "--interval",
                    help = "interval of averaging weight (default: 10)",
                    default = 10,
                    type = int
                    )

parser.add_argument("-e", "--edge",
                    help = "number of edge devices (default: 10)",
                    default = 10,
                    type = int
                    )

parser.add_argument("-b", "--bit",
                    help = "flag for constraint of weight bit width",
                    action = 'store_true'
                    )

args = parser.parse_args()

bflag     = args.bit
eta       = args.learning_rate
N_H       = args.n_hidden
coeff     = args.coeff
iteration = args.iteration
interval  = args.interval
num_edge  = args.edge

epoch     = int(iteration/interval)

def main():
    ##### DISPLAY PARAMS #####
    print ''
    print 'learning iteration:             %d' % iteration
    print 'interval of averaging weight:   %d' % interval
    print 'total epoch(iteration/interval: %d' % epoch
    print ''
    print 'learning rate:                  %f' % eta
    print 'initial weight:                 %f to %f' % (-coeff, coeff)
    print ''
    print '# of neurons in hidden layer:   %d' % N_H
    print '# of edge devices:              %d' % num_edge
    print ''

    ##### INPUT & LABEL #####
    # input data
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    # label
    y = np.array([0, 1, 1, 0])

    ##### SET LAYERS & ACTIVATION #####
    layers = np.array([2, N_H, 1]) # struction
    act = np.array(['sigmoid','sigmoid'])

    ##### GENERATE INITIAL WEIGHTS #####
    init_weight = []
    for i in range(2):
        init_weight.append(bit(coeff * np.random.uniform(-1.0, 1.0, (layers[i] + 1, layers[i + 1]))))

    ##### BUILD MLP #####
    edge_id = [0 for i in range(num_edge)]
    for i in range(num_edge):
        edge_id[i] = MultiLayer(layers, coeff)
        edge_id[i].overwrite_weight(np.array(init_weight))

    ##### LEARNING #####
    for j in range(epoch):
        ww0 = np.array(edge_id[0].get_weight())
        w_sum =[]
        for i in range(2):
            w_sum.append(np.zeros(ww0[i].shape))

        ##### LOOP OF EDGE DEVICE #####
        for i in range(num_edge):
            ##### LOOP OF INTERVAL #####
            loss = 0.0
            for k in range(interval):
                loss += edge_id[i].fit(X, y, act, eta)
                if k % 10 == 0:
                    print 'edge_id %3d : epoch %3d : %4d/%4d : loss = %2.13f' %\
                          (i, j, j * interval + k, iteration, loss / 10.0)
            ##### ACCUMULATE WEIGHTS IN ALL DEVICES #####
            w_sum += edge_id[i].get_weight()

        ##### AVERAGE WEIGHTS IN ALL DEVICES #####
        w_sum /= num_edge

        ##### OVERWRITE WEIGHTS IN EACH DEVICES WITH AVERAGED WEIGHTS #####
        for i in range(num_edge):
            edge_id[i].overwrite_weight(w_sum)


    ##### INFERENCE #####
    for i in range(4):
        print y[i],
        print edge_id[0].result(X[i], act)

if __name__ == "__main__":
    main()
