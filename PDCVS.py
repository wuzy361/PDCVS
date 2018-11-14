#! -*- coding: utf-8 -*-
from __future__ import print_function
from keras import backend as K
from keras.engine.topology import Layer
import heapq
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import *
import tensorflow as tf


class My_Reshape1(Layer):

    def __init__(self, target_shape, **kwargs):
        super(My_Reshape1, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def compute_output_shape(self, input_shape):
            return ( self.target_shape)

    def call(self, inputs):
        return K.reshape(inputs,  self.target_shape)


class My_Reshape2(Layer):

    def __init__(self, target_shape,batch_num, **kwargs):
        super(My_Reshape2, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)
        self.batch_num = batch_num

    def compute_output_shape(self, input_shape):
            return (self.batch_num,45, input_shape[2:])

    def call(self, inputs):
        return K.reshape(inputs, (self.batch_num,45,) + self.target_shape[2:])

    def get_config(self):
        config = {'target_shape': self.target_shape}
        base_config = super(Reshape, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




class Position_Embedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., \
                                2 * K.arange(self.size / 2, dtype='float32' \
                                             ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class Attention(Layer):

    def __init__(self, nb_head, size_per_head,time_mask = False,vist_mask = True, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        self.time_mask = time_mask
        self.visit_mask = vist_mask
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def get_mask(self,n):
        e = K.one_hot(K.arange(n), n)
        e = K.expand_dims(K.expand_dims(e, 0), 0)
        e = K.cumsum(e, 3) - e
        e = e * (-1e12)
        return e

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            if self.visit_mask:
                Q_seq, K_seq, V_seq, Q_len, V_len = x
            else:
                Q_seq, K_seq, V_seq ,Q_len, V_len= x
                Q_len, V_len = None, None
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # 计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        if self.time_mask:
            A = A + self.get_mask(K.shape(A)[-1])

        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)
        # 输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)



from keras.preprocessing import sequence
import pickle
import numpy as np
max_features = 20000
maxlen = 80
batch_size = 32
num_class = 945
#num_class = 875
head_num = 8
patient_vist_mlen = 45
code_mlen = 40

emb_num = 64
epochs_num = 100
file_name = "samples"
# train_file = file_name + ".train"
# valid_file = file_name + ".valid"
# test_file = file_name + ".test"
# options = locals().copy()

TIME_STEPS = 45
SINGLE_ATTENTION_VECTOR = False

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul



def load_data(seqFile, labelFile):
    '''

    :param seqFile:序列文件
    :param labelFile: label文件，这里跟序列文件是一样的
    :return: 返回这两个文件根据序列长度从小到大排列后的顺序
    '''
    train_set_x = pickle.load(open(seqFile+'.train', 'rb'))
    valid_set_x = pickle.load(open(seqFile+'.valid', 'rb'))
    test_set_x = pickle.load(open(seqFile+'.test', 'rb'))
    train_set_y = pickle.load(open(labelFile+'.train', 'rb'))
    valid_set_y = pickle.load(open(labelFile+'.valid', 'rb'))
    test_set_y = pickle.load(open(labelFile+'.test', 'rb'))


    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]


    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]

    train_set = (train_set_x, train_set_y,)
    valid_set = (valid_set_x, valid_set_y,)
    test_set = (test_set_x, test_set_y, )

    for  i ,x in enumerate(train_set[0]):
        train_set[0][i] = train_set[0][i][:-1]
    for i,x in enumerate(train_set[1]):
        train_set[1][i] = train_set[1][i][1:]

    for  i ,x in enumerate(valid_set[0]):
        valid_set[0][i] = valid_set[0][i][:-1]
    for i,x in enumerate(valid_set[1]):
        valid_set[1][i] = valid_set[1][i][1:]

    for  i ,x in enumerate(test_set[0]):
        test_set[0][i] = test_set[0][i][:-1]
    for i,x in enumerate(test_set[1]):
        test_set[1][i] = test_set[1][i][1:]


    return train_set, valid_set, test_set

def padMatrixWithoutTime(seqs, labels):
    lengths = np.array([len(seq) for seq in seqs]) - 1
    n_samples = len(seqs)
    maxlen = np.max(lengths)
    inputDimSize = 942
    numClass = inputDimSize
    x = np.zeros((maxlen, n_samples, inputDimSize))
    y = np.zeros((maxlen, n_samples, numClass))
    mask = np.zeros((maxlen, n_samples))
    for idx, (seq,label) in enumerate(zip(seqs,labels)):
        for xvec, subseq in zip(x[:,idx,:], seq[:-1]):
            xvec[subseq] = 1.
        for yvec, subseq in zip(y[:,idx,:], label[1:]):
            yvec[subseq] = 1.
        mask[:lengths[idx], idx] = 1.

    lengths = np.array(lengths)

    return x, y, mask, lengths

#找到visit中最多code次数
'''
test = [[[1,2,5],[2,3]],[[9,11],[11]]]
'''
def get_pad_mask(train_list):
    length = []
    for patient in train_list:
        length.append(len(patient))
    return length

def get_code_len(train_list):
    length = []
    for i,patient in enumerate(train_list):
        for visit in patient:
            length.append(len(visit))
        for visit_len in range(patient_vist_mlen -  get_pad_mask(train_list)[i]):
            length.append(0)
    return np.array(length)

def pad_list(train_list, visit_padding='post',code_padding = 'post'):
    new_test = []
    for patient in train_list:
        patient_vist_pad = patient_vist_mlen - len(patient)
        if visit_padding == 'pre':
            for x in range(patient_vist_pad):
                patient = [[0]] + patient
            new_test.append(patient)
        if visit_padding == 'post':
            for x in range(patient_vist_pad):
                patient =  patient + [[0]]
            new_test.append(patient)

    train_list = []
    for patient in new_test:
        code_list = []
        for code in patient:
            code_pad = code_mlen - len(code)
            if code_padding == 'pre':
                for x in range(code_pad):
                    code = [0] + code
                code_list.append(code)
            if code_padding == 'post':
                for x in range(code_pad):
                    code =  code + [0]
                code_list.append(code)
        train_list.append(code_list)

    return np.array(train_list)



def get_multi_hot_label (pd_seq,num_class):
    patient_num = pd_seq.shape[0]
    visit_num = pd_seq.shape[1]
    code_multi = num_class
    multi_hot_label = np.zeros((patient_num,visit_num,code_multi))
    for patient_index in range(patient_num):
        for visit_index in range(visit_num):
            for code in pd_seq[patient_index][visit_index]:
                multi_hot_label[patient_index][visit_index][code] = 1.0
                multi_hot_label[patient_index][visit_index][0] = 0.0
    return multi_hot_label
#
# train_X = sequence.pad_sequences(train_X, maxlen=mlen)
# train_Y = sequence.pad_sequences(train_Y, maxlen=mlen)
#


def to_list_Y(y_pred,mask,k=30,mode= 'pre'):
    list_vist = []
    # list_patient = []
    for i in range(y_pred.shape[0]):
        list_code = []
        if mode == 'post':
            for j in range(y_pred.shape[1])[-mask[i]:]:
                code = heapq.nlargest(k, range(len(y_pred[i][j])), y_pred[i][j].take)
                list_code.append(code)
        if mode == 'pre':
            for j in range(y_pred.shape[1])[:mask[i]]:
                code = heapq.nlargest(k, range(len(y_pred[i][j])), y_pred[i][j].take)
                list_code.append(code)

        list_vist.append(list_code)

    return list_vist
# test1 = [[[1,2,5],[2,3]],[[9,11],[11]],[[11]]]
test = [[[1,2,5],[2,3],[4,5,6],[3,2,7]],[[9,11],[11],[3,7]],[[11]],[[12]],[[12]],[[12]],[[12]],[[11,2],[3,2]]]

def more_than_1 (y_pred):
    new_y_pred = []
    for patient in y_pred:
        if len(patient)>1:
            new_y_pred.append(patient)
    return new_y_pred

def more_than_2(y_pred):
    new_y_pred = []
    for patient in y_pred:
        if len(patient)>2:
            new_y_pred.append(patient)
    return new_y_pred




def recall(y_true, y_pred ):
    w_cnt = 0
    for x in y_true:
        for y in x:
            for z in y:
                w_cnt+=1
    cnt = 0
    for vist_true,vist_pred in zip(y_true,y_pred):
        for code_true,code_pred in zip(vist_true,vist_pred):
            for x in code_true:
                if x in code_pred:
                    cnt +=1
    return  cnt/w_cnt

def recall_last(y_true, y_pred ):
    w_cnt = 0
    cnt = 0
    for x in y_true:
        # for y in x:
        y = x[-1]
        for z in y:
            w_cnt+=1
    for vist_true,vist_pred in zip(y_true,y_pred):
        # for code_true,code_pred in zip(vist_true,vist_pred):
        code_true, code_pred = vist_true[-1], vist_pred[-1]
        for x in code_true:
            if x in code_pred:
                cnt +=1
    return  cnt/w_cnt

# def to_multi_hot (pd_seq,num_class):
#     multi_hot_seq = np.zeros([len(pd_seq),num_class])
#     for i,x in enumerate(pd_seq):
#         temp = to_categorical(x,num_classes=num_class)
#         temp = np.sum(temp,axis=0)
#         multi_hot_seq[i] = temp
#         #消除pad多产生的零的影响
#         multi_hot_seq[i][0] -= (code_mlen-len(seqs[1][-1][i]))
#     multi_hot_seq[multi_hot_seq>0] =  1
#     return multi_hot_seq

'''
max_features = 20000
maxlen = 80
batch_size = 32
num_class = 945

patient_vist_mlen = 45
code_mlen = 40

emb_num = 32

file_name = "samples"
'''

train_set, valid_set, test_set = load_data(file_name,file_name)

# seqs = labels = train_set
train_X = train_set[0]
train_Y = train_set[1]

row_train_X = train_X

valid_X = valid_set[0]
valid_Y = valid_set[1]

row_valid_X = valid_X

test_X = test_set[0]
test_Y = test_set[1]

row_test_X = test_X

row_train_Y = train_Y
row_valid_Y = valid_Y
row_test_Y =test_Y


mask_train = get_pad_mask(train_X)
mask_valid = get_pad_mask(valid_X)
mask_test = get_pad_mask(test_X)

train_X = pad_list(train_X)
valid_X = pad_list(valid_X)
test_X = pad_list(test_X)
print("padding X done!,train_X.shape:",train_X.shape)

train_Y = pad_list(train_Y)
valid_Y = pad_list(valid_Y)
test_Y = pad_list(test_Y)
print("padding Y done!")

train_Y = get_multi_hot_label(train_Y,num_class)
valid_Y = get_multi_hot_label(valid_Y,num_class)
test_Y = get_multi_hot_label(test_Y,num_class)

print("multi_hot_label done!,train_Y.shape",train_Y.shape)

print('Build seq2seq model...')



def return_model_1():
    # pool + gru
    S_inputs = Input(shape=(patient_vist_mlen, code_mlen), dtype='int32')
    reshape1 = Reshape((patient_vist_mlen, code_mlen), input_shape=(patient_vist_mlen, code_mlen,))(S_inputs)
    emb_code = Embedding(num_class, emb_num, name="emb", )(reshape1)
    emb_code = My_Reshape1((-1, 40, 64))(emb_code)
    emb_vist = GlobalAveragePooling1D()(emb_code)
    # emb_vist = Dense(emb_num, activation='relu')(emb_vist)
    emb_vist = My_Reshape1((-1, 45, 64))(emb_vist)
#    emb = Position_Embedding()(emb_vist)
    emb_patient = GRU(128, name="gru", return_sequences=True, )(emb_vist)
    predict = TimeDistributed(Dense(num_class, activation='sigmoid', name="time_dense2"))(emb_patient)

    model = Model(inputs=[S_inputs], outputs=predict)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                                   schedule_decay=0.004), )
    return model

def return_model_2():
    #mlp+gru
    S_inputs = Input(shape=(patient_vist_mlen, code_mlen), dtype='int32')
    reshape1 = Reshape((patient_vist_mlen, code_mlen), input_shape=(patient_vist_mlen, code_mlen,))(S_inputs)
    emb = Embedding(num_class, emb_num, name="emb", )(reshape1)
    reshape2 = Reshape((patient_vist_mlen, code_mlen * emb_num))(emb)
    emb_vist = Dense(emb_num, activation='relu')(reshape2)
    emb_patient = GRU(128, dropout=0.2, recurrent_dropout=0.2, name="gru", return_sequences=True, )(emb_vist)
    predict = TimeDistributed(Dense(num_class, activation='sigmoid', name="time_dense2"))(emb_patient)

    model = Model(inputs=[S_inputs], outputs=predict)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                                   schedule_decay=0.004), )
    return model

def return_model_3():
    # att + gru
    S_inputs = Input(shape=(patient_vist_mlen, code_mlen), dtype='int32')
    #    Len_inputs = Input(shape=(patient_vist_mlen, 1), dtype='int32')
    reshape1 = Reshape((patient_vist_mlen, code_mlen), input_shape=(patient_vist_mlen, code_mlen,))(S_inputs)
    emb_code = Embedding(num_class, emb_num, name="emb", )(reshape1)
    emb_code = My_Reshape1((-1, 40, 64))(emb_code)
    #   len_seq = My_Reshape1((-1, 1))(Len_inputs)
    attention_layer = Attention(head_num, emb_num // head_num)
    emb_code_contex = attention_layer([emb_code, emb_code, emb_code])
    emb_vist = GlobalAveragePooling1D()(emb_code_contex)
    emb_vist = My_Reshape1((-1, 45, 64))(emb_vist)
#    emb = Position_Embedding()(emb_vist)
    emb_patient = GRU(128,name="gru", return_sequences=True, )(emb_vist)
    predict = TimeDistributed(Dense(num_class, activation='sigmoid', name="time_dense2"))(emb_patient)

    model = Model(inputs=[S_inputs], outputs=predict)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                                   schedule_decay=0.004), )
    return model




def return_model_4():
    # pool + att
    S_inputs = Input(shape=(patient_vist_mlen, code_mlen), dtype='int32')
    reshape1 = Reshape((patient_vist_mlen, code_mlen), input_shape=(patient_vist_mlen, code_mlen,))(S_inputs)
    emb_code = Embedding(num_class, emb_num, name="emb", )(reshape1)
    emb_code = My_Reshape1((-1, 40, 64))(emb_code)
    emb_vist = GlobalAveragePooling1D()(emb_code)
    emb_vist = My_Reshape1((-1, 45, 64))(emb_vist)
#    emb = Position_Embedding()(emb_vist)
    emb_vist = Position_Embedding()(emb_vist)
    emb_vist_context = Attention(8, 16, time_mask=True)([emb_vist, emb_vist, emb_vist])
    emb_vist_context = Dropout(0.5)(emb_vist_context)
    emb_patient = TimeDistributed(Dense(128, activation='relu', name="time_dense1"))(emb_vist_context)
    predict = TimeDistributed(Dense(num_class, activation='sigmoid', name="time_dense2"))(emb_patient)

    model = Model(inputs=[S_inputs], outputs=predict)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                                   schedule_decay=0.004), )
    return model

def return_model_4b():
    # without time_mask
    S_inputs = Input(shape=(patient_vist_mlen, code_mlen), dtype='int32')
    reshape1 = Reshape((patient_vist_mlen, code_mlen), input_shape=(patient_vist_mlen, code_mlen,))(S_inputs)
    emb_code = Embedding(num_class, emb_num, name="emb", )(reshape1)
    emb_code = My_Reshape1((-1, 40, 64))(emb_code)
    emb_vist = GlobalAveragePooling1D()(emb_code)
    emb_vist = My_Reshape1((-1, 45, 64))(emb_vist)
#    emb = Position_Embedding()(emb_vist)
    emb_vist = Position_Embedding()(emb_vist)
    emb_vist_context = Attention(8, 16, time_mask=False)([emb_vist, emb_vist, emb_vist])
    emb_vist_context = Dropout(0.5)(emb_vist_context)
    emb_patient = TimeDistributed(Dense(128, activation='relu', name="time_dense1"))(emb_vist_context)
    predict = TimeDistributed(Dense(num_class, activation='sigmoid', name="time_dense2"))(emb_patient)

    model = Model(inputs=[S_inputs], outputs=predict)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                                   schedule_decay=0.004), )
    return model




def return_model_5():
    # mlp + att
    S_inputs = Input(shape=(patient_vist_mlen, code_mlen), dtype='int32')
    reshape1 = Reshape((patient_vist_mlen, code_mlen), input_shape=(patient_vist_mlen, code_mlen,))(S_inputs)
    emb = Embedding(num_class, emb_num, name="emb", )(reshape1)
    reshape2 = Reshape((patient_vist_mlen, code_mlen * emb_num))(emb)
    emb_vist = Dense(emb_num, activation='relu')(reshape2)
    emb_vist = Position_Embedding()(emb_vist)
    emb_vist_context = Attention(8, 16, time_mask=True)([emb_vist, emb_vist, emb_vist])
    emb_vist_context = Dropout(0.5)(emb_vist_context)
    emb_patient = TimeDistributed(Dense(128, activation='relu', name="time_dense1"))(emb_vist_context)
    predict = TimeDistributed(Dense(num_class, activation='sigmoid', name="time_dense2"))(emb_patient)

    model = Model(inputs=[S_inputs], outputs=predict)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                                   schedule_decay=0.004), )
    return model




def return_model_6():
    # att pool +att
    S_inputs = Input(shape=(patient_vist_mlen, code_mlen), dtype='int32')
#    Len_inputs = Input(shape=(patient_vist_mlen, 1), dtype='int32')
    reshape1 = Reshape((patient_vist_mlen, code_mlen), input_shape=(patient_vist_mlen, code_mlen,))(S_inputs)
    emb_code = Embedding(num_class, emb_num, name="emb", )(reshape1)
    emb_code = My_Reshape1((-1, 40, emb_num))(emb_code)
 #  len_seq = My_Reshape1((-1, 1))(Len_inputs)
    attention_layer = Attention(head_num, emb_num//head_num)
    emb_code_contex = attention_layer([emb_code, emb_code, emb_code])
    emb_vist = GlobalAveragePooling1D()(emb_code_contex)
    emb_vist = My_Reshape1((-1, 45, emb_num))(emb_vist)
    emb_vist = Position_Embedding()(emb_vist)
    emb_vist_context = Attention(8, 16, time_mask=True)([emb_vist, emb_vist, emb_vist])
    emb_vist_context = Dropout(0.5)(emb_vist_context)
    emb_patient = TimeDistributed(Dense(128, activation='relu', name="time_dense1"))(emb_vist_context)
    predict = TimeDistributed(Dense(num_class, activation='sigmoid', name="time_dense2"))(emb_patient)

    model = Model(inputs=[S_inputs], outputs=predict)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                                   schedule_decay=0.004) )
    return model



import sys
 # print "脚本名：", sys.argv[0]
# for i in range(1, len(sys.argv)):
#     print "参数", i, sys.argv[i]
import time
if __name__ == '__main__':
    t1 = time.time()
    op = sys.argv[1]
    model = locals()['return_model_'+str(op)]()
    # model = return_model_1()

    model.summary()
    best_valid_recall_10 = 0
    best_valid_recall_20 = 0
    best_valid_recall_30 = 0

    test_recall_10 = 0
    test_recall_20 = 0
    test_recall_30 = 0

    test_recall_10_mt1 = 0
    test_recall_20_mt1 = 0
    test_recall_30_mt1 = 0

    best_epoch = 0

    len_seq_train = get_code_len(row_train_X).reshape((-1,patient_vist_mlen,1))
    len_seq_valid = get_code_len(row_valid_Y).reshape((-1,patient_vist_mlen,1))
    len_seq_test = get_code_len(row_test_X).reshape((-1,patient_vist_mlen,1))

    t2 = time.time()
    print("data_processing time used:",t2-t1)
    record = [] #record the results of every epoch
    for alg_num in range(0, 3):
        record.append([])
    for alg in record:
        for x in range(0, 3):
            alg.append([])
    for x in range(epochs_num):
        print("The whole epoch:"+str(x)+"/"+str(epochs_num))
        model.fit(train_X, train_Y,
                  batch_size=batch_size,
                  epochs=1,
                  )
        predict_valid = model.predict(valid_X)
        v_predict_list_10 = to_list_Y(predict_valid,mask_valid,10)
        v_recall_10 = recall(row_valid_Y,v_predict_list_10)

        v_predict_list_20 = to_list_Y(predict_valid,mask_valid,20)
        v_recall_20 = recall(row_valid_Y,v_predict_list_20)

        v_predict_list_30 = to_list_Y(predict_valid,mask_valid,30)
        v_recall_30 = recall(row_valid_Y,v_predict_list_30)


        predict_test = model.predict(test_X)
        t_predict_list_10 = to_list_Y(predict_test, mask_test, 10)
        t_recall_10 = recall(row_test_Y, t_predict_list_10)

        t_predict_list_20 = to_list_Y(predict_test, mask_test, 20)
        t_recall_20 = recall(row_test_Y, t_predict_list_20)

        t_predict_list_30 = to_list_Y(predict_test, mask_test, 30)
        t_recall_30 = recall(row_test_Y, t_predict_list_30)


        row_test_Y_mt1 = more_than_1(row_test_Y)
        t_predict_list_10_mt1 = more_than_1(t_predict_list_10)
        t_predict_list_20_mt1 = more_than_1(t_predict_list_20)
        t_predict_list_30_mt1 = more_than_1(t_predict_list_30)

        t_recall_10_mt1 = recall(row_test_Y_mt1, t_predict_list_10_mt1)
        t_recall_20_mt1 = recall(row_test_Y_mt1, t_predict_list_20_mt1)
        t_recall_30_mt1 = recall(row_test_Y_mt1, t_predict_list_30_mt1)


        if best_valid_recall_30 < v_recall_30:
            best_valid_recall_30 = v_recall_30
            best_valid_recall_20 = v_recall_20
            best_valid_recall_10 = v_recall_10

            test_recall_10 = t_recall_10
            test_recall_20 = t_recall_20
            test_recall_30 = t_recall_30

            test_recall_10_mt1 = t_recall_10_mt1
            test_recall_20_mt1 = t_recall_20_mt1
            test_recall_30_mt1 = t_recall_30_mt1

            best_epoch = x

           # model.save_weights("model{}.h5".format("@"+str(x)+"epoch"))

        record[0][0].append(v_recall_10)
        record[0][1].append(v_recall_20)
        record[0][2].append(v_recall_30)
        record[1][0].append(t_recall_10)
        record[1][1].append(t_recall_20)
        record[1][2].append(t_recall_30)
        record[2][0].append(t_recall_10_mt1)
        record[2][1].append(t_recall_20_mt1)
        record[2][2].append(t_recall_30_mt1)
        print("on valid set : recall@10:{0:.3f} , recall@20:{1:.3f} , recall@30:{2:.3f}".format(v_recall_10,v_recall_20,v_recall_30))
        print("on test set : recall@10:{0:.3f} , recall@20:{1:.3f} , recall@30:{2:.3f} ".format(t_recall_10,t_recall_20,t_recall_30))
        print("on test set(more than 1) : recall@10:{0:.3f} , recall@20:{1:.3f} , recall@30:{2:.3f} \n".format(t_recall_10_mt1,t_recall_20_mt1,t_recall_30_mt1))

    t3 = time.time()
    print("whole time used:",t3-t1)
    print("\nthe best  :")
    print("best epoch :",best_epoch)
    print("on valid set : recall@10:{0:.3f} , recall@20:{1:.3f} , recall@30:{2:.3f}".format(best_valid_recall_10, best_valid_recall_20, best_valid_recall_30))
    print("on test set : recall@10:{0:.3f} , recall@20:{1:.3f} , recall@30:{2:.3f}".format(test_recall_10, test_recall_20, test_recall_30))
    print("on test set(more than 1) : recall@10:{0:.3f} , recall@20:{1:.3f} , recall@30:{2:.3f} \n".format(test_recall_10_mt1,
                                                                                               test_recall_20_mt1,
                                                                                               test_recall_30_mt1))
    pickle.dump(record,open('record'+str(op),'wb'))

