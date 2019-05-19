from keras import backend as K

'''
TP - true positive
TN - true negative 
FP - false positive
FN - false negative
P - # of real positive
N - # of real negative
intersection = TP
'''

def dice(y_true, y_pred):
    '''
    dice = 2*TP / (2*TP + FP + FN)
    '''
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    y_true_abs = K.sum(y_true, axis=[1,2,3])
    y_pred_abs = K.sum(y_pred, axis=[1,2,3])
    return (2. * intersection / (y_true_abs + y_pred_abs))

def jaccard(y_true, y_pred):
    '''
    jaccard index = Intersection over union (IoU) = intersection / union
    '''
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    y_true_abs = K.sum(y_true, axis=[1,2,3])
    y_pred_abs = K.sum(y_pred, axis=[1,2,3])
    union = y_true_abs + y_pred_abs - intersection
    return (intersection / union)

def precision(y_true, y_pred):
    '''
    precision = TP / (TP + FP)
    '''
    TP = K.sum(y_true * y_pred, axis=[1,2,3])
    TP_FP = K.sum(y_pred, axis=[1,2,3])
    return TP / TP_FP

def sensitivity(y_true, y_pred):
    '''
    sensitivity = TP / P
    '''
    TP = K.sum(y_true * y_pred, axis=[1,2,3])
    P = K.sum(y_true, axis=[1,2,3])
    return (TP / P)

def specificity(y_true, y_pred):
    '''
    specificity = TN / N
    '''
    TN = K.sum(K.abs((1. - y_true) * (1. - y_pred)), axis=[1, 2, 3])
    N = K.sum(K.abs(1. - y_true), axis=[1, 2, 3])
    return (TN / N)

