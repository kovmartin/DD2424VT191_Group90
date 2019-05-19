from metrics import dice, jaccard
from keras.losses import binary_crossentropy
from keras import backend as K

def jaccard_log_loss(y_true, y_pred):
    jaccard_index = jaccard(y_true, y_pred)
    return -K.log(jaccard_index)
    
def jaccard_loss(y_true, y_pred):
    jaccard_index = jaccard(y_true, y_pred)
    return 1 - jaccard_index

def dice_log_loss(y_true, y_pred):
    dice_coef = dice(y_true, y_pred)
    return -K.log(dice_coef)

def dice_loss(y_true, y_pred):
    dice_coef = dice(y_true, y_pred)
    return 1 - dice_coef

def dice_log_bce_loss(y_true, y_pred, param=0.7):
    return ((1-param) * K.mean(binary_crossentropy(y_true, y_pred), axis=[1,2]) + param * dice_log_loss(y_true, y_pred))

def jaccard_log_bce_loss(y_true, y_pred, param=0.7):
    return ((1-param) * K.mean(binary_crossentropy(y_true, y_pred), axis=[1,2]) + param * jaccard_log_loss(y_true, y_pred))