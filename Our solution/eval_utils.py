from keras import backend as K

def soft_jaccard(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    y_true_abs = K.sum(y_true, axis=[1,2,3])
    y_pred_abs = K.sum(y_pred, axis=[1,2,3])
    union = y_true_abs + y_pred_abs - intersection
    jacard_index = (intersection / union)
    
    return jacard_index

def hard_jaccard(y_true, y_pred, pixel_threshold=0.7):

    y_pred = K.greater(y_pred, pixel_threshold)
    y_pred = K.cast(y_pred, 'float32')

    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    y_true_abs = K.sum(y_true, axis=[1,2,3])
    y_pred_abs = K.sum(y_pred, axis=[1,2,3])
    union = y_true_abs + y_pred_abs - intersection
    jacard_index = (intersection / union)
    
    return jacard_index

def soft_jaccard_threshold(y_true, y_pred, threshold=0.65):

    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    y_true_abs = K.sum(y_true, axis=[1,2,3])
    y_pred_abs = K.sum(y_pred, axis=[1,2,3])
    union = y_true_abs + y_pred_abs - intersection
    jaccard_index = (intersection / union)
    
    temp = K.greater(jaccard_index, threshold)
    temp = K.cast(temp, 'float32')

    return (temp*jaccard_index)

def hard_jaccard_threshold(y_true, y_pred, threshold=0.65, pixel_threshold=0.7):

    y_pred = K.greater(y_pred, pixel_threshold)
    y_pred = K.cast(y_pred, 'float32')

    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    y_true_abs = K.sum(y_true, axis=[1,2,3])
    y_pred_abs = K.sum(y_pred, axis=[1,2,3])
    union = y_true_abs + y_pred_abs - intersection
    jaccard_index = (intersection / union)
    
    temp = K.greater(jaccard_index, threshold)
    temp = K.cast(temp, 'float32')

    return (temp*jaccard_index)

