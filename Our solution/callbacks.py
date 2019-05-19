import os
import warnings

import numpy as np

import pandas as pd
from collections import defaultdict, OrderedDict

from keras.callbacks import Callback

import matplotlib.pyplot as plt
from skimage.measure import find_contours
from skimage.util.montage import montage2d
montage_rgb = lambda x: np.stack([montage2d(x[:, :, :, i]) for i in range(x.shape[3])], -1)

class ProgeressPlot(Callback):
    def __init__(self, test_gen, monitor='val_jaccard', mode='auto', save_dir='.'):        
        self.probe_img, self.probe_mask = test_gen
        self.save_dir=save_dir
        self.monitor=monitor
        
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('Progress_plot mode %s is unknown, '
                          'fallback to auto mode.' % (mode))
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
        
    def on_train_begin(self, logs={}):
        if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        return
  
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is not None:
            if self.monitor_op(current, self.best):
                predict=self.model.predict(self.probe_img)
                    
                fig, (ax1, ax3, ax4) = plt.subplots(1, 3, figsize = (40, 10))
                ax1.imshow(montage_rgb(self.probe_img)*0.5+0.5, cmap='gray')
                ax1.set_title('Original images')
                ax3.imshow(montage2d(predict[:, :, :, 0]), cmap = 'gray')
                ax3.set_title('Predictions');
                ax4.imshow(montage2d(self.probe_mask[:, :, :, 0]), cmap = 'gray')
                ax4.set_title('Original Masks');            
                plt.savefig(os.path.join(
                    self.save_dir, 'sample1_%d_epoch.png' % (epoch+1)),dpi=300)
                plt.close('all')
                
                fig, ax1= plt.subplots(1, 1, figsize = (10, 10))
                ax1.imshow(montage_rgb(self.probe_img)*0.5+0.5)
                ax1.imshow(montage2d(predict[:, :, :, 0]), cmap='gray', alpha=0.3)
                ax1.set_title('Image VS prediction');

                contours_masks = find_contours(montage2d(self.probe_mask[:, :, :, 0]), level=0.99)
                for contour in contours_masks:
                    ax1.plot(contour[:, 1], contour[:, 0], "--r", linewidth=0.5)

                contours_predictions = find_contours(montage2d(predict[:, :, :, 0]), level=0.9)
                for contour in contours_predictions:
                    ax1.plot(contour[:, 1], contour[:, 0], "--b", linewidth=0.5)
                
                plt.savefig(os.path.join(
                    self.save_dir, 'sample2_%d_epoch.png' % (epoch+1)),dpi=300)
                plt.close('all')
                self.best = current
                print("Saved predicted images.")        
        return

class ProgressSave(Callback):
    def __init__(self, file_mode='a', save_dir='.', filename='unknown.csv'):
        self.file_mode=file_mode
        self.save_dir=save_dir
        self.filename = filename
        self.filepath = os.path.join(self.save_dir, self.filename)
        
    def on_train_begin(self, logs={}):
        if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
       
        if os.path.exists(self.filepath) and self.file_mode == 'w':
            os.remove(self.filepath)
            self.file_df=pd.DataFrame()
            f = open(self.filepath,"w+")
            print("ProgressSave: [INFO] Creating file...")
            f.close()
        
        elif os.path.exists(self.filepath) and self.file_mode == 'a':    
            self.file_df=pd.read_csv(self.filepath)
            print("ProgressSave: [INFO] File found.")
            
        if not os.path.exists(self.filepath):
            self.file_df=pd.DataFrame()
            f = open(self.filepath,"w+")
            print("ProgressSave: [INFO] Creating file...")
            f.close()
        return
 
    def on_epoch_end(self, epoch, logs={}):
        self.logs=logs
        if 'lr' not in logs.keys():
            logs['lr'] = K.get_value(self.model.optimizer.lr)
        
        row_dict = OrderedDict({'epoch': [epoch+1]})
        row_dict.update((key, logs[key]) for key in logs.keys())
        temp_df = pd.DataFrame.from_dict(row_dict)
        self.file_df=pd.concat([self.file_df,temp_df])
        self.file_df.to_csv(self.filepath,index=False)
        return