# plot frequency with importance
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np

def ts_importance(ax, importance, timeseries, importance_2 = None, axis = None, normalize = True, width = 1.0, cmap = 'Blues', alpha = 0.7):
    if axis is None:
        axis = np.arange(len(timeseries))
    my_cmap_mean = cm.get_cmap(cmap)
    importance = importance.squeeze()
    if normalize:
        if importance_2 is not None:
            # normalize with respect to the maximum of both
            imp = np.concatenate((importance, importance_2))
            scale_min = np.nanmin(imp)       
            scale_max = np.nanmax(imp)      
        else:
            scale_min = np.nanmin(importance)
            scale_max = np.nanmax(importance)
        importance = (importance - scale_min)/(scale_max - scale_min)
        
    length = len(importance)
    # min max scale importance between 0 and 1
    #scale_col_mean = np.expand_dims((importance - np.nanmin(importance))/(np.nanmax(importance) - np.nanmin(importance)), 1)
    #color_weight_mean = np.ones((length, 3))
    #color_weight_mean = np.concatenate((color_weight_mean, scale_col_mean), 1)
    plot_col_mean = my_cmap_mean(importance)#*color_weight_mean
    if importance_2 is not None:
        my_cmap_mean = cm.get_cmap('Reds')
        importance_2 = importance_2.squeeze()
        if normalize:
            importance_2 = (importance_2 - scale_min)/(scale_max - scale_min)
        length = len(importance_2)
        # min max scale importance between 0 and 1
        #scale_col_mean = np.expand_dims((importance_2 - scale_min)/(scale_max - scale_min), 1)
        #color_weight_mean = np.ones((length, 3))
        #color_weight_mean = np.concatenate((color_weight_mean, scale_col_mean), 1)
        plot_col_mean_2 = my_cmap_mean(importance_2)#*color_weight_mean
    ax.bar(axis, np.ones_like(importance)*(np.max(timeseries)-np.min(timeseries))*2, bottom=np.min(timeseries), width=width,
            color=plot_col_mean, alpha = alpha)
    ax.plot(axis, timeseries, color='black', alpha = 0.5)
    if importance_2 is not None:
        ax.bar(axis, np.ones_like(importance)*(np.max(timeseries)-np.min(timeseries))*2, bottom=np.min(timeseries), width=width,
            color=plot_col_mean_2)
    
    
    ax.set_ylim(timeseries.min(), timeseries.max()+(timeseries.max()-timeseries.min())*0.10)