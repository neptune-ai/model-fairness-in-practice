from collections import OrderedDict
from IPython.display import HTML, Image
from functools import partial

from aif360.datasets import BinaryLabelDataset
import neptune
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def make_dataset(data, outcome, protected_columns, 
                 privileged_groups, unprivileged_groups, 
                 favorable_label, unfavorable_label):
    df = data.copy()
    df['outcome'] = data[outcome].values

    dataset = BinaryLabelDataset(df=df, label_names=['outcome'], protected_attribute_names=protected_columns,
                                 favorable_label=favorable_label, unfavorable_label=unfavorable_label,
                                 unprivileged_protected_attributes=unprivileged_groups)
    return dataset


def display_results(results_path):
    aif_metric, roc_auc = joblib.load(results_path)
    
    fig = plot_confusion_matrix_by_group(aif_metric, figsize=(12,4))
    plt.tight_layout()
    fig.savefig('../results/temp_img.png')
    plt.close()

    metrics_dict = OrderedDict(
        roc_auc_score=roc_auc,
        
        true_positive_rate_difference=aif_metric.true_positive_rate_difference(),
        false_positive_rate_difference=aif_metric.false_positive_rate_difference(),
        false_omission_rate_difference=aif_metric.false_omission_rate_difference(),
        false_discovery_rate_difference=aif_metric.false_discovery_rate_difference(),
        error_rate_difference=aif_metric.error_rate_difference(),

        false_positive_rate_ratio=aif_metric.false_positive_rate_ratio(),
        false_negative_rate_ratio=aif_metric.false_negative_rate_ratio(),
        false_omission_rate_ratio=aif_metric.false_omission_rate_ratio(),
        false_discovery_rate_ratio=aif_metric.false_discovery_rate_ratio(),
        error_rate_ratio=aif_metric.error_rate_ratio(),

        average_odds_difference=aif_metric.average_odds_difference(),

        disparate_impact=aif_metric.disparate_impact(),
        statistical_parity_difference=aif_metric.statistical_parity_difference(),
        equal_opportunity_difference=aif_metric.equal_opportunity_difference(),
        theil_index=aif_metric.theil_index(),
        between_group_theil_index=aif_metric.between_group_theil_index(),
        between_all_groups_theil_index=aif_metric.between_all_groups_theil_index(),
        coefficient_of_variation=aif_metric.coefficient_of_variation(),
        between_group_coefficient_of_variation=aif_metric.between_group_coefficient_of_variation(),
        between_all_groups_coefficient_of_variation=aif_metric.between_all_groups_coefficient_of_variation(),

        generalized_entropy_index=aif_metric.generalized_entropy_index(),
        between_group_generalized_entropy_index=aif_metric.between_group_generalized_entropy_index(),
        between_all_groups_generalized_entropy_index=aif_metric.between_all_groups_generalized_entropy_index())
    
    metrics = pd.DataFrame()
    metrics['metric_names']=metrics_dict.keys()
    metrics['scores'] = metrics_dict.values()
    
    
    html_table = "<table style='width:100%; border:0px'>{content}</table>"
    html_row = "<tr style='border:0px'>{content}</tr>"
    html_cell_left = "<td style='width:30%;vertical-align:top;border:0px'>{content}</td>"
    html_cell_right = "<td style='width:100%;vertical-align:center;border:100px'>{content}</td>"
        
    cell_left = html_cell_left.format(content=metrics.to_html())
    cell_right = html_cell_right.format(content='<img src="../results/temp_img.png" style="height:300px">')

    row = [cell_left, cell_right]
    
    display(HTML(html_table.format(content="".join(row))))

        
def plot_confusion_matrix_by_group(aif_metric, figsize=None):
    if not figsize:
        figsize=(18,4)

    cmap = plt.get_cmap('Blues')
    fig, axs = plt.subplots(1,3, figsize=figsize)

    axs[0].set_title('all')
    cm = _format_aif360_to_sklearn(aif_metric.binary_confusion_matrix(privileged=None))
    sns.heatmap(cm, cmap=cmap, annot=True, fmt='g', ax=axs[0])
    axs[0].set_xlabel('predicted values')
    axs[0].set_ylabel('actual values')

    axs[1].set_title('privileged')
    cm = _format_aif360_to_sklearn(aif_metric.binary_confusion_matrix(privileged=True))
    sns.heatmap(cm, cmap=cmap, annot=True, fmt='g', ax=axs[1])
    axs[1].set_xlabel('predicted values')
    axs[1].set_ylabel('actual values')

    axs[2].set_title('unprivileged')
    cm = _format_aif360_to_sklearn(aif_metric.binary_confusion_matrix(privileged=False))
    sns.heatmap(cm, cmap=cmap, annot=True, fmt='g', ax=axs[2])
    axs[2].set_xlabel('predicted values')
    axs[2].set_ylabel('actual values')
    return fig

def plot_performance_by_group(aif_metric, metric_name, ax=None):
    performance_metrics = ['TPR', 'TNR', 'FPR', 'FNR', 'PPV', 'NPV', 'FDR', 'FOR', 'ACC']
    
    func_dict = {'selection_rate':lambda x: aif_metric.selection_rate(privileged=x),
                 'precision':lambda x: aif_metric.precision(privileged=x),
                 'recall':lambda x: aif_metric.recall(privileged=x),
                 'sensitivity':lambda x: aif_metric.sensitivity(privileged=x),
                 'specificity':lambda x: aif_metric.specificity(privileged=x),
                 'power':lambda x: aif_metric.power(privileged=x),
                 'error_rate':lambda x: aif_metric.error_rate(privileged=x),
                }
    
    if not ax:
        fig, ax = plt.subplots()
    
    if metric_name in performance_metrics:
        metric_func = lambda x: aif_metric.performance_measures(privileged=x)[metric_name]  
    elif metric_name in func_dict.keys():
        metric_func = func_dict[metric_name]
    else:
        raise NotImplementedError

    df = pd.DataFrame()
    df['Group'] = ['all','priveleged','unpriveleged'] 
    df[metric_name] = [metric_func(group) for group in [None, True, False]]

    sns.barplot(x='Group',y=metric_name, data=df, ax=ax)
    ax.set_title('{} by group'.format(metric_name))
    ax.set_xlabel(None)
    
    _add_annotations(ax)
    
def _add_annotations(ax):
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center',
                    xytext = (0, -10), textcoords = 'offset points')

def _format_aif360_to_sklearn(aif360_mat):
    return np.array([[aif360_mat['TN'],aif360_mat['FP']],
                     [aif360_mat['FN'],aif360_mat['TP']]])