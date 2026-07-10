from sklearn.metrics import precision_recall_curve, auc

def pr_auc_func(y_true, y_pred):
    prec, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, prec)