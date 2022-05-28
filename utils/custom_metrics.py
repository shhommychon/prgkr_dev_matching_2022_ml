import torch


def macro_f1(y_pred, y_true, num_classes=16):
    """implementation of F1 (macro) in PyTorch
    
    original code from https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d
    """
    one_hot_matrix = torch.eye(num_classes)
    y_pred = one_hot_matrix[y_pred]
    y_true = one_hot_matrix[y_true]

    confusion_vector = (y_pred / y_true).T
    f1_list = list()
    for cv in confusion_vector:
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)

        true_positives = torch.sum(cv == 1).item()
        false_positives = torch.sum(cv == float('inf')).item()
        # true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives = torch.sum(cv == 0).item()
        if true_positives + false_positives + false_negatives != 0:
            f1_list.append((2*true_positives)/(2*true_positives+false_positives+false_negatives))

    return sum(f1_list) / len(f1_list)