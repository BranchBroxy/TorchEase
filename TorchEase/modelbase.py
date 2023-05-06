
class ModelBase:
    """
    A base class for machine learning models.
    """
    def evaluate_result(self):
        """
        Computes and displays evaluation metrics such as accuracy, precision, recall, f1-score, and confusion matrix.
        """
        from sklearn.metrics import confusion_matrix
        target = self.evaluation_target
        prediction = self.evaluation_prediction
        self.cm = confusion_matrix(y_true=target, y_pred=prediction)
        import numpy as np
        import pandas as pd
        import copy
        cm = self.cm
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        total_cm = copy.deepcopy(cm)
        np.fill_diagonal(total_cm, 0)
        total_wrong = total_cm.sum()

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)

        F1_Score = 2 * (PPV * TPR) / (PPV + TPR)
        # Overall accuracy
        ACC = TP / (TP + FP + FN)
        unique, counts = np.unique(target, return_counts=True)
        Support = counts
        weighted_f1_score = np.average(F1_Score, weights=Support)
        avg_ACC = sum(TP) / (sum(TP) + sum(FP))
        data = {"Accuarcy (Micro average f1-score)": avg_ACC, "Precision": PPV, "Recall": TPR, "F1 Score": F1_Score, "weighted f1-score": weighted_f1_score, "Support": Support}
        pd.DataFrame(data)
        print(cm)
        #print(f"Support: {Support}")
        #print(f"Accuarcy (Micro average f1-score): {np.around(avg_ACC, decimals=4)}")
        #print(f"weighted f1-score {np.around(weighted_f1_score, decimals=4)}")
        #print(f"class Accuarcy: {np.around(ACC, decimals=4)}")
        #print(f"class Precision: {np.around(PPV, decimals=4)}")
        #print(f"class Recall: {np.around(TPR, decimals=4)}")
        #print(f"class F1_Score: {np.around(F1_Score, decimals=4)}")
        print(f"Total Wrong: {total_wrong}")


        from sklearn.metrics import classification_report, confusion_matrix, roc_curve
        print(classification_report(target, prediction))

    def plot_evaluate_result(self):
        """
        Plots the confusion matrix using seaborn heatmap.
        """
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt
        df_cm = pd.DataFrame(self.cm)
        sns.heatmap(df_cm, annot=True)
        plt.show()



