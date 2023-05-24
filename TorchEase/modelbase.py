import time
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        args[0].total_training_time += runtime
        return result
    return wrapper
class ModelBase:
    """
    A base class for machine learning models.
    """
    def evaluate_result(self, verbose=True):
        """
        Computes and displays evaluation metrics such as accuracy, precision, recall, f1-score, and confusion matrix.
        """
        import warnings
        # Disable all warnings
        warnings.filterwarnings("ignore")
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report
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
        weighted_precision = np.average(PPV, weights=Support)
        weighted_recall = np.average(TPR, weights=Support)
        avg_ACC = sum(TP) / (sum(TP) + sum(FP))
        data = {"Accuarcy (Micro average f1-score)": avg_ACC, "Precision": PPV, "Recall": TPR, "F1 Score": F1_Score, "weighted f1-score": weighted_f1_score, "Support": Support}
        pd.DataFrame(data)

        #print(f"Support: {Support}")
        #print(f"Accuarcy (Micro average f1-score): {np.around(avg_ACC, decimals=4)}")
        #print(f"weighted f1-score {np.around(weighted_f1_score, decimals=4)}")
        #print(f"weighted precision {np.around(weighted_precision, decimals=4)}")
        #print(f"weighted recall {np.around(weighted_recall, decimals=4)}")
        #print(f"class Accuarcy: {np.around(ACC, decimals=4)}")
        #print(f"class Precision: {np.around(PPV, decimals=4)}")
        #print(f"class Recall: {np.around(TPR, decimals=4)}")
        #print(f"class F1_Score: {np.around(F1_Score, decimals=4)}")
        #print(f"Total: {sum(Support)}, Total Wrong: {total_wrong}, Total Right: {sum(np.diag(cm))}, Total: {total_wrong + sum(np.diag(cm))}")

        #self.accuarcy = np.around(avg_ACC, decimals=4)
        #self.weighted_f1_score = np.around(weighted_f1_score, decimals=4)
        #self.class_accuarcy = np.around(ACC, decimals=4)
        #self.class_precision = np.around(PPV, decimals=4)
        #self.class_recall = np.around(TPR, decimals=4)
        #self.class_f1_score = np.around(F1_Score, decimals=4)
        self.confussion_martix = cm
        self.total_wrong = total_wrong
        self.total_right = sum(np.diag(cm))
        #self.support = Support
        #self.total = total_wrong + sum(np.diag(cm))

        self.metrics = classification_report(target, prediction, digits=4, output_dict=True)


        # from sklearn.metrics import classification_report, roc_curve
        if verbose:
            print(cm)
            print(f"Total Wrong: {total_wrong}")
            if self.total_training_time != 0:
                self.print_formated_duration()
            print(classification_report(target, prediction, digits=4))

    def save_result(self, path_to_save="", CM_fname="ConfussionMatrix.png"):
        """
        Plots the confusion matrix using seaborn heatmap.
        """
        import os
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt
        df_cm = pd.DataFrame(self.cm)
        sns.heatmap(df_cm, annot=True)
        plt.savefig(os.path.join(path_to_save, CM_fname))
        plt.close()


    def print_formated_duration(self):
        # Überprüfen, ob die Dauer weniger als eine Minute beträgt
        if  self.total_training_time < 60:
            print(f"Training took: {self.total_training_time:.2f} Seconds")

        # Überprüfen, ob die Dauer weniger als eine Stunde beträgt
        elif  self.total_training_time < 3600:
            minutes = int(self.total_training_time // 60)
            seconds = int(self.total_training_time % 60)
            print(f"Training took: {minutes:02d}:{seconds:02d} minutes")

        # Überprüfen, ob die Dauer weniger als ein Tag beträgt
        elif self.total_training_time < 86400:
            hours = int(self.total_training_time // 3600)
            minutes = int((self.total_training_time % 3600) // 60)
            seconds = int((self.total_training_time % 3600) % 60)
            print(f"Training took: {hours:02d}:{minutes:02d}:{seconds:02d} hours")

        # Dauer beträgt ein oder mehr Tage
        else:
            days = int(self.total_training_time // 86400)
            hours = int((self.total_training_time % 86400) // 3600)
            minutes = int(((self.total_training_time % 86400) % 3600) // 60)
            seconds = int(((self.total_training_time % 86400) % 3600) % 60)
            print(f"Training took: {days} Days, {hours:02d}:{minutes:02d}:{seconds:02d}")








