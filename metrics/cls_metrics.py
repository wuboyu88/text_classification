from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class EvaluateScore:
    def __init__(self):
        self.labels = []
        self.preds = []

    def result(self):
        # 汇总效果
        accuracy = accuracy_score(self.labels, self.preds)
        macro_precision = precision_score(self.labels, self.preds, average='macro', zero_division=0)
        macro_recall = recall_score(self.labels, self.preds, average='macro', zero_division=0)
        macro_f1 = f1_score(self.labels, self.preds, average='macro', zero_division=0)
        total_info = {
            'accuracy': accuracy,
            'macro_precision:': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        }

        # 每一类单独效果
        label_names = sorted(set(self.labels))
        precision = precision_score(self.labels, self.preds, average=None, labels=label_names, zero_division=0)
        recall = recall_score(self.labels, self.preds, average=None, labels=label_names, zero_division=0)
        f1 = f1_score(self.labels, self.preds, average=None, labels=label_names, zero_division=0)
        class_info = dict()
        for i, ele in enumerate(label_names):
            class_info[ele] = {
                'precision:': precision[i],
                'recall': recall[i],
                'f1': f1[i]
            }
        return total_info, class_info

    def update(self, labels, preds):
        """
        :param labels: list
        :param preds: list
        :return:
        """
        self.labels.extend(labels)
        self.preds.extend(preds)
