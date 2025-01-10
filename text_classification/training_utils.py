from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import evaluate

metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def get_class_weights(df):
    compute_class_weight("balanced", classes=sorted(df['labels'].unique().tolist()), y=df['labels'].tolist())
