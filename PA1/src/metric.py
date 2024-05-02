from sklearn.metrics import f1_score
from torchmetrics import Metric
import torch
import numpy as np
np.set_printoptions(threshold=np.inf)

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('average_f1score', default=torch.tensor(0),dist_reduce_fx = 'sum')
                       
    def update(self, preds, target):
        # [Trial 1] : F1Score from Scikit-Learn Library 
        # self.f1_score = f1_score(np.array(target.cpu()),np.array(preds.cpu()),average='macro')
        
        self.num_clss = preds.shape[-1]
        preds = torch.argmax(preds,axis=1)
        assert preds.shape == target.shape, "preds and target must have the same shape"

        # Confusion Matrix 
        cm = torch.zeros(self.num_clss,self.num_clss)
        
        for pred, gt in zip(preds,target):
            cm[int(pred.item()),int(gt.item())] += 1
        
        # per-class F1-score
        per_class_f1score = torch.zeros(self.num_clss)
        for clss in range(self.num_clss):
            # [Trial 2] : Implement custom F1Score from Scratch
            # tp = cm[clss,clss]
            # fp = sum(cm[clss,:]) - tp 
            # fn = sum(cm[:,clss]) - tp
            # if tp+fp ==0 or tp+fn ==0:continue
            # precision = tp / ( tp + fp ) 
            # recall = tp / ( tp + fn )

            # [Trial 3] : Implement custom F1Score from Scratch
            precision = cm[clss,clss] / sum(cm[clss,:])
            recall = cm[clss,clss] / sum(cm[:,clss])
            
            f1_score = 2 * precision * recall / (precision + recall)
            if np.isnan(f1_score):continue
        
            per_class_f1score[clss] += f1_score

        self.average_f1score = torch.mean(per_class_f1score)
        
    def compute(self):
        return self.average_f1score
        

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds,axis=1)
        
        # [TODO] check if preds and target have equal shape
        assert preds.shape == target.shape, "preds and target must have the same shape"

        # [TODO] Cound the number of correct prediction
        corr = torch.sum(preds == target).item()
        
        # Accumulate to self.correct
        self.correct += corr
        
        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
