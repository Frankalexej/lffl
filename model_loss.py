import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats

# class ContrastiveLoss(torch.nn.Module):

#     def __init__(self, margin=2.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, output1, output2, label):
#         euclidean_distance = F.pairwise_distance(output1, output2)
#         pos = (1-label) * torch.pow(euclidean_distance, 2)
#         neg = (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
#         loss_contrastive = torch.mean( pos + neg )
#         return loss_contrastive

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = False)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive


def calculate_mean_ci(dissimilarities, labels, alpha=0.95):
    """
    Calculate mean, upper confidence interval, and lower confidence interval of dissimilarities for each label category.

    Args:
    - dissimilarities (list): Tensor of dissimilarities.
    - labels (list): Tensor of binary labels (0 for "same" and 1 for "different").
    - alpha (float): Confidence level for calculating the confidence interval.

    Returns:
    - dict: Dictionary containing mean, upper confidence interval, and lower confidence interval for each label category.
    """

    # Convert PyTorch tensors to NumPy arrays
    dissimilarities_np = np.array(dissimilarities)
    labels_np = np.array(labels)

    # Calculate mean, upper confidence interval, and lower confidence interval for "same" label
    indices_same = labels_np == 1
    dissimilarities_same = dissimilarities_np[indices_same]

    mean_same = np.mean(dissimilarities_same)
    ci_same = stats.t.interval(confidence=alpha, df=len(dissimilarities_same)-1, loc=mean_same, scale=stats.sem(dissimilarities_same))
    mean_same, ci_upper_same, ci_lower_same = mean_same, ci_same[1], ci_same[0]

    # Calculate mean, upper confidence interval, and lower confidence interval for "different" label
    indices_diff = labels_np == 0
    dissimilarities_diff = dissimilarities_np[indices_diff]

    mean_diff = np.mean(dissimilarities_diff)
    ci_diff = stats.t.interval(confidence=alpha, df=len(dissimilarities_diff)-1, loc=mean_diff, scale=stats.sem(dissimilarities_diff))
    mean_diff, ci_upper_diff, ci_lower_diff = mean_diff, ci_diff[1], ci_diff[0]

    same_res = {'mean': mean_same, 'ci_upper': ci_upper_same, 'ci_lower': ci_lower_same}
    diff_res = {'mean': mean_diff, 'ci_upper': ci_upper_diff, 'ci_lower': ci_lower_diff}

    return same_res, diff_res