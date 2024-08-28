from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import cv2
#import fire
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

import extract_utils as utils

import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys
import warnings

import torch.nn as nn

def calculate_entropy(probabilities):
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def entropy(emd,maxENT=1):

    e = emd.shape[1]; n = emd.shape[0]; hist=[]
    for v in range(0,e):
        hist.append(np.histogram(emd[:,v].ravel(), bins=30)[0])
    
    length = n * e

    entropy = [np.nan_to_num(calculate_entropy((h/length)+0.0001), nan=maxENT) for h in hist]
    
    return np.mean(entropy)

class TELloss(nn.Module):
    def __init__(self):
        super(TELloss, self).__init__()

    def geteig(self,feats,K):

      B,C,H,W= feats.size()
      new_H = int(H // 4)
      new_W = int(W // 4)
      feats = F.interpolate(feats, size=(new_H, new_W), mode='bilinear', align_corners=False)
      feats = feats.reshape(B,C,new_H*new_W).permute(0,2,1)
      feats = F.normalize(feats, p=2, dim=-1)

      # Initialize a list to store results
      eigenvalues_list = []
      eigenvectors_list = []
      for b in range(B):
          W_feat = feats[b]

          # Feature affinities
          W_feat = (W_feat @ W_feat.T)
          W_feat = (W_feat * (W_feat > 0))

          W_feat = W_feat / W_feat.max()  # NOTE: If features are normalized, this naturally does nothing
          W_feat = W_feat.detach().cpu().numpy()

          D_comb = np.array(utils.get_diagonal(W_feat).todense())  # Check if dense or sparse is faster

          try:
              eigenvalues, eigenvectors = eigsh(D_comb - W_feat, k=K, sigma=0, which='LM', M=D_comb)
          except:
              eigenvalues, eigenvectors = eigsh(D_comb - W_feat, k=K, which='SM', M=D_comb)

          eigenvalues_list.append(torch.from_numpy(eigenvalues).cuda())
  
      
      # Convert list of eigenvalues to a tensor
      eigenvalues_tensor = torch.stack(eigenvalues_list)

      return eigenvalues_tensor

    
    def forward(self, feats , K):

      lan1 = self.geteig(feats[:,:1],K)
      lan2 = self.geteig(feats[:,1:2],K)
      lan3 = self.geteig(feats[:,2:3],K)

        # Compute the pairwise MSE losses
      landa1 = F.l1_loss(lan1, lan2).requires_grad_()
      landa2 = F.l1_loss(lan1, lan3).requires_grad_()
      landa3 = F.l1_loss(lan2, lan3).requires_grad_()
        
      return [landa1,landa2,landa3];

