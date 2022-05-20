#---------------------
#
#  This is the code for our ProtoSeg paper:
#  Segmentation Ability Map: Interpret deep features for medical image segmentation
#
#  @Author: Sheng He
#  @Email: heshengxgd@gmail.com
#
#--------------------------

import torch
import torch.nn as nn

class EnhanceNet(nn.Module):
    def __init__(self,nchanels=1,out_classes=1):
        super().__init__()
        
        # for 2D image: self.dims=(2,3)
        # for 3D image: self.dims=(2,3,4)
        self.dims = (2,3)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,xfeat,pred,mask=None):
        #@ xfeat: the deep feature need to be inperpreted
        #@ pred: the initial segmentation results from the last layer of the network
        #@ mask is to maks out the background of the image (without any tissue)
        
        if mask is not None:
            pos_prototype = torch.sum(xfeat*pred*mask,dim=self.dims,keepdim=True)
            num_prototype = torch.sum(pred*mask,dim=self.dims,keepdim=True)
            pos_prototype = pos_prototype / num_prototype
            
            rpred = 1 - pred
            
            neg_prototype = torch.sum(xfeat*rpred*mask,dim=self.dims,keepdim=True)
            num_prototype = torch.sum(rpred*mask,dim=self.dims,keepdim=True)
            neg_prototype = neg_prototype / num_prototype
            
            pfeat = -torch.pow(xfeat-pos_prototype,2).sum(1,keepdim=True)
            nfeat = -torch.pow(xfeat-neg_prototype,2).sum(1,keepdim=True)
            
            disfeat = torch.cat([nfeat,pfeat],dim=1)
            pred = self.softmax(disfeat)

        else:
            pos_prototype = torch.sum(xfeat*pred,dim=self.dims,keepdim=True)
            num_prototype = torch.sum(pred,dim=self.dims,keepdim=True)
            pos_prototype = pos_prototype / num_prototype
            
            rpred = 1 - pred
            
            neg_prototype = torch.sum(xfeat*rpred,dim=self.dims,keepdim=True)
            num_prototype = torch.sum(rpred,dim=self.dims,keepdim=True)
            neg_prototype = neg_prototype / num_prototype
            
            pfeat = -torch.pow(xfeat-pos_prototype,2).sum(1,keepdim=True)
            nfeat = -torch.pow(xfeat-neg_prototype,2).sum(1,keepdim=True)
            
            disfeat = torch.cat([nfeat,pfeat],dim=1)
            pred = self.softmax(disfeat)
            
        return pred
