#---------------------
#
#  This is the code for our ProtoSeg paper:
#  Segmentation Ability Map: Interpret deep features for medical image segmentation, Medical Image Analysis 
#
#  @Author: Sheng He
#  @Email: heshengxgd@gmail.com
#
#--------------------------

import torch
import torch.nn as nn

class ProtoSeg(nn.Module):
    def __init__(self,ndims='2d'):
        super().__init__()
        
        # for 1D: self.dims=(2)
        # for 2D image: self.dims=(2,3)
        # for 3D image: self.dims=(2,3,4)
        if ndims == '1d':
            self.dims = (2)
        elif ndims == '2d':
            self.dims = (2,3)
        elif ndims == '3d':
            self.dims = (2,3,4)
        else:
            raise ValueError('ndims must be in [1d,2d,3d]')
       
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

if __name__ == '__main__':
    # examples to show how to use it
    #----------------------------------------
    # this is an example of feature tensors extracted on any layers
    # The feature can be extracted on any layers in your network
    # If the size of the feature does not match your input size, please resize it 
    
    x = torch.rand(2,64,32,32) 
    
    # This is the probability map obtained from the output of your network, which is 
    # an guide for the protoSeg to compute the prototype of target leision or no-leison
    # Note: this is not the ground-truth (on test set the ground-truth are not available)
    # The values of pred_map should be in [0,1] where 1 represents the target lesion.
    # If you use the softmax on the last layer, convert it to probability map into [0,1] where 1 represents target leision.
    
    pred_map = torch.rand(2,1,32,32) 

    neters = ProtoSeg(nchanels=64,out_classes=1)
    
    probability_map = neters(x,pred_map,mask=None)
    
    # you will get a binary map (target lesion: 1, others: 0) based on the input features "x"
    binary_map = torch.argmax(probability_map,1) # Note: this is not differentiable.
    
    
