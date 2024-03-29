# Segmentation Ability Map (SAM) for interpreting segmentation neural networks.

Sheng He, Yanfang Feng, P. Ellen Grant, Yangming Ou "Segmentation ability map: Interpret deep features for medical image segmentation", Medical Image Analysis. [`PDF`](https://www.sciencedirect.com/science/article/pii/S1361841522003541)

# How to use it?

This is an example of feature tensors extracted on any layer.

The feature can be extracted on any layer in your network
If the size of the feature does not match your input size, please resize it to match the target size.

```Python
x = torch.rand(2,64,32,32) 
```

This is the probability map obtained from the output of your network, which is 
a guide for the protoSeg to compute the prototype of target leision or no-leison.

Note: this is not the ground-truth (on test set the ground-truth is not available)
The values of pred_map should be in [0,1] where 1 represents the target lesion.
If you use the softmax on the last layer, convert it to probability map into [0,1] where 1 represents target leision.

```Python
pred_map = torch.rand(2,1,32,32) 
neters = ProtoSeg(ndims='2d')
probability_map = neters(x,pred,mask=None)
```

you will get a binary map (target lesion: 1, others: 0) based on the input features "x"
```Python
binary_map = torch.argmax(probability_map,1) # Note: this is not differentiable.
```
