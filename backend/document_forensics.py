"""Dense, versioned CF patch features for localized document edits.

Pixel labels/boxes are never inference inputs. Dense coverage includes every
working pixel; mapping records the analysis resize and ignores padding.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms

FEATURE_SPEC='cf224_dense_patch384_texture12_float32_side768_stride192_v1'
DIMENSION=396
MAX_SIDE=768
STRIDE=192


def working_image(image):
    factor=min(1.,MAX_SIDE/max(image.size))
    size=(max(1,round(image.width*factor)),max(1,round(image.height*factor)))
    return (image if size==image.size else image.resize(size,Image.Resampling.LANCZOS)),factor


def starts(length):
    last=max(0,length-224)
    return sorted(set(list(range(0,last+1,STRIDE))+[last]))


def tile_plan(image):
    working,factor=working_image(image)
    padded=ImageOps.expand(working,(0,0,max(0,224-working.width),max(0,224-working.height)),fill=(127,127,127))
    positions=[(x,y) for y in starts(working.height) for x in starts(working.width)]
    return working,padded,positions,factor


def patch_average(array):
    if array.shape[:2]!=(224,224):raise ValueError('Patch-grid dimensions differ')
    tail=array.shape[2:]
    return array.reshape(14,16,14,16,*tail).mean(axis=(1,3)).reshape(196,*tail)


def texture_features(image):
    rgb=np.asarray(image,dtype=np.float32)/255.
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    lap=cv2.Laplacian(gray,cv2.CV_32F,ksize=1)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    edge=np.hypot(gx,gy)
    mean=patch_average(rgb);std=np.sqrt(np.maximum(0.,patch_average(rgb**2)-mean**2))
    scalars=[]
    for values in (gray,np.abs(lap),edge):
        m=patch_average(values);s=np.sqrt(np.maximum(0.,patch_average(values**2)-m**2))
        scalars.extend([m,s])
    result=np.column_stack([mean,std,*scalars]).astype(np.float32)
    if result.shape!=(196,12) or not np.isfinite(result).all():raise ValueError('Invalid texture patch features')
    return result


@torch.inference_mode()
def dense_features(image,backbone,device,batch_size=8):
    working,padded,positions,factor=tile_plan(image)
    normalize=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    all_features=[];valid=[]
    for start in range(0,len(positions),batch_size):
        batch_positions=positions[start:start+batch_size]
        crops=[padded.crop((x,y,x+224,y+224)) for x,y in batch_positions]
        tensor=torch.stack([normalize(c) for c in crops]).to(device)
        tokens=backbone.forward_features(tensor).float()
        if tokens.shape!=(len(crops),197,384):raise ValueError('CF dense token layout differs')
        vectors=tokens[:,1:].cpu().numpy()
        for crop,(x,y),vector in zip(crops,batch_positions,vectors):
            all_features.append(np.column_stack([vector,texture_features(crop)]))
            bitmap=np.zeros((224,224),np.float32)
            bitmap[:max(0,min(224,working.height-y)),:max(0,min(224,working.width-x))]=1.
            valid.append(patch_average(bitmap))
    return np.asarray(all_features,np.float32),np.asarray(valid,np.float32),positions,working.size,factor


def mask_grid(mask,positions,size):
    working=mask.resize(size,Image.Resampling.NEAREST)
    padded=ImageOps.expand(working,(0,0,max(0,224-size[0]),max(0,224-size[1])),fill=0)
    return np.asarray([patch_average(np.asarray(padded.crop((x,y,x+224,y+224)),dtype=np.float32)/255.) for x,y in positions])


def reconstruct(values,positions,size,original_size):
    if np.asarray(values).shape!=(len(positions),196):raise ValueError('Dense score grid differs')
    total=np.zeros((max(224,size[1]),max(224,size[0])),np.float32)
    count=np.zeros_like(total)
    for row,(x,y) in zip(values,positions):
        enlarged=np.repeat(np.repeat(np.asarray(row).reshape(14,14),16,0),16,1)
        total[y:y+224,x:x+224]+=enlarged;count[y:y+224,x:x+224]+=1
    if np.any(count[:size[1],:size[0]]==0):raise ValueError('Dense tiling left unobserved working pixels')
    mapped=(total/np.maximum(count,1))[:size[1],:size[0]]
    return np.asarray(Image.fromarray(mapped).resize(original_size,Image.Resampling.BILINEAR),np.float32)


def validate(spec,backbone_sha):
    if spec.get('feature_spec')!=FEATURE_SPEC or spec.get('backbone_sha256')!=backbone_sha or spec.get('test_used') is not False:
        raise ValueError('Document patch source/layout/fitting boundary differs')
    for key in ('scaler_mean','scaler_scale','coefficients'):
        array=np.asarray(spec[key],float)
        if array.shape!=(DIMENSION,) or not np.isfinite(array).all():raise ValueError('Invalid document numeric head')
    if np.any(np.asarray(spec['scaler_scale'])<=0) or not np.isfinite(spec['intercept']):raise ValueError('Invalid document normalization')


def logits(spec,features):
    values=np.asarray(features,float)
    if values.shape[-1]!=DIMENSION or not np.isfinite(values).all():raise ValueError('Invalid document patch inputs')
    result=((values-spec['scaler_mean'])/spec['scaler_scale'])@spec['coefficients']+spec['intercept']
    if not np.isfinite(result).all():raise ValueError('Invalid document patch outputs')
    return result
