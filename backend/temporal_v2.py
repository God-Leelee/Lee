"""Time-aware residual TCN; windows retain actual PTS and valid-frame masks."""
import cv2
import numpy as np
import torch
from torch import nn
ARCHITECTURE='pts_velocity_quality_tcn_v2'


def frame_quality(image):
    gray=cv2.cvtColor(np.asarray(image.resize((128,128))).copy(),cv2.COLOR_RGB2GRAY)
    return [float(np.log1p(cv2.Laplacian(gray,cv2.CV_32F).var())/10.),min(image.size)/1024.]


def validate_window(features,mask,timestamps,quality):
    features=np.asarray(features,float);mask=np.asarray(mask,bool);times=np.asarray(timestamps,float);quality=np.asarray(quality,float)
    if features.ndim!=2 or features.shape[1]!=768 or mask.shape!=features.shape[:1] or times.shape!=mask.shape or quality.shape!=(len(mask),2):
        raise ValueError('Invalid temporal feature/mask/PTS/quality dimensions')
    if not np.isfinite(features).all() or not np.isfinite(times).all() or not np.isfinite(quality).all():raise ValueError('Non-finite temporal inputs')
    if mask.sum()<4 or np.any(np.diff(times[mask])<=0):raise ValueError('Not enough distinct, increasing actual frame timestamps')


class TemporalNetV2(nn.Module):
    def __init__(self):
        super().__init__();self.appearance=nn.Linear(768,48);self.velocity=nn.Linear(768,32);self.context=nn.Linear(3,16)
        self.blocks=nn.Sequential(nn.Conv1d(96,96,3,padding=1),nn.GELU(),nn.Conv1d(96,96,3,padding=2,dilation=2),nn.GELU())
        self.classifier=nn.Linear(192,1)

    def forward(self,features,mask,timestamps,quality):
        valid=mask.unsqueeze(-1).float();normalized=torch.nn.functional.normalize(features.float(),dim=-1)
        center=(normalized*valid).sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1.)
        residual=(normalized-center)*valid;dt=torch.zeros_like(timestamps);dt[:,1:]=(timestamps[:,1:]-timestamps[:,:-1]).clamp_min(.001)
        pair_mask=mask.clone();pair_mask[:,0]=False;pair_mask[:,1:]=mask[:,1:]&mask[:,:-1]
        velocity=torch.zeros_like(normalized);velocity[:,1:]=(normalized[:,1:]-normalized[:,:-1])/dt[:,1:,None]
        velocity=velocity.clamp(-10.,10.)*pair_mask[:,:,None]
        context=torch.cat([torch.log1p(dt.clamp_min(0.))[:,:,None],quality],dim=-1)
        x=torch.cat([self.appearance(residual),self.velocity(velocity),self.context(context)],dim=-1)*valid
        x=self.blocks(x.transpose(1,2));m=mask[:,None].float()
        mean=(x*m).sum(-1)/m.sum(-1).clamp_min(1.);maximum=x.masked_fill(m==0,-1e4).max(-1).values
        return self.classifier(torch.cat([mean,maximum],dim=1)).squeeze(-1)


def bag_logit(window_logits):
    """Whole-video label supervises a bag without inventing window ground truth."""
    return torch.logsumexp(window_logits/.5,dim=-1)*.5 - .5*np.log(window_logits.shape[-1])
