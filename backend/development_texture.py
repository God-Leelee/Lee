"""CPU texture descriptors for development experiments, not a deployed detector.

No metadata, filename, OCR text, label, or model checkpoint is used. These
statistics can reflect compression as well as generation and must never be
treated as proof of AI generation on their own.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

TEXTURE_SPEC = 'gray_residual_frequency_dct_uniform_lbp_cpu_v1'


def texture_features(image: Image.Image):
    gray = np.asarray(image.convert('L'), np.float32) / 255.
    height, width = gray.shape
    if max(height, width) > 1024:
        scale = 1024 / max(height, width)
        gray = cv2.resize(gray, (max(16, round(width * scale)), max(16, round(height * scale))), interpolation=cv2.INTER_AREA)
    if min(gray.shape) < 16:
        raise ValueError('Texture descriptor requires a minimum 16-pixel edge')
    names, values = [], []
    def add(name, value):
        names.append(name); values.append(float(value))
    for sigma in [.7, 1.2, 2.]:
        residual = gray - cv2.GaussianBlur(gray, (0, 0), sigma)
        absolute = np.abs(residual)
        mean, std = float(residual.mean()), float(residual.std())
        for key, val in [
            ('mad', np.median(np.abs(residual - np.median(residual)))),
            ('abs_mean', absolute.mean()), ('abs_q75', np.quantile(absolute, .75)),
            ('abs_q95', np.quantile(absolute, .95)), ('std', std),
            ('kurtosis', np.mean(((residual - mean) / max(std, 1e-6)) ** 4))]:
            add(f'gauss_{sigma}_{key}', val)
        if sigma == 1.2:
            mid_residual = residual
    def correlation(a, b):
        a, b = a.ravel(), b.ravel()
        return float(np.mean((a-a.mean())*(b-b.mean())) / max(1e-8, float(a.std()*b.std())))
    for name, a, b in [('h',mid_residual[:,1:],mid_residual[:,:-1]),
                       ('v',mid_residual[1:],mid_residual[:-1]),
                       ('diag',mid_residual[1:,1:],mid_residual[:-1,:-1]),
                       ('antidiag',mid_residual[1:,:-1],mid_residual[:-1,1:])]:
        add('residual_corr_' + name, correlation(a,b))
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    for q in [.5,.9,.99]: add(f'lap_abs_q{q}', np.quantile(lap,q))
    gx, gy = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3), cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    energy = gx*gx+gy*gy
    add('gradient_mean_energy', energy.mean())
    add('gradient_q95_energy', np.quantile(energy,.95))
    add('gradient_anisotropy', np.abs(np.mean(gx*gx)-np.mean(gy*gy))/max(1e-8,float(energy.mean())))
    fixed = cv2.resize(gray,(256,256),interpolation=cv2.INTER_AREA)
    residual = fixed-cv2.GaussianBlur(fixed,(0,0),1.2)
    window = np.outer(np.hanning(256),np.hanning(256)).astype(np.float32)
    power = np.abs(np.fft.fftshift(np.fft.fft2(residual*window)))**2
    f = np.fft.fftshift(np.fft.fftfreq(256))
    xx,yy=np.meshgrid(f,f);radius=np.sqrt(xx*xx+yy*yy)
    positive=radius>=.02;total=max(1e-20,float(power[positive].sum()))
    bounds=[.02,.05,.1,.15,.25,.35,.5,.708]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        add(f'fft_band_{lo}_{hi}',power[(radius>=lo)&(radius<hi)].sum()/total)
    p=power[positive]
    add('fft_flatness',np.exp(np.mean(np.log(p+1e-20)))/max(1e-20,float(p.mean())))
    add('fft_anisotropy',abs(float(power[(np.abs(xx)>np.abs(yy))&positive].sum()-power[(np.abs(xx)<=np.abs(yy))&positive].sum()))/total)
    dcts=[]
    for y in range(0,256,8):
        for x in range(0,256,8):
            dcts.append(np.abs(cv2.dct(fixed[y:y+8,x:x+8])))
    dcts=np.asarray(dcts);dcts[:,0,0]=0
    ac=max(1e-8,float(dcts.mean()))
    add('dct_ac_mean',ac)
    for y,x in [(0,1),(1,0),(1,1),(0,2),(2,0)]:add(f'dct_ratio_{y}_{x}',dcts[:,y,x].mean()/ac)
    center=fixed[1:-1,1:-1]
    neighbors=[fixed[:-2,:-2],fixed[:-2,1:-1],fixed[:-2,2:],fixed[1:-1,2:],
               fixed[2:,2:],fixed[2:,1:-1],fixed[2:,:-2],fixed[1:-1,:-2]]
    bits=np.stack([n>=center for n in neighbors]);transitions=np.sum(bits!=np.roll(bits,1,axis=0),axis=0)
    ones=bits.sum(axis=0);code=np.where(transitions<=2,ones,9)
    hist=np.bincount(code.ravel(),minlength=10)/code.size
    for i,val in enumerate(hist):add(f'uniform_lbp_{i}',val)
    result=np.asarray(values,np.float64)
    if not np.isfinite(result).all():raise ValueError('Non-finite texture feature')
    return result,names


def layout_statistics(image: Image.Image):
    """Unlabelled visual proxies; not a medical/document semantic classifier."""
    rgb=np.asarray(image.convert('RGB').resize((256,256),Image.Resampling.BILINEAR),np.float32)/255
    gray=rgb.mean(axis=2);saturation=rgb.max(axis=2)-rgb.min(axis=2)
    light_neutral=(gray>.8)&(saturation<.12)
    light_fraction=float(light_neutral.mean())
    return {'light_neutral_pixel_fraction':light_fraction,
            'mean_saturation':float(saturation.mean()),
            'paper_like_visual_proxy':bool(light_fraction>.45 and saturation.mean()<.11)}
