from __future__ import annotations

from contextlib import nullcontext
import json
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import transforms

from .contracts import ROOT, sha256
from .domain_generation import (MULTIBLOCK_FEATURE_SPEC, block_descriptor, descriptor,
                                global_view, predict as domain_predict, validate_spec)


SCENES = [
    ('vehicle', 'a photograph of a car or damaged vehicle'),
    ('document', 'a scanned document, invoice, receipt or certificate'),
    ('building', 'a photograph of a building or room'),
    ('person', 'a photograph of a person'),
    ('landscape', 'a photograph of outdoor scenery'),
    ('object', 'a photograph of an object or product'),
    ('screen', 'a screenshot or a computer generated graphic'),
]


class TemporalNet(nn.Module):
    """Real trainable TCN. Only a trained, hashed checkpoint may produce a score."""
    def __init__(self, dimensions=768):
        super().__init__()
        self.project = nn.Linear(dimensions, 64)
        self.blocks = nn.Sequential(nn.Conv1d(64, 64, 3, padding=1), nn.GELU(),
                                    nn.Conv1d(64, 64, 3, padding=2, dilation=2), nn.GELU())
        self.classifier = nn.Linear(128, 1)

    def forward(self, feature, mask):
        # Remove time-constant appearance to prevent the temporal head from
        # acting as a disguised mean frame classifier.
        valid = mask.unsqueeze(-1).float()
        center = (feature * valid).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)
        feature = (feature - center) * valid
        x = self.project(feature).transpose(1, 2)
        x = self.blocks(x)
        m = mask[:, None].float()
        mean = (x * m).sum(-1) / m.sum(-1).clamp_min(1)
        maximum = x.masked_fill(m == 0, -1e4).max(-1).values
        return self.classifier(torch.cat([mean, maximum], dim=1)).squeeze(-1)


class Models:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() and config['device'] != 'cpu' else 'cpu'
        torch.set_num_threads(4)
        cv2.setNumThreads(2)
        if self.device == 'cuda':
            # The prototype accepts varying image/video dimensions. Exhaustive
            # per-shape cuDNN profiling adds seconds to the first new shape.
            torch.backends.cudnn.benchmark = False
        self.clip = self.head = self.trufor = self.ocr = self.temporal = self.waverep = self.domain_head = None
        self.image_adapter = None
        self.video_adapter = None
        self.errors, self.info = {}, {}
        self.ready = False

    def _disable_model(self, branch, error):
        self.errors[branch] = f'{type(error).__name__}: {error}'
        attributes = {'generation': ('clip', 'head', 'scene_text', 'preprocess'),
                      'forensic_generation': ('waverep', 'wave_preprocess'),
                      'localization': ('trufor',), 'ocr': ('ocr',),
                      'temporal': ('temporal',), 'domain_generation': ('domain_head',),
                      'standalone_generation': ('image_adapter',), 'video_origin': ('video_adapter',)}
        for attribute in attributes[branch]:
            setattr(self, attribute, None)
        self.info.pop(branch, None)

    def load(self, generation_only=False):
        began = time.perf_counter()
        try:
            import open_clip
            cached=list((ROOT/'models/cache').glob('models--timm--vit_large_patch14_clip_224.openai/snapshots/*/open_clip_model.safetensors'))
            pretrained=str(cached[0]) if len(cached)==1 else 'openai'
            self.clip, _, _ = open_clip.create_model_and_transforms(
                'ViT-L-14', pretrained=pretrained, device=self.device,
                precision='fp16' if self.device == 'cuda' else 'fp32', force_quick_gelu=True,
                cache_dir=str(ROOT / 'models/cache'))
            self.clip.eval()
            self.head = nn.Linear(768, 1).to(self.device).eval()
            weights = torch.load(ROOT / 'models/ufd_fc.pth', map_location='cpu', weights_only=True)
            self.head.load_state_dict(weights)
            # The published UFD validation uses CenterCrop only, without Resize.
            self.preprocess = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize([.48145466, .4578275, .40821073], [.26862954, .26130258, .27577711])])
            tokenizer = open_clip.get_tokenizer('ViT-L-14')
            with torch.inference_mode():
                self.scene_text = self.clip.encode_text(tokenizer([s[1] for s in SCENES]).to(self.device), normalize=True).float()
            self.info['generation'] = {'model': 'UFD / OpenAI CLIP ViT-L/14',
                'head_sha256': sha256(ROOT / 'models/ufd_fc.pth'),
                'encoder_sha256':sha256(cached[0]) if len(cached)==1 else 'see models/reproducibility_manifest.json',
                'source': 'https://github.com/WisconsinAIVision/UniversalFakeDetect',
                'preprocess': 'CenterCrop(224), official CLIP mean/std; no feature normalization before head',
                'precision': 'fp16 encoder, fp32 head' if self.device == 'cuda' else 'fp32'}
        except Exception as exc:
            self._disable_model('generation', exc)
        domain_config = self.config.get('domain_generation', {})
        if domain_config.get('enabled'):
            try:
                checkpoint = ROOT / domain_config['checkpoint']
                if sha256(checkpoint) != domain_config['sha256']:
                    raise ValueError('Domain-head checkpoint differs from the frozen configuration')
                spec = json.loads(checkpoint.read_text(encoding='utf-8'))
                validate_spec(spec, self.info['generation']['encoder_sha256'])
                self.domain_head = spec
                self.info['domain_generation'] = {'model':'CLIP two-view + CPU texture / locally trained AI-origin head'
                    if spec['feature_spec'] == MULTIBLOCK_FEATURE_SPEC else 'CLIP two-view / locally trained AI-origin head',
                    'sha256':sha256(checkpoint),'version':spec['version'],'trained':True,
                    'feature_spec':spec['feature_spec'],'scope':spec['scope'],
                    'threshold':spec['threshold'],'source_manifest_sha256':spec['source_manifest_sha256'],
                    'frozen_split_sha256':spec['frozen_split_sha256'],
                    'partitions':{key:len(ids) for key,ids in spec['sample_ids'].items()},
                    'decision_role':domain_config['decision_role']}
            except Exception as exc:
                self._disable_model('domain_generation',exc)
        if generation_only:
            self.ready = True
            self.info['runtime'] = {'device': self.device, 'gpu': torch.cuda.get_device_name(0) if self.device == 'cuda' else None,
                                    'torch': torch.__version__, 'startup_seconds': round(time.perf_counter() - began, 3)}
            return
        wave_path = ROOT / 'models/weights_dinov2_G4.ckpt'
        if wave_path.exists():
            try:
                import timm
                # Full released checkpoint includes the backbone. There is no
                # need to download a second generic DINO checkpoint.
                self.waverep = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', num_classes=1,
                                                pretrained=False, img_size=504).to(self.device).eval()
                wave = torch.load(wave_path, map_location='cpu', weights_only=False)
                state = {k[6:]:v for k,v in wave['state_dict'].items() if k.startswith('model.')} if 'state_dict' in wave else wave
                self.waverep.load_state_dict(state, strict=True)
                self.wave_preprocess = transforms.Compose([transforms.CenterCrop((504,504)),transforms.ToTensor(),
                    transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
                self.info['forensic_generation'] = {'model':'WaveRep DINOv2-B/14 G4', 'sha256':sha256(wave_path),
                    'source':'https://github.com/grip-unina/WaveRep-SyntheticVideoDetection',
                    'scope':'informational/nonprofit research only; retain original notices',
                    'preprocess':'official CenterCrop(504), ImageNet normalization',
                    'adaptation':'PTS sampled frames; mixed precision; mean native logits for video'}
            except Exception as exc:
                self._disable_model('forensic_generation', exc)
        adapter_config = self.config.get('image_origin_adapter', {})
        if adapter_config.get('enabled'):
            try:
                from .standalone_generation import validate_spec as validate_adapter
                adapter_path = ROOT / adapter_config['checkpoint']
                if sha256(adapter_path) != adapter_config['sha256']:
                    raise ValueError('Public image-adapter checkpoint differs from the frozen configuration')
                adapter = json.loads(adapter_path.read_text(encoding='utf-8'))
                validate_adapter(adapter, self.info['generation']['encoder_sha256'], self.info['forensic_generation']['sha256'])
                self.image_adapter = adapter
                self.info['standalone_generation'] = {'model': 'CLIP / WaveRep multi-view image-origin adapter',
                    'version': adapter['version'], 'sha256': sha256(adapter_path), 'trained': True,
                    'feature_spec': adapter['feature_spec'], 'candidate': adapter['candidate'],
                    'threshold': adapter['threshold'], 'scope': adapter['scope'],
                    'partitions': {key: len(value) for key, value in adapter['sample_ids'].items()},
                    'development_manifest_sha256': adapter['development_manifest_sha256']}
            except Exception as exc:
                self._disable_model('standalone_generation', exc)
        video_config = self.config.get('video_origin_adapter', {})
        if video_config.get('enabled'):
            try:
                from .video_origin import validate_spec as validate_video_adapter
                video_path=ROOT/video_config['checkpoint']
                if sha256(video_path)!=video_config['sha256']:raise ValueError('Video-origin checkpoint differs from frozen configuration')
                video_adapter=json.loads(video_path.read_text(encoding='utf-8'))
                validate_video_adapter(video_adapter,self.info['generation']['encoder_sha256'],self.info['forensic_generation']['sha256'])
                if self.image_adapter is None:raise ValueError('Video-origin requires the validated shared visual descriptor chain')
                self.video_adapter=video_adapter
                self.info['video_origin']={'model':'CLIP / WaveRep two-keyframe video-origin adapter',
                    'version':video_adapter['version'],'sha256':sha256(video_path),'trained':True,
                    'threshold':video_adapter['threshold'],'scope':video_adapter['scope'],
                    'source_groups':{k:len(v) for k,v in video_adapter['source_groups'].items()},
                    'aggregation':video_adapter['aggregation'],'frame_selection':video_adapter['frame_selection']}
            except Exception as exc:
                self._disable_model('video_origin',exc)
        try:
            base = ROOT / 'third_party/TruFor/TruFor_train_test'
            sys.path.insert(0, str(base))
            from lib.config import config as original_config
            from lib.models.cmx.builder_np_conf import EncoderDecoder
            cfg = original_config.clone()
            cfg.defrost()
            cfg.merge_from_file(str(base / 'lib/config/trufor_ph3.yaml'))
            cfg.MODEL.PRETRAINED = ''
            cfg.freeze()
            self.trufor = EncoderDecoder(cfg).to(self.device).eval()
            checkpoint = torch.load(ROOT / 'models/trufor.pth.tar', map_location='cpu', weights_only=False)
            self.trufor.load_state_dict(checkpoint['state_dict'], strict=True)
            self.info['localization'] = {'model': 'TruFor / Noiseprint++ + SegFormer-B2',
                'sha256': sha256(ROOT / 'models/trufor.pth.tar'),
                'source': 'https://github.com/grip-unina/TruFor',
                'scope': 'nonprofit research only; original terms retained',
                'preprocess': 'RGB float / 256.0 as upstream; resize is a recorded prototype adaptation'}
        except Exception as exc:
            self._disable_model('localization', exc)
        try:
            from rapidocr_onnxruntime import RapidOCR
            ocr_threads=self.config.get('ocr_intra_op_threads',2)
            self.ocr = RapidOCR(intra_op_num_threads=ocr_threads, inter_op_num_threads=1)
            self.info['ocr'] = {'model': 'RapidOCR / PP-OCR ONNX', 'source': 'https://github.com/RapidAI/RapidOCR',
                                'intra_op_num_threads':ocr_threads,'inter_op_num_threads':1}
        except Exception as exc:
            self._disable_model('ocr', exc)
        temporal_file = ROOT / self.config.get('temporal_checkpoint', 'models/temporal_tcn.pth')
        if temporal_file.exists():
            try:
                checkpoint = torch.load(temporal_file, map_location='cpu', weights_only=True)
                if checkpoint.get('trained') is not True or not checkpoint.get('training_manifest_sha256'):
                    raise ValueError('Temporal checkpoint has no completed training provenance')
                if checkpoint.get('architecture') not in ['centered_temporal_tcn_v1', 'pts_velocity_quality_tcn_v2']:
                    raise ValueError('Checkpoint architecture differs from the current centered temporal network')
                if checkpoint['architecture']=='pts_velocity_quality_tcn_v2':
                    from .temporal_v2 import TemporalNetV2
                    self.temporal = TemporalNetV2().to(self.device).eval()
                else:
                    self.temporal = TemporalNet().to(self.device).eval()
                expected=self.config.get('temporal_checkpoint_sha256')
                if expected and sha256(temporal_file)!=expected:raise ValueError('Temporal checkpoint hash differs from frozen configuration')
                self.temporal.load_state_dict(checkpoint['state_dict'])
                self.info['temporal'] = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
                self.info['temporal']['sha256'] = sha256(temporal_file)
            except Exception as exc:
                self._disable_model('temporal', exc)
        # Startup work is measured separately. No random input becomes a detection.
        warm = Image.new('RGB', (384, 384), (128, 128, 128))
        self._warmup_models(warm)
        self.ready = True
        self.info['runtime'] = {'device': self.device, 'gpu': torch.cuda.get_device_name(0) if self.device == 'cuda' else None,
            'torch': torch.__version__, 'startup_seconds': round(time.perf_counter() - began, 3)}

    def _warmup_models(self, image):
        jobs = [
            ('generation', self.clip is not None and self.head is not None,
             lambda: self.encode([image])),
            ('localization', self.trufor is not None, lambda: self.localize(image)),
            ('forensic_generation', self.waverep is not None, lambda: self.wave_scores([image])),
            ('ocr', self.ocr is not None, lambda: self.ocr(np.asarray(image))),
            ('temporal', self.temporal is not None,
             lambda: self.temporal_score(np.zeros((self.config['window_frames'], 768), np.float32),
                                         [True] * self.config['window_frames'],np.arange(self.config['window_frames'])*.125,
                                         np.zeros((self.config['window_frames'],2)))),
            ('standalone_generation', getattr(self, 'image_adapter', None) is not None,
             lambda: self.image_adapter_score(image, self.encode([image])[0][0])),
            ('video_origin', getattr(self, 'video_adapter', None) is not None,
             lambda: self.video_origin_score([image,image],self.encode([image,image])[0])),
        ]
        warmup = {}
        for branch, available, function in jobs:
            if not available or branch in self.errors:
                warmup[branch] = {'status': 'skipped'}
                continue
            began = time.perf_counter()
            try:
                function()
                warmup[branch] = {'status': 'success'}
            except Exception as exc:
                self._disable_model(branch, exc)
                warmup[branch] = {'status': 'failed', 'error': self.errors[branch]}
            warmup[branch]['elapsed_ms'] = round((time.perf_counter() - began) * 1000, 2)
        self.info['warmup'] = warmup
        # Pre-create convolution plans for recorded development input shapes.
        # These dummy images never become material evidence. Startup time is
        # visible; unseen live shapes still remain inside the live stopwatch.
        shape_start=time.perf_counter()
        warmed=[]
        if self.trufor is not None:
            shapes={tuple(size) for size in self.config.get('trufor_warmup_shapes',[])}
            shapes.update((height,width) for width,height in list(shapes))
            for width,height in sorted(shapes):
                try:
                    self.localize(Image.new('RGB',(width,height),(128,128,128)))
                    warmed.append([width,height])
                except Exception as exc:
                    self._disable_model('localization',exc)
                    break
        warmup['localization_shapes']={'shapes':warmed,'elapsed_ms':round((time.perf_counter()-shape_start)*1000,2),
                                       'source':'development input shapes only; no heldout inspection'}

    def autocast(self):
        return torch.autocast('cuda', dtype=torch.float16) if self.device == 'cuda' else nullcontext()

    @torch.inference_mode()
    def encode(self, images):
        if self.clip is None or self.head is None or 'generation' in self.errors:
            raise RuntimeError(self.errors.get('generation', 'Generation model unavailable'))
        features, scores, scenes = [], [], []
        for offset in range(0, len(images), self.config['model_batch_size']):
            batch = torch.stack([self.preprocess(im) for im in images[offset:offset + self.config['model_batch_size']]]).to(self.device)
            if self.device == 'cuda':
                batch = batch.half()
            feat = self.clip.encode_image(batch, normalize=False).float()
            score = self.head(feat).sigmoid().flatten()
            similarity = torch.nn.functional.normalize(feat, dim=-1) @ self.scene_text.T
            features.append(feat.cpu().numpy())
            scores.extend(score.cpu().tolist())
            scenes.extend(similarity.cpu().numpy())
        return np.concatenate(features), scores, np.asarray(scenes)

    def domain_score(self, image, center_feature, global_feature=None):
        if self.domain_head is None or 'domain_generation' in self.errors:
            raise RuntimeError(self.errors.get('domain_generation','Domain head unavailable'))
        if global_feature is None:
            global_feature = self.encode([global_view(image)])[0][0]
        support = descriptor(center_feature, global_feature)
        if self.domain_head['feature_spec'] == MULTIBLOCK_FEATURE_SPEC:
            from .development_texture import TEXTURE_SPEC, texture_features
            texture, names = texture_features(image)
            if TEXTURE_SPEC != self.domain_head['feature_blocks'][2]['texture_spec'] or len(names) != 53:
                raise ValueError('Domain texture schema differs from the training checkpoint')
            return domain_predict(self.domain_head, block_descriptor(self.domain_head, support, texture), support)
        return domain_predict(self.domain_head, support)

    @torch.inference_mode()
    def image_adapter_score(self, image, center_feature, global_feature=None, return_descriptor=False):
        if self.image_adapter is None or 'standalone_generation' in self.errors:
            raise RuntimeError(self.errors.get('standalone_generation', 'Public image-origin adapter unavailable'))
        if self.waverep is None or 'forensic_generation' in self.errors:
            raise RuntimeError('Image adapter requires the frozen WaveRep backbone')
        from .standalone_generation import descriptor as image_descriptor, predict as image_predict
        if global_feature is None:
            global_feature = self.encode([global_view(image)])[0][0]
        whole = ImageOps.pad(image, (504, 504), method=Image.Resampling.BICUBIC, color=(127, 127, 127), centering=(.5, .5))
        tensor = torch.stack([self.wave_preprocess(image), self.wave_preprocess(whole)]).to(self.device)
        with self.autocast():
            tokens = self.waverep.forward_features(tensor)
            views = self.waverep.forward_head(tokens, pre_logits=True).float().cpu().numpy()
            native = self.waverep.forward_head(tokens).float().flatten().cpu().tolist()
        texture = None
        if any(block['name'] == 'texture' for block in self.image_adapter['feature_blocks']):
            from .development_texture import texture_features
            texture, _ = texture_features(image)
        features = image_descriptor(center_feature, global_feature, views[0], views[1], texture)
        result=(image_predict(self.image_adapter, features), native[0], global_feature)
        return (*result,features) if return_descriptor else result

    @torch.inference_mode()
    def video_origin_score(self, images, center_features):
        if self.video_adapter is None or 'video_origin' in self.errors:
            raise RuntimeError(self.errors.get('video_origin','Video-origin head unavailable'))
        from .video_origin import predict as video_predict
        descriptors=[]
        for image,center in zip(images,center_features):
            *_,value=self.image_adapter_score(image,center,return_descriptor=True)
            if len(value)==3072 and any(b['name']=='texture' for b in self.video_adapter['feature_blocks']):
                from .development_texture import texture_features
                texture,_=texture_features(image);value=np.r_[value,texture]
            descriptors.append(value)
        return video_predict(self.video_adapter,descriptors)

    @torch.inference_mode()
    def localize(self, image, max_side=None):
        if self.trufor is None or 'localization' in self.errors:
            raise RuntimeError(self.errors.get('localization', 'Localization model unavailable'))
        width, height = image.size
        factor = min(1, (max_side or self.config['spatial_max_side']) / max(width, height))
        size = (max(64, int(round(width * factor))), max(64, int(round(height * factor))))
        small = image.resize(size, Image.Resampling.BILINEAR) if size != image.size else image
        array = np.asarray(small).copy()
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 256.0
        with self.autocast():
            prediction, confidence, detection, _ = self.trufor(tensor)
        probability = prediction.float().softmax(1)[0, 1].cpu().numpy()
        reliability = confidence.float().sigmoid()[0, 0].cpu().numpy()
        score = float(detection.float().sigmoid().item())
        return score, probability, reliability, size

    @torch.inference_mode()
    def temporal_score(self, features, valid_mask, timestamps=None, quality=None):
        if self.temporal is None or 'temporal' in self.errors:
            return None
        tensor = torch.from_numpy(np.asarray(features, np.float32)).unsqueeze(0).to(self.device)
        mask = torch.as_tensor(valid_mask, device=self.device).unsqueeze(0)
        if self.info.get('temporal',{}).get('architecture')=='pts_velocity_quality_tcn_v2':
            from .temporal_v2 import validate_window
            if timestamps is None or quality is None:raise ValueError('Time-aware TCN requires actual PTS and frame quality')
            validate_window(features,valid_mask,timestamps,quality)
            times=torch.as_tensor(timestamps,dtype=torch.float32,device=self.device).unsqueeze(0)
            q=torch.as_tensor(quality,dtype=torch.float32,device=self.device).unsqueeze(0)
            return float(self.temporal(tensor,mask,times,q).sigmoid().item())
        return float(self.temporal(tensor, mask).sigmoid().item())

    @torch.inference_mode()
    def localize_batch(self, images, max_side=320, batch_size=4):
        if self.trufor is None or 'localization' in self.errors:
            raise RuntimeError(self.errors.get('localization','Localization model unavailable'))
        results=[]
        for offset in range(0,len(images),batch_size):
            group=images[offset:offset+batch_size];sizes=[];tensors=[]
            for image in group:
                factor=min(1.,max_side/max(image.size))
                size=tuple(max(64,round(v*factor)) for v in image.size);sizes.append(size)
                small=image.resize(size,Image.Resampling.BILINEAR) if size!=image.size else image
                tensors.append(torch.from_numpy(np.asarray(small).copy().transpose(2,0,1)).float()/256.)
            if len(set(sizes))!=1:raise ValueError('Spatial batches require identical decoded frame sizes')
            batch=torch.stack(tensors).to(self.device)
            with self.autocast():prediction,confidence,detection,_=self.trufor(batch)
            probabilities=prediction.float().softmax(1)[:,1].cpu().numpy()
            reliabilities=confidence.float().sigmoid()[:,0].cpu().numpy()
            scores=detection.float().sigmoid().reshape(-1).cpu().tolist()
            if len(scores)!=len(group):raise ValueError('Batched localization detection dimensions differ')
            results.extend(zip(scores,probabilities,reliabilities,sizes))
        return results

    @torch.inference_mode()
    def wave_scores(self, images):
        if self.waverep is None or 'forensic_generation' in self.errors:
            raise RuntimeError(self.errors.get('forensic_generation','WaveRep checkpoint unavailable'))
        logits=[]
        for offset in range(0,len(images),2):
            tensor=torch.stack([self.wave_preprocess(im) for im in images[offset:offset+2]]).to(self.device)
            with self.autocast():
                values=self.waverep(tensor).float().flatten()
            logits.extend(values.cpu().tolist())
        return logits
