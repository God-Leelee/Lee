"""Offline, complementary image-origin experts with explicit raw-score semantics.

Community Forensics preprocessing follows the official evaluation transform.
B-Free uses its original embedding-level five-crop wrapper and normalisation.
No filename, metadata, ground truth, or external API enters these networks.
"""
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import timm
from safetensors.torch import load_file
from torchvision import transforms

from .contracts import ROOT, sha256

FEATURES = ['commfor224.center_logit', 'commfor224.global_logit',
            'commfor384.center_logit', 'commfor384.global_logit', 'bfree.logit']
FEATURE_SPEC = 'commfor224_384_center_global_bfree_native5_v1'


class GenerationExperts:
    def __init__(self, config, device):
        from .representation_adapter import EXPERT_SPEC, NATIVE_EXPERT_SPEC, NATIVE_FEATURE_SPEC, validate, numeric_arrays
        if config.get('feature_spec') not in (FEATURE_SPEC, EXPERT_SPEC, NATIVE_EXPERT_SPEC) or config.get('precision') != 'float32':
            raise ValueError('Expert feature/preprocessing precision differs from frozen policy')
        self.device = device
        self.models, self.transforms = {}, {}
        self.config = config
        self.adaptation = None
        if config['feature_spec'] in (EXPERT_SPEC, NATIVE_EXPERT_SPEC):
            import json
            path = ROOT / config['adaptation']['checkpoint']
            if sha256(path) != config['adaptation']['sha256']:
                raise ValueError('Adaptation checkpoint differs from frozen configuration')
            self.adaptation = json.loads(path.read_text('utf-8'))
            validate(self.adaptation, config['commfor224']['sha256'])
            if (config['feature_spec'] == NATIVE_EXPERT_SPEC) != (self.adaptation['feature_spec'] == NATIVE_FEATURE_SPEC):
                raise ValueError('Expert and adaptation preprocessing versions differ')
            self.adaptation = numeric_arrays(self.adaptation)
        for size in ((224,) if self.adaptation is not None else (224, 384)):
            key = f'commfor{size}'
            spec = config[key]
            path = ROOT / spec['checkpoint']
            if sha256(path) != spec['sha256']:
                raise ValueError(f'{key} checkpoint hash mismatch')
            net = timm.create_model(f'vit_small_patch16_{size}.augreg_in21k_ft_in1k',
                                    pretrained=False, num_classes=1)
            weights = load_file(str(path), device='cpu')
            net.load_state_dict({k.removeprefix('vit.'): v for k, v in weights.items()}, strict=True)
            self.models[key] = net.to(device).eval()
            self.transforms[key] = transforms.Compose([
                transforms.Resize(256 if size == 224 else 440), transforms.CenterCrop(size),
                transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        if self.adaptation is not None:
            return
        spec = config['bfree']
        path = ROOT / spec['checkpoint']
        if sha256(path) != spec['sha256']:
            raise ValueError('B-Free checkpoint hash mismatch')
        # Load only the audited upstream wrapper, without importing its CLI,
        # pandas, training tools or downloading backbone weights at runtime.
        wrapper_file = ROOT / 'third_party/B-Free/code/networks/wrapper5crops.py'
        if sha256(wrapper_file) != config['bfree']['wrapper_sha256']:
            raise ValueError('B-Free crop implementation differs from frozen policy')
        module_spec = importlib.util.spec_from_file_location('cleartrace_bfree_crops', wrapper_file)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        net = timm.create_model(spec['architecture'], num_classes=1, pretrained=False)
        net = module.Wrapper5crops(net, 504)
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        net.load_state_dict(checkpoint['model'])
        self.models['bfree'] = net.to(device).eval()
        self.transforms['bfree'] = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    @torch.inference_mode()
    def features(self, image):
        if image.mode != 'RGB':
            raise ValueError('Experts require the same decoded RGB image as the other branches')
        # A padded square preserves edge content; native centre and native
        # B-Free crops preserve fine texture. These are distinct, versioned views.
        global_image = ImageOps.pad(image, (512, 512), method=Image.Resampling.BICUBIC,
                                   color=(127, 127, 127), centering=(.5, .5))
        values = []
        if self.adaptation is not None:
            from .representation_adapter import descriptor, raw_logit, views, FEATURES as adapted_features
            feature_spec = self.adaptation['feature_spec']
            tensor = torch.stack(views(image, feature_spec, self.transforms['commfor224'])).to(self.device)
            tokens = self.models['commfor224'].forward_features(tensor).float()
            logits = self.models['commfor224'].forward_head(tokens).float().flatten().cpu().tolist()
            feature = descriptor(tokens, feature_spec).cpu().numpy()
            values = logits[:2] + [raw_logit(self.adaptation, feature)]
            if not np.isfinite(values).all():
                raise ValueError('Non-finite adapted expert output')
            return dict(zip(adapted_features, values))
        for size in (224, 384):
            key = f'commfor{size}'
            tensor = torch.stack([self.transforms[key](view) for view in (image, global_image)]).to(self.device)
            # Full precision matches the upstream reference. Mixed precision
            # shifted marginal portrait scores in development, so it is not used.
            logits = self.models[key](tensor).float().flatten()
            values.extend(logits.cpu().tolist())
        # Cap only the raw patch projection size to bound memory on 50MP inputs.
        # The native crop policy remains intact below 2048px; this cap is logged.
        factor = min(1., 2048 / max(image.size))
        bfree_image = image if factor == 1 else image.resize(
            (max(14, round(image.width * factor)), max(14, round(image.height * factor))), Image.Resampling.LANCZOS)
        if min(bfree_image.size) < 14:
            bfree_image = ImageOps.pad(bfree_image, (max(14, bfree_image.width), max(14, bfree_image.height)))
        tensor = self.transforms['bfree'](bfree_image).unsqueeze(0).to(self.device)
        logit = self.models['bfree'](tensor).float().item()
        values.append(logit)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite expert output')
        return dict(zip(FEATURES, values))


def expert_config():
    """Resolve local, hashed weights; does not enable any live branch."""
    import json
    out = ROOT / 'models/generation_experts_v1.6'
    records = {r['file']: r for r in json.loads((out / 'sources.json').read_text('utf-8'))}
    import yaml
    bfree = yaml.safe_load((out / 'bfree_config.yaml').read_text('utf-8'))
    config = {'feature_spec': FEATURE_SPEC, 'enabled': True, 'precision': 'float32'}
    for size in (224, 384):
        name = f'commfor_{size}.safetensors'
        config[f'commfor{size}'] = {'checkpoint': (out / name).relative_to(ROOT).as_posix(),
                                     'sha256': records[name]['sha256']}
    name = 'bfree_model_epoch_best.pth'
    config['bfree'] = {'checkpoint': (out / name).relative_to(ROOT).as_posix(),
                       'sha256': records[name]['sha256'], 'architecture': bfree['arch'][12:],
                       'wrapper_sha256': sha256(ROOT / 'third_party/B-Free/code/networks/wrapper5crops.py')}
    return config
