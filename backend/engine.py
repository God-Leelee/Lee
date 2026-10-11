from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
from PIL import Image

from .contracts import Branch, sha256, atomic_json
from .copy_move import detect_copy_move, refine_with_localization
from .fusion import fuse
from .media import read_image, sample_video, metadata_signals, d3_statistics
from .multimodal import analyze_multimodal


def measured(callable_):
    start = time.perf_counter()
    result = callable_()
    return result, round((time.perf_counter() - start) * 1000, 2)


def guarded(branch_id, fn):
    try:
        return fn()
    except Exception as exc:
        return Branch(branch_id, 'failed', reason=f'{type(exc).__name__}: {exc}')


def save_localization(image, models, out: Path, config, prefix='', max_side=None):
    began = time.perf_counter()
    prediction = models.localize(image, max_side=max_side) if max_side else models.localize(image)
    return persist_localization(image,prediction,out,config,prefix,max_side,began)


def persist_localization(image, prediction, out, config, prefix='', max_side=None, began=None):
    began=began or time.perf_counter()
    score, probability, reliability, model_size = prediction
    probability, reliability = np.asarray(probability), np.asarray(reliability)
    if (not np.isfinite(score) or not 0 <= score <= 1 or probability.ndim != 2 or not probability.size
        or probability.shape != reliability.shape or not np.isfinite(probability).all()
        or not np.isfinite(reliability).all() or np.any(probability < 0) or np.any(probability > 1)
        or np.any(reliability < 0) or np.any(reliability > 1)):
        raise ValueError('局部模型输出无效；不能把无效掩码解释为正常材料。')
    width, height = image.size
    # Pixel masks are restored with nearest interpolation, preserving label values.
    binary = ((probability >= config['mask_threshold']) & (reliability >= .5)).astype(np.uint8)
    mask_full = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
    mask_name, heat_name, reliability_name = prefix + 'mask.png', prefix + 'heatmap.png', prefix + 'reliability.png'
    Image.fromarray(mask_full * 255).save(out / mask_name, optimize=False, compress_level=1)
    # Evidence display uses a transparent raw probability layer, independent of mask.
    heat = np.zeros((*probability.shape, 4), np.uint8)
    heat[..., 0], heat[..., 1], heat[..., 2] = 255, 63, 89
    heat[..., 3] = np.clip(probability * reliability * 210, 0, 210).astype(np.uint8)
    Image.fromarray(heat).save(out / heat_name)
    Image.fromarray((reliability * 255).astype(np.uint8)).save(out / reliability_name)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    regions = []
    sx, sy = width / probability.shape[1], height / probability.shape[0]
    for x, y, w, h, area in stats[1:]:
        if area < max(9, binary.size * .0002):
            continue
        regions.append({'bbox': [float(x * sx), float(y * sy), float(w * sx), float(h * sy)],
                        'area_pixels': int(area * sx * sy), 'coordinate_space': 'original_raw_pixels'})
    regions = sorted(regions, key=lambda r: r['area_pixels'], reverse=True)[:12]
    from .evidence_fusion import localization_statistics
    spatial_statistics=localization_statistics(probability,reliability,config['mask_threshold'])
    return Branch('localization', 'success', score, model='TruFor',
                  elapsed_ms=round((time.perf_counter() - began) * 1000, 2),
                  evidence=[{'kind': 'localization', 'text': '模型预测的篡改候选区域；低可靠性像素已从二值掩码排除。',
                             'regions': regions, 'mask': mask_name, 'heatmap': heat_name,
                             'reliability': reliability_name, 'width': width, 'height': height,
                             'coordinate_space': 'original_raw_pixels'}],
                  details={'model_input_size': model_size, 'original_size': [width, height],
                           'mask_threshold': config['mask_threshold'], 'reliability_threshold': .5,
                           'resolution_tier': 'coarse_sampled_frame' if max_side else 'detailed_frame',
                           'reliable_area_ratio': float(binary.mean()), 'mean_reliability': float(reliability.mean()),
                           'region_count': len(regions), 'mask': mask_name, 'heatmap': heat_name,
                           **spatial_statistics,
                           'reliability': reliability_name,
                           'full_generation_localization_limit': '整图生成可能没有局部篡改边界；空掩码不能证明非AI。'})


def human_explanation(branches, fusion):
    lines = []
    for b in branches:
        if b.status == 'failed':
            lines.append({'branch_id': b.branch_id, 'text': '该分支运行失败，未纳入风险评分。', 'evidence_ids': []})
        elif b.branch_id == 'generation' and b.score is not None:
            lines.append({'branch_id': b.branch_id, 'text': f'生成特征风险分为{b.score:.3f}；属于未经概率校准的模型输出。',
                          'evidence_ids': [e['evidence_id'] for e in b.evidence]})
        elif b.branch_id == 'domain_generation' and b.status == 'success':
            lines.append({'branch_id':b.branch_id,'text':'已运行本地样本训练的AI来源适配模型；该分支目前用于辅助复核，独立测试效果见实测看板。',
                          'evidence_ids':[e['evidence_id'] for e in b.evidence]})
        elif b.branch_id == 'standalone_generation' and b.status == 'success':
            lines.append({'branch_id': b.branch_id,
                          'text': f'多视图模型的公开图片来源校准概率为{b.score:.1%}；当前仅辅助复核，不改变主判定，亦不等于理赔欺诈概率。' if b.details.get('decision_role') == 'review_only' else f'多视图模型的公开图片来源校准概率为{b.score:.1%}；局部编辑和复制几何证据独立保留，此概率不等于理赔欺诈概率。',
                          'evidence_ids': [e['evidence_id'] for e in b.evidence]})
        elif b.branch_id == 'localization' and b.status == 'success':
            lines.append({'branch_id': b.branch_id, 'text': f'取证模型标出{b.details.get("region_count", 0)}个候选区域，可靠像素占{b.details.get("reliable_area_ratio", 0):.1%}。',
                          'evidence_ids': [e['evidence_id'] for e in b.evidence]})
        elif b.branch_id == 'copy_move' and b.evidence:
            lines.append({'branch_id': b.branch_id, 'text': b.evidence[0]['text'], 'evidence_ids': [e['evidence_id'] for e in b.evidence]})
        elif b.branch_id == 'temporal':
            lines.append({'branch_id': b.branch_id, 'text': '连续窗口实际计算了时序变化；只有有效训练模型输出才参与融合。',
                          'evidence_ids': [e['evidence_id'] for e in b.evidence]})
    if fusion['verdict'] == 'lower_risk':
        lines.append({'branch_id': 'fusion', 'text': '当前证据未达到风险阈值；这不是真实性证明，仍需结合来源和案件资料。', 'evidence_ids': []})
    return lines


class Engine:
    def __init__(self, config, models):
        self.config, self.models = config, models
        self.cpu = ThreadPoolExecutor(max_workers=2, thread_name_prefix='evidence')

    def image(self, path: Path, out: Path, description=''):
        image, metadata = read_image(path, self.config['image_max_pixels'])
        image.save(out / 'preview.jpg', quality=93)
        metadata_branch = Branch('metadata', 'success', score_kind='not_a_fake_score', model='Pillow original EXIF/XMP/info',
                                 evidence=metadata_signals(metadata), details=metadata,
                                 reason='未发现元数据来源标记；元数据缺失不表示伪造。' if not metadata_signals(metadata) else None)
        copy_future = self.cpu.submit(guarded, 'copy_move', lambda: detect_copy_move(image, self.config['copy_max_side']))
        branches = [metadata_branch]
        native_wave = None
        clip_global = None
        try:
            (features, scores, scenes), elapsed = measured(lambda: self.models.encode([image]))
            generation = Branch('generation', 'success', float(scores[0]), model='UFD / CLIP ViT-L/14', elapsed_ms=elapsed,
                                evidence=[{'kind': 'generation_feature', 'text': '原始全图中心裁剪的CLIP特征经公开UFD分类头实际推理。',
                                           'score': float(scores[0]), 'preprocess': self.models.info['generation']['preprocess']}],
                                details={'crop': 'center_224', 'original_size': list(image.size),
                                         'scope_limit': '中心裁剪可能漏掉边缘局部编辑；局部取证由TruFor和复制匹配补充。'})
            branches.append(generation)
            multimodal_future = self.cpu.submit(guarded, 'multimodal', lambda: analyze_multimodal(image, scenes[0], self.models, self.config, description))
            if self.config.get('image_origin_adapter', {}).get('enabled'):
                try:
                    (probability, native_wave, clip_global), origin_ms = measured(lambda: self.models.image_adapter_score(image, features[0]))
                    info = self.models.info['standalone_generation']
                    branches.append(Branch('standalone_generation', 'success', probability,
                        score_kind='public_image_origin_probability', model=info['model'], elapsed_ms=origin_ms,
                        evidence=[{'kind': 'image_origin', 'text': '中心与全图的冻结视觉特征联合训练，在独立开发分区校准；生成来源判断与局部篡改证据共同参与材料核验。',
                                   'probability_scope': info['scope']}],
                        details={'model_sha256': info['sha256'], 'threshold': info['threshold'],
                                 'partitions': info['partitions'], 'candidate': info['candidate'],
                                 'decision_role': self.config['image_origin_adapter']['decision_role'],
                                 'validation_note': self.config['image_origin_adapter'].get('validation_note'),
                                 'shared_native_wave_center': True,
                                 'non_ai_limit': '低生成来源概率不能排除传统篡改或证明理赔真实。'}))
                    if probability >= info['threshold']:
                        branches[-1].evidence.append({'kind':'full_frame_source_suspicion','bbox':[0,0,image.width,image.height],
                          'coordinate_space':'original_raw_pixels','probability':probability,
                          'text':'来源模型提示整图生成风险；区域范围为全图，不是局部像素掩码，也不能证明每个像素被伪造。'})
                except Exception as exc:
                    branches.append(Branch('standalone_generation', 'failed', reason=f'{type(exc).__name__}: {exc}'))
            if self.config.get('domain_generation',{}).get('enabled'):
                branches.append(guarded('domain_generation',lambda:self.domain_branch(image,features[0],clip_global)))
        except Exception as exc:
            branches.append(Branch('generation', 'failed', reason=str(exc)))
            multimodal_future = None
        branches.append(guarded('localization', lambda: save_localization(image, self.models, out, self.config)))
        if self.models.waverep is not None:
            branches.append(guarded('forensic_generation', lambda: self.wave_branch([image], native_wave)))
        copied = copy_future.result()
        localized = next(b for b in branches if b.branch_id=='localization')
        if (out/'mask.png').exists():
            copied = refine_with_localization(copied,localized,out/'mask.png',image.size)
        branches.append(copied)
        branches.append(multimodal_future.result() if multimodal_future else Branch('multimodal', 'failed', reason='场景编码失败，无法完成图文联合分析。'))
        if metadata['coordinate_space']=='decoded_image_pixels':
            def decoded_coordinates(value):
                if isinstance(value,dict):
                    if value.get('coordinate_space')=='original_raw_pixels':value['coordinate_space']='decoded_image_pixels'
                    for nested in value.values():decoded_coordinates(nested)
                elif isinstance(value,list):
                    for nested in value:decoded_coordinates(nested)
            for branch in branches:
                decoded_coordinates(branch.details);decoded_coordinates(branch.evidence)
        return branches, {'width': image.width, 'height': image.height, 'preview': 'preview.jpg', 'metadata': metadata}

    def domain_branch(self,image,center_feature,global_feature=None):
        (probability,supported,similarity),elapsed=measured(lambda:self.models.domain_score(image,center_feature,global_feature))
        info=self.models.info['domain_generation'];role=self.config['domain_generation']['decision_role']
        return Branch('domain_generation','success' if supported else 'not_applicable',probability if supported else None,
            score_kind='dataset_scoped_ai_origin_probability',model=info['model'],elapsed_ms=elapsed,
            reason=None if supported else '与训练材料的特征距离较大，本分支暂不提供AI来源判断。',
            evidence=[{'kind':'domain_ai_origin','text':'使用全图与中心细节两种视觉特征，按资源包AI／非AI标签实际训练；不读取标签、文件名或表格总结。',
                       'probability_scope':info['scope'],'decision_role':role,'threshold':info['threshold']}],
            details={'supported':supported,'nearest_training_cosine':similarity,'support_min_cosine':self.models.domain_head['support_min_cosine'],
                     'decision_role':role,'threshold':info['threshold'],'model_sha256':info['sha256'],
                     'frozen_split_sha256':info['frozen_split_sha256'],'partitions':info['partitions'],
                     'validation_note':f'{self.config["domain_generation"].get("validation_label", "保留测试")}包含{self.config["domain_generation"]["validation_metrics"]["samples"]}张；拒判计入分母后的准确率为{self.config["domain_generation"]["validation_metrics"]["accuracy"]:.0%}，目前用于辅助复核。',
                     'non_ai_limit':'非AI标签不证明无传统篡改，也不证明理赔真实。'})

    def video(self, path: Path, out: Path, description=''):
        (frames, windows, metadata), decode_ms = measured(lambda: sample_video(path, self.config))
        if not frames:
            raise ValueError('没有有效视频帧。')
        branches = [Branch('metadata', 'success', score_kind='not_a_fake_score', model='PyAV original container/track fields',
                           elapsed_ms=decode_ms, evidence=metadata_signals(metadata), details=metadata)]
        frame_rows = []
        for f in frames:
            name = f'frame_{f.frame_id:03d}.jpg'
            frame_rows.append({'frame_id': f.frame_id, 'timestamp_seconds': f.timestamp, 'pts': f.pts,
                               'time_base': f.time_base, 'preview': name, 'width': f.image.width, 'height': f.image.height})
        def persist_previews():
            for frame,row in zip(frames,frame_rows):
                frame.image.save(out/row['preview'],quality=88)
        preview_future=self.cpu.submit(persist_previews)
        features = scores = scenes = None
        try:
            (features, scores, scenes), elapsed = measured(lambda: self.models.encode([f.image for f in frames]))
            for row, score in zip(frame_rows, scores):
                row['generation_risk'] = float(score)
            image_score = float(np.quantile(scores, .9))
            branches.append(Branch('generation', 'success', image_score, model='UFD sampled-frame quantile', elapsed_ms=elapsed,
                            evidence=[{'kind': 'frame_generation', 'text': '采样帧风险的90%分位聚合；保留每帧实际PTS。', 'frames': frame_rows}],
                            details={'aggregation': 'q90', 'frame_scores': scores, 'frames': len(scores)}))
        except Exception as exc:
            branches.append(Branch('generation', 'failed', reason=f'{type(exc).__name__}: {exc}'))
        if self.models.waverep is not None:
            wave_branch = guarded('forensic_generation', lambda: self.wave_branch([f.image for f in frames]))
            branches.append(wave_branch)
        else:
            wave_branch = None
        if self.config.get('video_origin_adapter',{}).get('enabled'):
            def video_source_branch():
                if features is None:raise RuntimeError('Shared keyframe encoding failed')
                began=time.perf_counter();ids=[len(frames)//4,3*len(frames)//4]
                probability,raw=self.models.video_origin_score([frames[i].image for i in ids],features[ids])
                locations=[{'frame_id':i,'pts':frames[i].pts,'time_base':frames[i].time_base,
                    'timestamp_seconds':frames[i].timestamp,'raw_origin_logit':logit} for i,logit in zip(ids,raw)]
                return Branch('video_origin','success',probability,score_kind='public_video_origin_reference_probability',
                    model=self.models.info['video_origin']['model'],elapsed_ms=(time.perf_counter()-began)*1000,
                    evidence=[{'kind':'video_origin','text':'两个固定时间点的实际关键帧，分析中心细节与全图来源特征；结果进入联合融合。',
                        'frames':locations,'meaning':'来源疑点，不代表每帧或每个像素均被伪造。'}],
                    details={**self.models.info['video_origin'],'frames':locations,'decision_role':'ensemble_feature'})
            branches.append(guarded('video_origin',video_source_branch))
        # Selection fallback is a position rule, never a synthetic risk score.
        selection_scores = np.asarray(scores).copy() if scores is not None else np.arange(len(frames), dtype=float)/len(frames)
        selection_rule = 'max(UFD, WaveRep) frame risk; fixed top-2 budget' if scores is not None else 'uniform-position fallback because encoder failed'
        if wave_branch and wave_branch.status == 'success':
            wave_risks = 1 / (1 + np.exp(-np.clip(wave_branch.details['frame_logits'], -50, 50)))
            selection_scores = np.maximum(selection_scores, wave_risks) if scores is not None else wave_risks
            selection_rule = 'max(UFD, WaveRep) frame risk; fixed top-2 budget' if scores is not None else 'WaveRep frame risk; encoder failed'
            for row, risk in zip(frame_rows, wave_risks):
                row['forensic_generation_risk'] = float(risk)
        main=int(np.argmax(selection_scores))
        text_future=self.cpu.submit(guarded,'multimodal',lambda: analyze_multimodal(frames[main].image,scenes[main],self.models,self.config,description,frames[main].timestamp)) if scenes is not None else None
        temporal_start = time.perf_counter()
        temporal_scores = []
        temporal_error = None
        for window in windows:
            window['temporal_risk'] = None
            try:
                if features is None:
                    raise RuntimeError('连续帧编码失败，无法计算真实时序特征。')
                feature = features[window['frame_ids']]
                window['d3_statistics'] = d3_statistics(feature, window['valid_mask'], window['actual_timestamps'])
                if sum(window['valid_mask']) >= 4:
                    if self.models.info.get('temporal',{}).get('architecture')=='pts_velocity_quality_tcn_v2':
                        from .temporal_v2 import frame_quality
                        quality=[frame_quality(frames[i].image) for i in window['frame_ids']]
                        window['frame_quality']=quality
                        window['temporal_risk']=self.models.temporal_score(feature,window['valid_mask'],window['actual_timestamps'],quality)
                    else:
                        window['temporal_risk']=self.models.temporal_score(feature,window['valid_mask'])
                else:
                    window['temporal_risk']=None
                if window['temporal_risk'] is not None:
                    temporal_scores.append(window['temporal_risk'])
            except Exception as exc:
                temporal_error = f'{type(exc).__name__}: {exc}'
                window['d3_statistics'] = {'status': 'failed', 'reason': temporal_error}
        if self.models.errors.get('temporal'):
            temporal_error = self.models.errors['temporal']
        tcn_status = 'failed' if temporal_error else 'success' if temporal_scores else 'insufficient_evidence'
        temporal_risk=max(temporal_scores) if temporal_scores and not temporal_error else None
        temporal_aggregation='max_window_risk'
        if temporal_risk is not None and self.models.info.get('temporal',{}).get('architecture')=='pts_velocity_quality_tcn_v2':
            from scipy.special import logsumexp, expit
            p=np.clip(temporal_scores,1e-8,1-1e-8);logits=np.log(p/(1-p))
            temporal_risk=float(expit(.5*logsumexp(logits/.5)-.5*np.log(len(logits))))
            temporal_aggregation='trained_MIL_logsumexp_temperature_0.5'
        branches.append(Branch('temporal', tcn_status, temporal_risk,
                        model='Trained TCN + D3-inspired continuous-window statistics',
                        elapsed_ms=round((time.perf_counter() - temporal_start) * 1000, 2),
                        evidence=[{'kind': 'temporal_window', 'text': '窗口含有效帧掩码及实际时间差；统计量不作为概率。', **w} for w in windows],
                        details={'windows': windows, 'checkpoint': self.models.info.get('temporal'),
                                 'aggregation':temporal_aggregation,
                                 'localization_kind': 'sampled_window_suspicion_not_ground_truth',
                                 'unobserved_gaps_not_bridged': True},
                        reason=temporal_error or (None if temporal_scores else 'TCN尚无已训练权重或窗口有效帧不足；仅展示真实时序统计，不以随机权重打分。')))
        # Sampled suspicious frames are separate locations, never a fabricated
        # interval spanning gaps. Model checks remain visible per selected frame.
        screening_start = time.perf_counter()
        screening, screening_error, coarse_localized, coarse_pending = [], None, {}, {}
        batched_predictions=None
        if self.config.get('video_screening_batch_size',1)>1:
            try:
                batched_predictions=self.models.localize_batch([f.image for f in frames],
                    max_side=self.config.get('video_screening_max_side',192),batch_size=self.config['video_screening_batch_size'])
                if len(batched_predictions)!=len(frames):raise ValueError('Incomplete spatial batch output')
            except Exception as exc:
                screening_error=f'{type(exc).__name__}: {exc}'
        for i in range(len(frames)):
            try:
                if screening_error:raise RuntimeError(screening_error)
                began=time.perf_counter()
                prediction=batched_predictions[i] if batched_predictions is not None else self.models.localize(frames[i].image,max_side=self.config.get('video_screening_max_side',192))
                coarse_pending[i]=self.cpu.submit(guarded,'localization',lambda i=i,prediction=prediction,began=began:persist_localization(frames[i].image,prediction,out,self.config,f'coarse_f{i:03d}_',self.config.get('video_screening_max_side',192),began))
                score,probability,reliability,size=prediction
                area=float(((probability>=self.config['mask_threshold'])&(reliability>=.5)).mean())
                record={'frame_id':i,'timestamp_seconds':frames[i].timestamp,'risk':score,
                        'reliable_area_ratio':area,'model_input_size':size,
                        'meaning':'低分辨率逐采样帧取证；掩码回投原始坐标，并保留模型网格大小'}
                screening.append(record)
                frame_rows[i]['coarse_localization']=record
            except Exception as exc:
                screening_error=f'{type(exc).__name__}: {exc}'
                break
        screening_ms=round((time.perf_counter()-screening_start)*1000,2)
        selected = np.argsort(selection_scores)[-self.config['video_spatial_frames']:].tolist()
        if screening:
            best_spatial=max(screening,key=lambda r:r['risk'] if r['reliable_area_ratio']>=.005 else r['risk']*.5)['frame_id']
            selected=[best_spatial]+[i for i in reversed(np.argsort(selection_scores).tolist()) if i!=best_spatial][:self.config['video_spatial_frames']-1]
            selection_rule+='; one slot reserved for max coarse TruFor over all sampled frames'
        selected=sorted(selected)
        for i,future in coarse_pending.items():
            coarse=future.result()
            coarse.details.update({'frame_id':i,'timestamp_seconds':frames[i].timestamp})
            for evidence in coarse.evidence:
                evidence.update({'frame_id':i,'timestamp_seconds':frames[i].timestamp,'resolution_tier':'coarse_sampled_frame'})
            coarse_localized[i]=coarse
            frame_rows[i]['localization']=coarse.to_dict()
            if coarse.status=='failed':
                screening_error=coarse.reason
        screening_ms=round((time.perf_counter()-screening_start)*1000,2)
        localized, copied = [], []
        copy_futures={i:self.cpu.submit(guarded,'copy_move',lambda i=i:detect_copy_move(frames[i].image,self.config['copy_max_side'])) for i in selected}
        for i in selected:
            local = guarded('localization', lambda i=i: save_localization(frames[i].image, self.models, out, self.config, f'f{i:03d}_'))
            local.details.update({'frame_id': i, 'timestamp_seconds': frames[i].timestamp})
            for evidence in local.evidence:
                evidence.update({'frame_id': i, 'timestamp_seconds': frames[i].timestamp})
            localized.append(local)
            copied_frame = copy_futures[i].result()
            if (out/f'f{i:03d}_mask.png').exists():
                copied_frame = refine_with_localization(copied_frame,local,out/f'f{i:03d}_mask.png',frames[i].image.size)
            copied_frame.details.update({'frame_id': i, 'timestamp_seconds': frames[i].timestamp})
            for evidence in copied_frame.evidence:
                evidence.update({'frame_id': i, 'timestamp_seconds': frames[i].timestamp})
            copied.append(copied_frame)
            frame_rows[i]['localization'] = local.to_dict()
            frame_rows[i]['copy_move'] = copied_frame.to_dict()
        for name, group in [('localization', localized), ('copy_move', copied)]:
            valid = [b for b in group if b.status == 'success' and b.score is not None]
            best = max(valid, key=lambda b: b.score) if valid else None
            any_failed = any(b.status == 'failed' for b in group)
            if name=='localization' and screening_error:
                any_failed=True
            snapshot_group=([coarse_localized[i] for i in sorted(coarse_localized) if i not in selected]+group) if name=='localization' else group
            aggregate = Branch(name, 'failed' if any_failed else 'success' if valid else 'insufficient_evidence',
                               None if any_failed else best.score if best else None,
                               model='TruFor all sampled coarse + selected detailed frames' if name == 'localization' else 'SIFT selected frames',
                               elapsed_ms=sum(b.elapsed_ms for b in group)+(screening_ms if name=='localization' else 0),
                               evidence=[e for b in snapshot_group for e in b.evidence],
                               details={**(best.details if best else {}), 'selected_frame_ids': selected,
                                        'frame_results': [b.to_dict() for b in snapshot_group],
                                        'localized_frame_ids': sorted(b.details['frame_id'] for b in snapshot_group),
                                        **({'coarse_screening':screening,'screening_ms':screening_ms} if name=='localization' else {}),
                                        'selection_rule': selection_rule},
                               reason=(screening_error or '至少一个必要帧的取证失败。') if any_failed else None)
            branches.append(aggregate)
        main = int(np.argmax(selection_scores))
        text_branch=text_future.result() if text_future else Branch('multimodal','failed',reason='场景编码失败；未伪造图文联合分析结果。')
        text_branch.details['frame_id']=main
        for evidence in text_branch.evidence:
            evidence['frame_id']=main
        branches.append(text_branch)
        branches.append(Branch('audio', 'not_applicable' if not metadata['audio_present'] else 'insufficient_evidence',
                               score_kind='not_a_fake_score', model='optional_audio',
                               reason='视频无音轨。' if not metadata['audio_present'] else '已检测音轨；1.1核心模式不将未分析的语音作为取证结论。'))
        preview_future.result()
        return branches, {'width': frames[0].image.width, 'height': frames[0].image.height, 'preview': frame_rows[main]['preview'],
                          'metadata': metadata, 'frames': frame_rows, 'windows': windows, 'decode_ms': decode_ms}

    def wave_branch(self, images, shared_native_logit=None):
        began=time.perf_counter()
        logits=self.models.wave_scores(images) if shared_native_logit is None else [float(shared_native_logit)]
        if len(logits) != len(images) or not np.isfinite(logits).all():
            raise ValueError('Invalid shared native WaveRep result')
        mean=float(np.mean(logits))
        score=float(1/(1+np.exp(-np.clip(mean,-50,50))))
        return Branch('forensic_generation','success',score,model='WaveRep DINOv2-B/14 G4',
                      elapsed_ms=round((time.perf_counter()-began)*1000,2),
                      evidence=[{'kind':'forensic_generation','text':'经频率取证增强训练的独立公开模型实际推理；沿用原生logit均值聚合。',
                                 'mean_logit':mean,'sampled_frame_logits':logits}],
                      details={'native_logit':mean,'frame_logits':logits,'aggregation':'mean_logits',
                               'score_kind':'uncalibrated_risk','preprocess':'official CenterCrop(504), ImageNet normalization',
                               'shared_encoding_with_image_origin':shared_native_logit is not None})

    def analyze(self, path: Path, out: Path, material_id, description='', received_at=None, material_type=None, original_name=None):
        started = received_at if received_at is not None else time.perf_counter()
        execute_start = time.perf_counter()
        out.mkdir(parents=True, exist_ok=True)
        kind = material_type or ('video' if path.suffix.lower() in ['.mp4', '.mov', '.mkv', '.webm', '.avi'] else 'image')
        try:
            branches, media = self.video(path, out, description) if kind == 'video' else self.image(path, out, description)
            for b in branches:
                for i, e in enumerate(b.evidence):
                    e['evidence_id'] = f'{material_id}:{b.branch_id}:{i:03d}'
            # Rebuild nested per-frame records after assigning evidence IDs.
            # Otherwise deep-copy snapshots of the same evidence lose IDs.
            if kind=='video':
                for name in ['localization','copy_move']:
                    aggregate=next((b for b in branches if b.branch_id==name),None)
                    if aggregate:
                        for row in media['frames']:
                            for nested in aggregate.details.get('frame_results',[]):
                                if nested.get('details',{}).get('frame_id')==row['frame_id']:
                                    nested['evidence']=[e for e in aggregate.evidence if e.get('frame_id')==row['frame_id']]
                                    row[name]=nested
            fusion = fuse(branches, self.config, kind)
            failed = [b.branch_id for b in branches if b.status == 'failed']
            result = {'schema_version': '1.1', 'version': self.config['version'], 'material_id': material_id,
                      'name': original_name or path.name, 'material_type': kind, 'sha256': sha256(path),
                      'bytes': path.stat().st_size, 'created_at': datetime.now(timezone.utc).isoformat(),
                      'execution_status': 'partial' if failed or not fusion.get('execution_complete',False) else 'complete', 'media': media,
                      'branches': [b.to_dict() for b in branches], 'fusion': fusion,
                      'explanation': human_explanation(branches, fusion),
                      'configuration_id': self.config['configuration_id'], 'configuration': self.config,
                      'model_manifest': self.models.info, 'business_risk': 'separate_review_required',
                      'runtime': {'device': self.models.device, 'gpu': self.models.info.get('runtime', {}).get('gpu')}}
        except Exception as exc:
            traceback.print_exc()
            result = {'schema_version': '1.1', 'version': self.config['version'], 'material_id': material_id,
                      'name': original_name or path.name, 'material_type': kind, 'sha256': sha256(path),
                      'execution_status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                      'branches': [], 'fusion': {'status': 'failed', 'risk': None, 'verdict': 'undetermined'},
                      'configuration_id': self.config['configuration_id']}
        result['timing'] = {'queue_ms': round((execute_start - started) * 1000, 2),
                            'analysis_ms': round((time.perf_counter() - execute_start) * 1000, 2),
                            'start_definition': 'complete_file_received_at_server',
                            'end_definition': 'required_evidence_and_initial_report_serialized_and_persisted',
                            'timing_footer_commit_excluded':True,'target_ms': 5000}
        # Two writes make the measured interval include real artifact/report I/O;
        # final JSON serialization is explicitly included rather than GPU-only.
        atomic_json(out / 'report.json', result)
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        result['timing']['total_ms'] = round((time.perf_counter() - started) * 1000, 2)
        result['timing']['within_target'] = result['timing']['total_ms'] <= 5000
        atomic_json(out / 'report.json', result)
        return result
