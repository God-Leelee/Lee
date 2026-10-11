from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import av
import numpy as np
from PIL import Image, ExifTags
from pillow_heif import register_heif_opener, open_heif

register_heif_opener(thumbnails=False, decode_threads=2)


class MediaError(ValueError):
    pass


@dataclass
class VideoFrame:
    frame_id: int
    timestamp: float
    pts: int
    time_base: str
    image: Image.Image


def safe_value(value):
    if isinstance(value, bytes):
        return value[:512].decode('utf-8', errors='replace')
    if isinstance(value, dict):
        return {str(k): safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_value(v) for v in value[:128]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value[:8000] if isinstance(value, str) else value
    return str(value)[:8000]


def read_image(path: Path, max_pixels: int):
    container_info = None
    container_exif = None
    if path.suffix.lower() in {'.heic', '.heif'}:
        heif = open_heif(path)
        if len(heif) != 1:
            raise MediaError('多图HEIF暂不支持完整核验；请逐张导出或提交视频。')
        if heif.size[0] * heif.size[1] > max_pixels:
            raise MediaError('HEIF图片超过像素输入上限。')
        container_info = safe_value(heif.info)
        if heif.info.get('exif'):
            container_exif = Image.Exif(); container_exif.load(heif.info['exif'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(path) as original:
            if original.width * original.height > max_pixels:
                raise MediaError(f'图片超过当前原型的 {max_pixels:,} 像素输入上限。')
            if getattr(original, 'n_frames', 1) > 1:
                raise MediaError('动态图像请转成视频；当前不会只取第一帧并当作完整检测。')
            exif = container_exif if container_exif is not None else original.getexif()
            nested_exif = {}
            for tag, name, names in [(34665, 'ExifIFD', ExifTags.TAGS), (34853, 'GPSIFD', ExifTags.GPSTAGS)]:
                if tag in exif:
                    try:
                        nested_exif[name] = {names.get(k, str(k)): safe_value(v) for k,v in exif.get_ifd(tag).items()}
                    except Exception as exc:
                        nested_exif[name] = {'parse_error': str(exc)}
            metadata = {'format': original.format, 'width': original.width, 'height': original.height,
                        'mode': original.mode, 'info': safe_value(original.info),
                        'exif': {ExifTags.TAGS.get(k, str(k)): safe_value(v) for k, v in exif.items()},
                        'exif_sub_ifds': nested_exif,
                        'coordinate_space': 'original_raw_pixels',
                        'orientation_policy': '预览和模型均使用原始像素坐标；EXIF方向单独列出，不隐式旋转。'}
            if container_info is not None:
                metadata.update({'original_container_info': container_info, 'coordinate_space':'decoded_image_pixels',
                  'decoder':'pillow-heif / libheif', 'orientation_policy':'HEIF解码器可应用容器方向；区域按解码预览坐标表示，原始容器EXIF单独保留。',
                  'hdr_policy':'模型使用解码器转换的8-bit RGB；高位深取证不在当前验收范围。'})
            original.load()
            image = original.convert('RGB')
    return image, metadata


def metadata_signals(metadata: dict):
    """Weak provenance clues; never used as a visual classifier input."""
    found = []
    fields = {**metadata.get('info', {}), **metadata.get('exif', {}), **metadata.get('tags', {})}
    for i, track in enumerate(metadata.get('tracks', [])):
        fields.update({f'tracks[{i}].{k}': v for k,v in track.get('tags', {}).items()})
    for ifd, values in metadata.get('exif_sub_ifds', {}).items():
        fields.update({f'{ifd}.{k}': v for k,v in values.items()})
    for name, value in fields.items():
        text = str(value)
        if re.search(r'stable.?diffusion|comfyui|automatic1111|midjourney|dall.?e|flux|novelai|civitai', text, re.I):
            found.append({'kind': 'generation_metadata', 'field': name, 'value': text[:1200],
                          'text': '原字段包含生成工具相关标记；字段可被删除或改写，仅作来源线索。'})
        elif name.rsplit('.',1)[-1].lower() in ['software', 'encoder', 'creator_tool'] and re.search(r'photoshop|gimp|lightroom|lavf|premiere', text, re.I):
            found.append({'kind': 'editing_metadata', 'field': name, 'value': text[:1200],
                          'text': '可见编辑或转码软件字段；正常后处理也可能留下此字段。'})
    return found


def video_metadata(container, stream):
    duration = float(stream.duration * stream.time_base) if stream.duration is not None else (container.duration / 1_000_000 if container.duration else None)
    tracks = []
    for track in container.streams:
        context = track.codec_context
        tracks.append({'type': track.type, 'codec': context.name if context is not None else None, 'time_base': str(track.time_base),
                       'frames_declared': track.frames, 'tags': safe_value(track.metadata),
                       'width': getattr(context, 'width', None), 'height': getattr(context, 'height', None),
                       'sample_rate': getattr(context, 'sample_rate', None)})
    return {'format': container.format.name, 'width': stream.codec_context.width,
            'height': stream.codec_context.height, 'duration_seconds': duration,
            'average_rate_declared': str(stream.average_rate), 'tracks': tracks,
            'tags': safe_value(container.metadata), 'coordinate_space': 'decoded_frame_raw_pixels',
            'audio_present': any(t.type == 'audio' for t in container.streams),
            'decoder': 'PyAV / FFmpeg', 'pyav_version': av.__version__}


def sample_video(path: Path, config: dict):
    """PTS based sampling. Seeking avoids decoding an entire long clip.

    Eight window targets refer to a continuous span; global samples cannot be
    substituted for those targets. Duplicate decoded PTS receive a false mask.
    Every decoded frame, including discarded keyframe preroll, is counted.
    """
    acceleration, acceleration_error, source_pixel_format = None, None, None
    seek_reuse_seconds=1.0
    if config.get('hardware_decode_long_h264'):
        with av.open(str(path)) as probe:
            if len(probe.streams.video)==1:
                candidate=probe.streams.video[0]
                duration=float(candidate.duration*candidate.time_base) if candidate.duration else (probe.duration or 0)/1e6
                context=candidate.codec_context
                source_pixel_format=context.format.name if context.format else None
                if duration>=10:
                    key_times=[float(packet.pts*packet.time_base) for packet in probe.demux(candidate)
                               if packet.is_keyframe and packet.pts is not None]
                    if len(key_times)>1:
                        seek_reuse_seconds=max(1.0,float(np.median(np.diff(sorted(key_times)))))
                if context.name=='h264' and context.width*context.height>=1280*720 and duration>=10 and source_pixel_format=='yuv420p':
                    try:
                        from av.codec.hwaccel import HWAccel
                        acceleration=HWAccel(device_type='cuda',allow_software_fallback=False)
                    except Exception as exc:
                        acceleration_error=str(exc)
    try:
        container=av.open(str(path),hwaccel=acceleration) if acceleration else av.open(str(path))
    except Exception as exc:
        acceleration_error=str(exc)
        acceleration=None
        container=av.open(str(path))
    with container:
        streams = [s for s in container.streams if s.type == 'video']
        if len(streams) != 1:
            raise MediaError('当前支持单视频轨输入；未选择性忽略其他视频轨。')
        stream = streams[0]
        stream.thread_type = 'AUTO'
        stream.codec_context.thread_count = 4
        metadata = video_metadata(container, stream)
        duration = metadata['duration_seconds']
        if duration is None or duration <= 0:
            raise MediaError('容器缺少可用时长，无法保证完整时间覆盖；请先规范化容器。')
        if duration > config.get('max_video_seconds', 120):
            raise MediaError(f'视频超过当前原型的 {config.get("max_video_seconds", 120)} 秒输入上限。')
        if metadata['width'] * metadata['height'] > config.get('video_max_pixels', 3840 * 2160):
            raise MediaError('视频帧尺寸超过当前原型输入上限。')
        frame_period = 1 / float(stream.average_rate) if stream.average_rate else .04
        last = max(0, duration - max(frame_period, .001))
        global_targets = np.linspace(0, last, config['global_frames']).tolist()
        effective_step = max(config['window_step_seconds'], frame_period)
        span = min(last, (config['window_frames'] - 1) * effective_step)
        windows = []
        for center in np.linspace(.25, .75, config['continuous_windows']):
            start = min(max(0, duration * float(center) - span / 2), max(0, last - span))
            windows.append(np.linspace(start, start + span, config['window_frames']).tolist())
        targets = sorted(set(round(t, 7) for t in global_targets + [t for w in windows for t in w]))
        start_time = float((stream.start_time or 0) * stream.time_base)
        collected, target_map, decoded = {}, {}, 0
        iterator, candidate, previous_target = None, None, -float('inf')
        for target in targets:
            absolute = target + start_time
            # Consecutive targets share the decoder. Re-seeking for every frame
            # would repeatedly decode the same GOP and exceed the time budget.
            if iterator is None or target - previous_target > seek_reuse_seconds:
                container.seek(max(stream.start_time or 0, int(absolute / float(stream.time_base))), stream=stream, backward=True)
                iterator, candidate = iter(container.decode(stream)), None
            previous_target = target
            if candidate is None or float(candidate.pts * candidate.time_base) - start_time + 1e-7 < target:
                for frame in iterator:
                    decoded += 1
                    if frame.pts is None:
                        continue
                    actual = float(frame.pts * frame.time_base) - start_time
                    candidate = frame
                    if actual + 1e-7 >= target:
                        break
            if candidate is None:
                raise MediaError(f'无法解码 {target:.3f}s 附近的帧。')
            actual = max(0.0, float(candidate.pts * candidate.time_base) - start_time)
            if candidate.pts not in collected:
                display_frame=candidate
                if acceleration:
                    # NVDEC yields NV12 and PyAV16 loses color properties while
                    # downloading. Restore the original planar format and tags
                    # before RGB conversion; verified against software pixels.
                    display_frame=candidate.reformat(format=source_pixel_format)
                    display_frame.colorspace=stream.codec_context.colorspace
                    display_frame.color_range=stream.codec_context.color_range
                collected[candidate.pts] = VideoFrame(0, actual, candidate.pts, str(candidate.time_base), display_frame.to_image().convert('RGB'))
            target_map[round(target, 7)] = candidate.pts
        frames = sorted(collected.values(), key=lambda f: f.timestamp)
        if len(frames) > config['max_visual_frames']:
            raise MediaError('采样预算不一致；不会静默截断连续窗口。')
        id_by_pts = {}
        for index, frame in enumerate(frames):
            frame.frame_id = index
            id_by_pts[frame.pts] = index
        window_data = []
        for index, targets_in_window in enumerate(windows):
            ids = [id_by_pts[target_map[round(t, 7)]] for t in targets_in_window]
            seen, mask = set(), []
            for frame_id in ids:
                mask.append(frame_id not in seen)
                seen.add(frame_id)
            window_data.append({'window_id': index, 'frame_ids': ids, 'valid_mask': mask,
                                'nominal_step_seconds': config['window_step_seconds'], 'effective_step_seconds': effective_step,
                                'target_timestamps': targets_in_window,
                                'actual_timestamps': [frames[i].timestamp for i in ids],
                                'start_seconds': min(frames[i].timestamp for i in ids),
                                'end_seconds': max(frames[i].timestamp for i in ids)})
        metadata.update({'frames_decoded_including_preroll': decoded, 'unique_visual_frames': len(frames),
                         'hardware_decode_active': bool(acceleration), 'hardware_decode_fallback_reason': acceleration_error,
                         'decoder_reuse_gap_seconds':seek_reuse_seconds,
                         'global_frame_ids': sorted({id_by_pts[target_map[round(t,7)]] for t in global_targets}),
                         'sampling_method': 'PTS global coverage + two continuous temporal windows',
                         'sampled_duration_seconds': last, 'start_time_seconds': start_time,
                         'coverage_note': '稀疏采样不能排除未采到的瞬时局部编辑；未采样间隔不作为定位区间。'})
        return frames, window_data, metadata


def d3_statistics(features: np.ndarray, valid_mask: list[bool], timestamps: list[float]):
    keep = np.asarray(valid_mask, bool)
    x, times = features[keep], np.asarray(timestamps)[keep]
    if len(x) < 4 or np.any(np.diff(times) <= 0):
        return {'status': 'insufficient_evidence', 'reason': '至少需要4个不同PTS的有效帧。'}
    first = np.linalg.norm(np.diff(x, axis=0), axis=-1)
    second = np.diff(first)
    return {'status': 'success', 'first_order_l2': first.tolist(), 'second_order': second.tolist(),
            'second_order_std': float(np.std(second, ddof=1)), 'valid_frames': int(len(x)),
            'delta_t_seconds': np.diff(times).tolist(),
            'method': 'D3-inspired shared CLIP-L/14 adaptation; raw statistic, not fake probability'}
