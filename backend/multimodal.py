from __future__ import annotations

import re
import time
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation
import numpy as np
from PIL import Image

from .contracts import Branch
from .models import SCENES


SCENE_LABELS = {'vehicle': '车辆', 'document': '单据', 'building': '房屋', 'person': '人物',
                'landscape': '户外', 'object': '物品', 'screen': '屏幕／图形'}


def extract_fields(text: str):
    text = unicodedata.normalize('NFKC', text)
    result = {}
    dates = re.findall(r'(?<!\d)(20\d{2})[-./年](\d{1,2})[-./月](\d{1,2})日?', text)
    if dates:
        valid_dates = []
        for parts in dates:
            try:
                valid_dates.append(date(*map(int, parts)).isoformat())
            except ValueError:
                continue
        if valid_dates:
            result['dates'] = sorted(set(valid_dates))
    amounts = re.findall(r'(?:[¥￥]|金额[:：]?\s*)(\d[\d,]*(?:\.\d{1,2})?)(?![\d.])', text)
    if amounts:
        normalized = [normalize_amount(value) for value in amounts]
        normalized = [value for value in normalized if value is not None]
        if normalized:
            result['amounts'] = sorted(set(normalized))
    identifiers = re.findall(r'(?i)(?:单号|发票号码|编号|invoice\s*(?:no\.?)?)[:：\s]*([A-Z0-9-]{4,30})', text)
    if identifiers:
        result['identifiers'] = sorted(set(value.upper() for value in identifiers))
    plates = re.findall(r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][A-Z0-9]{5,6}', text.upper())
    if plates:
        result['plates'] = plates
    return result


def normalize_amount(value):
    if not re.fullmatch(r'(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d{1,2})?', value):
        return None
    try:
        return format(Decimal(value.replace(',', '')).quantize(Decimal('.01')), 'f')
    except InvalidOperation:
        return None


def arithmetic_issues(text):
    """Check only explicitly counted, complete amount lists; partial OCR abstains."""
    text = unicodedata.normalize('NFKC', text)
    counts = re.findall(r'明细共\s*(\d+)\s*项', text)
    totals = re.findall(r'(?:合计|总计)[:：]?\s*[¥￥]?\s*(\d[\d,]*(?:\.\d{1,2})?)(?![\d.])', text)
    items = re.findall(r'明细金额[:：]?\s*[¥￥]?\s*(\d[\d,]*(?:\.\d{1,2})?)(?![\d.])', text)
    if len(counts) != 1 or len(totals) != 1 or int(counts[0]) != len(items) or not items:
        return []
    values = [normalize_amount(value) for value in [totals[0], *items]]
    if any(value is None for value in values):
        return []
    total, *amounts = map(Decimal, values)
    observed_sum = sum(amounts, Decimal('0'))
    if abs(total - observed_sum) <= Decimal('.01'):
        return []
    return [{'kind': 'amount_total_mismatch', 'reported_total': str(total), 'complete_items_sum': str(observed_sum),
             'item_count': len(items), 'text': '明确标示的完整明细与合计不一致；请核对OCR原文和金额含义。'}]


def analyze_multimodal(image: Image.Image, similarities, models, config, description='', timestamp=None):
    start = time.perf_counter()
    if models.ocr is None:
        return Branch('multimodal', 'failed', model='CLIP scene + RapidOCR', reason=models.errors.get('ocr', 'OCR unavailable'))
    width, height = image.size
    factor = min(1, config['ocr_max_side'] / max(image.size))
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    raw = np.asarray(image.resize(size)).copy()
    rows, ocr_stage_seconds = models.ocr(raw)
    boxes = []
    for polygon, text, recognition_confidence in rows or []:
        boxes.append({'text': text, 'recognition_confidence': float(recognition_confidence),
                      'polygon': (np.asarray(polygon) * np.asarray([width/size[0],height/size[1]])).tolist(),
                      'timestamp_seconds': timestamp, 'coordinate_space': 'original_raw_pixels'})
    text = '\n'.join(row['text'] for row in boxes)
    confidence_min = float(config.get('ocr_field_min_confidence', .85))
    trusted_boxes = [row for row in boxes if np.isfinite(row['recognition_confidence']) and row['recognition_confidence'] >= confidence_min]
    trusted_text = '\n'.join(row['text'] for row in trusted_boxes)
    best = np.argsort(similarities)[::-1][:3]
    scenes = [{'label': SCENES[i][0], 'name': SCENE_LABELS[SCENES[i][0]],
               'cosine_similarity': float(similarities[i]), 'meaning': '候选场景相似度，不是真伪概率'} for i in best]
    issues = arithmetic_issues(trusted_text)
    if description.strip():
        expected = extract_fields(description)
        observed = extract_fields(trusted_text)
        for key in ['dates', 'amounts', 'identifiers', 'plates']:
            if key in expected and key in observed and not set(expected[key]).intersection(observed[key]):
                issues.append({'kind': 'field_mismatch', 'field': key, 'expected': expected[key], 'observed': observed[key],
                               'text': '案件描述与OCR候选字段不一致；先核对识别文本和业务上下文。'})
        expected_scene = next((key for words, key in [('车辆|车损|汽车', 'vehicle'), ('发票|单据|收据', 'document'), ('房屋|房产|室内', 'building')]
                              if re.search(words, description)), None)
        scene_values = np.asarray(similarities, dtype=float)
        best_score = float(scene_values[best[0]])
        expected_index = next((i for i, value in enumerate(SCENES) if value[0] == expected_scene), None)
        confident_scene = (expected_index is not None and np.isfinite(scene_values).all()
                           and best_score >= config.get('scene_review_min_similarity', .26)
                           and best_score - scene_values[expected_index] >= config.get('scene_review_min_margin', .04))
        if expected_scene and expected_scene not in [s['label'] for s in scenes] and confident_scene:
            issues.append({'kind': 'scene_mismatch', 'expected': SCENE_LABELS[expected_scene],
                           'observed': scenes[0]['name'], 'text': '描述与候选场景不同，仅提示内容核验。'})
    return Branch('multimodal', 'success', score_kind='not_a_fake_score', model='CLIP scene + RapidOCR + explicit field rules',
                  elapsed_ms=round((time.perf_counter() - start) * 1000, 2),
                  evidence=[{'kind': 'ocr', 'text': row['text'], 'polygon': row['polygon'],
                             'recognition_confidence': row['recognition_confidence'], 'timestamp_seconds': timestamp,
                             'coordinate_space':'original_raw_pixels'} for row in boxes] + issues,
                  details={'scene_candidates': scenes, 'ocr_boxes': boxes, 'fields': extract_fields(text), 'text': text,
                           'comparison_fields': extract_fields(trusted_text), 'ocr_field_min_confidence': confidence_min,
                           'low_confidence_boxes_excluded_from_rules': len(boxes) - len(trusted_boxes),
                           'business_issues': issues, 'business_risk_separate': True,
                           'description_used': description, 'ocr_has_text': bool(boxes),
                           'ocr_stage_timings_ms': [float(v)*1000 for v in (ocr_stage_seconds or [])],
                           'ocr_stage_order':'detection, orientation, recognition'})
