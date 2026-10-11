"""Regression cases where formatting or incomplete OCR must not imply a conflict."""
from types import SimpleNamespace
import numpy as np
from PIL import Image
from backend.multimodal import extract_fields, arithmetic_issues, analyze_multimodal


def test_equivalent_dates_amounts_and_identifiers():
    left = extract_fields('2026年1月2日 金额：￥1,000.00 发票号码：ab-1234')
    right = extract_fields('2026-01-02 金额：1000 发票号码：AB-1234')
    assert left == right
    assert extract_fields('2026-02-30') == {}
    assert 'amounts' not in extract_fields('金额：1,20.00')


def test_arithmetic_requires_a_complete_list():
    assert not arithmetic_issues('明细共3项 明细金额￥20 明细金额￥30 合计￥99')
    assert not arithmetic_issues('明细共2项 明细金额￥20 明细金额￥30 合计￥50')
    issue = arithmetic_issues('明细共2项 明细金额￥20 明细金额￥30 合计￥99')
    assert issue[0]['complete_items_sum'] == '50.00'


def test_low_confidence_ocr_is_visible_but_cannot_create_business_issue():
    polygon = [[0,0],[20,0],[20,10],[0,10]]
    models = SimpleNamespace(ocr=lambda image: ([[polygon, '2025-02-03 金额：999', .3]], [0,0,0]))
    result = analyze_multimodal(Image.new('RGB',(40,40)), np.full(7,.1), models, {'ocr_max_side':100}, '2026-01-02 金额：1000')
    assert result.details['ocr_boxes'] and not result.details['business_issues']
    assert result.details['comparison_fields'] == {}
    assert result.details['low_confidence_boxes_excluded_from_rules'] == 1


def test_confident_different_date_is_review_evidence_not_fake_score():
    polygon = [[0,0],[20,0],[20,10],[0,10]]
    models = SimpleNamespace(ocr=lambda image: ([[polygon, '2025-02-03', .95]], [0,0,0]))
    result = analyze_multimodal(Image.new('RGB',(40,40)), np.full(7,.1), models, {'ocr_max_side':100}, '2026-01-02')
    assert result.details['business_issues'][0]['field'] == 'dates'
    assert result.score is None and result.score_kind == 'not_a_fake_score'
