from __future__ import annotations

import csv
from html import escape
import io
import json

VERDICTS = {'suspected_fake': '发现较强风险线索', 'lower_risk': '未检出显著生成／篡改证据',
            'undetermined': '证据不足，暂不作真伪判断'}


def csv_safe(value):
    text = str(value if value is not None else '')
    return "'" + text if text.startswith(('=', '+', '-', '@', '\t', '\r')) else text


def case_csv(case):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(['材料ID', '文件', '类型', '执行状态', '材料风险分', '判断', '完整耗时ms', '5秒内', 'SHA256'])
    for material in case['materials']:
        result = material.get('result', {})
        fusion, timing = result.get('fusion', {}), result.get('timing', {})
        writer.writerow([csv_safe(v) for v in [material['material_id'], material['name'], material['material_type'],
                          result.get('execution_status', material['status']), fusion.get('risk'), fusion.get('verdict'),
                          timing.get('api_total_ms',timing.get('total_ms')), timing.get('api_within_target',timing.get('within_target')), result.get('sha256')]])
    return '\ufeff' + stream.getvalue()


def case_html(case):
    rows = []
    versions=sorted({m.get('result',{}).get('version','未知版本') for m in case['materials']})
    version_label=' / '.join(versions)
    for material in case['materials']:
        result = material.get('result', {})
        f, t = result.get('fusion', {}), result.get('timing', {})
        evidence = ''.join('<li>' + escape(e['text']) + '</li>' for e in result.get('explanation', []))
        visual=''
        if case.get('case_id') and result.get('media',{}).get('preview'):
            asset_base=f'/api/cases/{case["case_id"]}/materials/{material["material_id"]}/assets/'
            preview=result['media']['preview']
            visual=f'<figure><img alt="原始坐标预览" src="{escape(asset_base+preview,quote=True)}"><figcaption>原始像素坐标预览；完整原件另行保存。</figcaption></figure>'
        branch_rows=[];details=[]
        for branch in result.get('branches',[]):
            score=branch.get('score')
            branch_rows.append('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in [branch['branch_id'],branch['status'],'—' if score is None else f'{score:.4f}',branch.get('model','')])+'</tr>')
            for item in branch.get('evidence',[]):
                if item.get('kind') not in ['localization','copy_pair','temporal_window']:continue
                summary={k:v for k,v in item.items() if k in ['evidence_id','frame_id','timestamp_seconds','regions','region_a','region_b','direction','start_seconds','end_seconds','actual_timestamps','valid_mask','temporal_risk','coordinate_space']}
                details.append('<pre>'+escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>')
                if case.get('case_id') and item.get('mask'):
                    details.append(f'<img class="mask" alt="预测二值掩码" src="{escape(asset_base+item["mask"],quote=True)}">')
        table='<table><thead><tr><th>分支</th><th>状态</th><th>原始风险分</th><th>方法</th></tr></thead><tbody>'+''.join(branch_rows)+'</tbody></table>' if branch_rows else ''
        timing=t.get('api_total_ms',t.get('total_ms','—'))
        verdict = VERDICTS.get(f.get('verdict'), VERDICTS['undetermined'])
        head_details = ''
        if f.get('head_probabilities'):
            head_details = '<p>独立参考集风险：' + ' · '.join(
                f'{label} {float(f["head_probabilities"][key]):.1%}'
                for key, label in [('generated_origin', '整图生成'), ('local_edit', '局部编辑')]) + '。两条分支分别校准后合并，不能解释为保险欺诈概率。</p>'
        limit = '<p><strong>未检出不能证明真实。新生成器、截图和压缩可能造成漏检，请结合原始文件与来源记录复核。</strong></p>' if f.get('verdict') == 'lower_risk' else ''
        disagreement = '<p><strong>模型之间存在明显分歧，建议优先人工复核。</strong></p>' if f.get('review_recommended') else ''
        rows.append(f'<section><h2>{escape(material["name"])}</h2><p>状态：{escape(result.get("execution_status", material["status"]))} · 判断：{escape(verdict)} · 风险分：{f.get("risk", "未形成")} · 含排队及落盘耗时：{timing} ms</p>{limit}{disagreement}{head_details}{visual}{table}<ul>{evidence}</ul>{"".join(details)}<p class="hash">SHA256: {escape(result.get("sha256", ""))}</p><p class="hash">配置：{escape(result.get("configuration_id","—"))} · 实际阈值：{f.get("threshold","未提供")}</p></section>')
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>澄鉴 · 材料核验报告</title>
<style>body{{font:15px/1.8 -apple-system,BlinkMacSystemFont,"Microsoft YaHei",sans-serif;background:#f4f6fa;color:#1d2430;margin:0;padding:48px}}main{{max-width:900px;margin:auto}}h1{{font-size:38px}}section{{background:white;border:1px solid #dce1e9;border-radius:18px;padding:24px;margin:20px 0}}.hash,pre{{font:11px/1.5 monospace;overflow-wrap:anywhere;white-space:pre-wrap}}small,figcaption{{color:#626d7f}}figure{{margin:16px 0}}img{{max-width:100%;max-height:400px;object-fit:contain}}.mask{{max-width:320px}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{border-bottom:1px solid #dce1e9;padding:8px;text-align:left}}@media print{{body{{padding:0;background:white}}figure,table,pre{{break-inside:avoid}}section{{border:none}}}}</style>
<main><small>澄鉴 / CLEARTRACE · 本地取证工作台 {escape(version_label)}</small><h1>{escape(case['title'])}</h1><p>材料风险分来自实际取证输出，尚未完成目标业务域概率校准。低风险不证明真实，材料风险不等同于业务欺诈概率。</p>{''.join(rows)}<small>配置与逐条证据见同案JSON报告。此HTML可使用浏览器打印成PDF。</small></main></html>'''
