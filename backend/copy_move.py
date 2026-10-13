from __future__ import annotations

import time
import cv2
import numpy as np
from PIL import Image

from .contracts import Branch


def detect_copy_move(image: Image.Image, max_side=1024, dense=False) -> Branch:
    start = time.perf_counter()
    width, height = image.size
    scale = min(1, max_side / max(width, height))
    gray = cv2.cvtColor(np.asarray(image.resize((max(1, round(width * scale)), max(1, round(height * scale))))), cv2.COLOR_RGB2GRAY)
    detector = cv2.SIFT_create(nfeatures=2000, contrastThreshold=.03, edgeThreshold=12)
    keys, descriptors = detector.detectAndCompute(gray, None)
    branch = Branch('copy_move', 'success', 0.0, model='SIFT + spatial exclusion + affine RANSAC',
                    details={'keypoints': len(keys), 'candidate_pairs': 0, 'inliers': 0,
                             'photometric_available': False,
                             'coordinate_space': 'original_raw_pixels', 'copy_direction': 'unknown',
                             'working_scale': scale,
                             'score_definition': 'uncalibrated geometric support, not a posterior'})
    if descriptors is None or len(keys) < 12:
        branch.status, branch.score = 'insufficient_evidence', None
        branch.reason = '纹理点不足，无法可靠验证复制粘贴。'
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = matcher.knnMatch(descriptors, descriptors, k=min(8, len(keys)))
        pairs = {}
        distance_min = max(24, min(gray.shape) * .07)
        for i, neighbours in enumerate(matches):
            eligible = [m for m in neighbours if m.trainIdx != i and np.linalg.norm(np.array(keys[i].pt) - keys[m.trainIdx].pt) > distance_min]
            if len(eligible) < 2 or eligible[0].distance >= .68 * eligible[1].distance:
                continue
            j = eligible[0].trainIdx
            # Canonical orientation eliminates symmetric duplicate matches.
            a, b = (i, j) if keys[i].pt < keys[j].pt else (j, i)
            # SIFT may emit several orientations at exactly the same position;
            # those are one geometric point, not independent corroboration.
            position_key = tuple(round(v, 1) for point in (keys[a].pt, keys[b].pt) for v in point)
            pairs[position_key] = (keys[a].pt, keys[b].pt)
        branch.details['candidate_pairs'] = len(pairs)
        if len(pairs) >= 6:
            source = np.float32([p[0] for p in pairs.values()])
            target = np.float32([p[1] for p in pairs.values()])
            matrix, mask = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                                    ransacReprojThreshold=3, maxIters=2000, confidence=.995)
            if matrix is not None and mask is not None:
                keep = mask.flatten().astype(bool)
                count = int(keep.sum())
                ratio = count / len(pairs)
                branch.details.update({'inliers': count, 'inlier_ratio': ratio, 'affine_matrix': matrix.tolist()})
                if count >= 6 and ratio >= .35:
                    points_a, points_b = source[keep] / scale, target[keep] / scale
                    spread = min(np.ptp(points_a[:, 0]), np.ptp(points_a[:, 1]), np.ptp(points_b[:, 0]), np.ptp(points_b[:, 1]))
                    if spread >= 12 / scale:
                        branch.score = round(min(.98, count / 24) * min(1, ratio / .7), 5)
                        def bounds(points):
                            lower = np.maximum(points.min(0) - 10 / scale, [0, 0])
                            upper = np.minimum(points.max(0) + 10 / scale, [width, height])
                            return [float(lower[0]), float(lower[1]), float(upper[0] - lower[0]), float(upper[1] - lower[1])]
                        bbox_a, bbox_b = bounds(points_a), bounds(points_b)
                        # Geometric keypoints only cover textured subregions.
                        # Refine with dense pixel agreement under the verified
                        # transform; never infer copy direction from this step.
                        rgb = np.asarray(image.resize((gray.shape[1], gray.shape[0])))
                        warped = cv2.warpAffine(rgb, matrix, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR)
                        valid = cv2.warpAffine(np.ones(gray.shape, np.uint8), matrix,
                                               (gray.shape[1], gray.shape[0]), flags=cv2.INTER_NEAREST) > 0
                        difference = np.max(np.abs(rgb.astype(np.int16)-warped.astype(np.int16)), axis=2)
                        # Pixel agreement around the geometric inliers distinguishes
                        # an actual copied patch from naturally repeated windows,
                        # text or lookalike objects. It remains learned evidence.
                        neighbourhood = np.zeros(gray.shape, np.uint8)
                        for point in target[keep]:
                            px, py = np.rint(point).astype(int)
                            cv2.circle(neighbourhood, (int(px), int(py)), 5, 1, -1)
                        supported = (neighbourhood > 0) & valid
                        if supported.any():
                            a = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[supported].astype(float)
                            b = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)[supported].astype(float)
                            a -= a.mean(); b -= b.mean()
                            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
                            branch.details.update(photometric_available=True,
                                photometric_agreement_ratio=float((difference[supported] <= 16).mean()),
                                photometric_normalized_error=float(difference[supported].mean() / 255),
                                photometric_correlation=float(np.dot(a, b) / denominator) if denominator > 1e-9 else 0.)
                        agreement = ((difference <= 8) & valid).astype(np.uint8)
                        _, labels, component_stats, _ = cv2.connectedComponentsWithStats(agreement, 8)
                        votes = {}
                        for point in target[keep]:
                            px, py = np.rint(point).astype(int)
                            if 0 <= py < labels.shape[0] and 0 <= px < labels.shape[1]:
                                label = int(labels[py, px])
                                if label: votes[label] = votes.get(label, 0) + 1
                        if votes:
                            selected = max(votes, key=votes.get)
                            x0,y0,rw,rh,area = component_stats[selected]
                            if area >= 400 and votes[selected] >= max(4,count*.25) and rw*rh <= gray.size*.35:
                                corners_b = np.float32([[x0,y0],[x0+rw,y0],[x0+rw,y0+rh],[x0,y0+rh]])
                                inverse = cv2.invertAffineTransform(matrix)
                                corners_a = cv2.transform(corners_b[None], inverse)[0]
                                def dense_bounds(points):
                                    lo=np.maximum(points.min(0)/scale,[0,0]);hi=np.minimum(points.max(0)/scale,[width,height])
                                    return [float(lo[0]),float(lo[1]),float(hi[0]-lo[0]),float(hi[1]-lo[1])]
                                bbox_a,bbox_b=dense_bounds(corners_a),dense_bounds(corners_b)
                                branch.details['dense_agreement_area']=int(area/scale**2)
                                branch.details['region_method']='verified affine + seeded dense RGB agreement'
                        branch.evidence.append({'kind': 'copy_pair', 'region_a': bbox_a, 'region_b': bbox_b,
                            'matches': [[*map(float, a), *map(float, b)] for a, b in list(zip(points_a, points_b))[:24]],
                            'direction': 'unknown', 'text': f'{count}组远距离特征通过同一仿射变换验证；相似纹理仍需人工复核。'})
    if dense:
        from .dense_copy import dense_copy
        parameters = dense if isinstance(dense, dict) else {}
        verified = dense_copy(image, max_side=max_side, **parameters)
        branch.details['dense_exact'] = verified
        branch.model += ' + dense pixel-verified translation candidates'
        if verified['verified_pair_count'] >= 4 and len(verified.get('regions', [])) == 2:
            # Sparse SIFT can be unavailable on small/flat copied patches.
            # Dense support remains a separate learned feature, not a verdict.
            if branch.status == 'insufficient_evidence':
                branch.status, branch.score = 'success', 0.
                branch.reason = '稀疏纹理点不足，另有逐像素验证的重复区域候选。'
            branch.evidence.append({'kind': 'copy_pair', 'region_a': verified['regions'][0],
                'region_b': verified['regions'][1], 'direction': 'unknown', 'dense_exact': True,
                'text': '两处区域存在逐像素一致的远距离重复块；重复文字或图案也可能出现此现象，不能单独证明欺诈。'})
    branch.elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return branch


def refine_with_localization(copy: Branch, localization: Branch, mask_path, original_size):
    """Cross-reference a strong localized edit with a separate verified pair.

    Geometry score remains independent. Region refinement explicitly depends on
    the localization mask and cannot be counted as another independent detector.
    """
    if copy.status!='success' or localization.status!='success' or not copy.evidence or (localization.score or 0)<.85:
        return copy
    binary=np.asarray(Image.open(mask_path))>0
    _,labels,stats,_=cv2.connectedComponentsWithStats(binary.astype(np.uint8),8)
    scale=copy.details.get('working_scale',1)
    matrix=np.asarray(copy.details.get('affine_matrix'))
    if matrix.shape!=(2,3):return copy
    pair=copy.evidence[0]
    points=pair.get('matches',[])
    votes={}
    for point in points:
        px,py=round(point[2]),round(point[3])
        if 0<=py<labels.shape[0] and 0<=px<labels.shape[1]:
            label=int(labels[py,px])
            if label:votes[label]=votes.get(label,0)+1
    if not votes:return copy
    label=max(votes,key=votes.get)
    x,y,w,h,area=stats[label]
    if votes[label]<max(4,len(points)*.25) or area>binary.size*.35:return copy
    b=np.float32([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
    a=cv2.transform((b*scale)[None],cv2.invertAffineTransform(matrix))[0]/scale
    width,height=original_size
    lower=np.maximum(a.min(0),[0,0]);upper=np.minimum(a.max(0),[width,height])
    pair['region_a']=[float(lower[0]),float(lower[1]),float(upper[0]-lower[0]),float(upper[1]-lower[1])]
    pair['region_b']=[int(x),int(y),int(w),int(h)]
    pair['localization_cross_reference']=True
    pair['text']+=' 候选区域结合独立TruFor可靠掩码细化，复制方向仍未知。'
    copy.details['region_method']='SIFT affine geometry + cross-referenced TruFor mask; score remains geometry-only'
    return copy
