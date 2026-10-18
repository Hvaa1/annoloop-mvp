"""Readable catalog names and a plain usable / needs-download / unusable verdict.

The verdict only sorts the selection list. It never grants adaptation capability:
preparation still connects the recipe, strict-loads weights and verifies on GPU.
"""
import re

# Families whose head already has a train / signed-ignore / calibration recipe.
RECIPE_FAMILIES = {'CenterPoint': 'CenterHead', 'PointPillars': 'Anchor3DHead', 'SECOND': 'Anchor3DHead'}


def _metres(digits):
    # OpenMMLab names drop the decimal point: 01 -> 0.1, 0075 -> 0.075, 02 -> 0.2.
    return '0.' + digits[1:] if digits.startswith('0') else digits


TOKENS = [(r'voxel(\d+)', lambda m: f'voxel {_metres(m.group(1))} m'),
          (r'pillar(\d+)', lambda m: f'pillar {_metres(m.group(1))} m'),
          (r'regnet-([0-9.]+[gm]f)', lambda m: 'RegNet-' + m.group(1).upper()),
          (r'(?<![a-z])secfpn', lambda m: 'SECFPN'), (r'(?<![a-z])fpn', lambda m: 'FPN'),
          (r'(?<![a-z])dcn', lambda m: 'DCN'), (r'strong-aug', lambda m: 'strong aug'),
          (r'free-anchor', lambda m: 'FreeAnchor'), (r'circlenms', lambda m: 'Circle-NMS')]


def details(model):
    name = model.get('name', '').lower().replace('_', '-')
    parts = []
    for pattern, render in TOKENS:
        found = re.search(pattern, name)
        if found:
            name = name[:found.start()] + ' ' + name[found.end():]
            text = render(found)
            if text not in parts:
                parts.append(text)
    return parts


def release_year(model):
    found = re.search(r'_(20\d\d)\d{4}_', model.get('weights_url', ''))
    return found.group(1) if found else None


def friendly(model):
    """e.g. 'PointPillars · SECFPN · nuScenes · 2021'."""
    family = model.get('family', model.get('architecture', 'Model'))
    parts = [family, *(d for d in details(model) if d.lower() != family.lower()), model.get('dataset', '')]
    year = release_year(model)
    return ' · '.join(p for p in parts + ([year] if year else []) if p)


def verdict(model, report, downloaded):
    """('usable'|'download'|'unusable', reason). Uses only catalog/cache evidence."""
    from .mmdet_recipe import legacy_blocker
    if report['missing_classes']:
        return 'unusable', 'Thiếu class: ' + ', '.join(report['missing_classes'])
    if not report['selectable']:
        return 'unusable', ' · '.join(dict.fromkeys(report['selection_blockers'] + report['reasons']))
    if model.get('framework') != 'mmdetection3d':
        return 'unusable', model.get('framework', 'Framework') + ' chưa có recipe train / ignore / calibration'
    legacy = legacy_blocker(model)
    if legacy:
        return 'unusable', legacy[0]
    if report['eligible']:
        return 'usable', 'Recipe đã nối, đầu vào khớp; còn kiểm tra GPU ở bước 3'
    recipe = model.get('mmdet_recipe')
    if recipe:  # downloaded and read: the real head is known
        return 'unusable', ' · '.join(recipe.get('blockers') or report['reasons'])
    family = model.get('family', model.get('architecture'))
    if family in RECIPE_FAMILIES and not downloaded:
        return 'download', 'Cần tải; head dự kiến ' + RECIPE_FAMILIES[family] + ' — kiểm tra lại sau tải'
    return 'unusable', 'Head ' + str(family) + ' chưa có recipe train / ignore / calibration'
