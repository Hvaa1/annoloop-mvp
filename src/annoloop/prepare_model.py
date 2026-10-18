"""Explicit CLI equivalent of download + architecture recipe binding."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from .settings import Settings
from .model_hub import load_catalog
from .adapter_factory import select_and_prepare
from .connect_model import connect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_id');parser.add_argument('--data',type=Path)
    parser.add_argument('--offline',action='store_true',help='Chỉ nối lại model đã tải, không dùng mạng')
    parser.add_argument('--runtime-check',action='store_true',help='Kiểm tra import runtime bằng Python model, không chạy GPU/train')
    args=parser.parse_args();settings=Settings.load();data=args.data or settings.data
    model=next((m for m in load_catalog()['models'] if m['id']==args.model_id),None)
    if model is None:parser.error('ID không nằm trong kho model')
    if not args.offline:select_and_prepare(data,model)
    report=connect(data,model)
    if args.runtime_check and model['framework']=='mmdetection3d':
        from .mmdet_adapter import preflight
        try:report['runtime_check']=preflight(replace(settings,data=data))
        except (ValueError,OSError) as error:report['runtime_check']=dict(ready=False,error=str(error))
    report['note']='Binding không phải chứng nhận adaptation; kiểm chứng theo dataset trong app trước khi chạy vòng.'
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report.get('mmdet_recipe',{}).get('input_supported') else 2


if __name__=='__main__':raise SystemExit(main())
