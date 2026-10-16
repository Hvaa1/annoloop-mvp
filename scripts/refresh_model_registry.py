"""Build a pinned 3D detector registry from primary model zoos; probe real bytes.

Maintainer tool, not run on app startup. Never executes upstream configs/pickles.
Only binary-reachable checkpoints with explicit class metadata are published.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import sys
import urllib.request
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'src/annoloop/catalog/models.json'
FAMILIES=['3dssd','centerpoint','dynamic_voxelization','fcaf3d','free_anchor','groupfree3d',
          'h3dnet','parta2','point_rcnn','pointpillars','pv_rcnn','regnet','second','ssn','votenet']
HEADERS={'User-Agent':'AnnoLoop-model-registry/1.0'}


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers=HEADERS),timeout=25) as r:
        return r.read()


def literal(node, env):
    if isinstance(node,ast.Name):return env[node.id]
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict' and not node.args:
        return {k.arg:literal(k.value,env) for k in node.keywords if k.arg is not None}
    if isinstance(node,(ast.List,ast.Tuple)):return [literal(x,env) for x in node.elts]
    if isinstance(node,ast.Dict):return {literal(k,env):literal(v,env) for k,v in zip(node.keys,node.values)}
    return ast.literal_eval(node)


def probe(url):
    try:
        req=urllib.request.Request(url,headers={**HEADERS,'Range':'bytes=0-4095'})
        with urllib.request.urlopen(req,timeout=25) as r:
            body=r.read(4096)
            if not (body.startswith(b'PK\x03\x04') or body.startswith(b'\x80\x02') or body.startswith(b'\x80\x03') or body.startswith(b'\x80\x04')):
                raise ValueError('Response is not a Torch checkpoint binary (HTML/login/denied)')
            content_range=r.headers.get('Content-Range','')
            size=int(content_range.rsplit('/',1)[1]) if '/' in content_range and content_range.rsplit('/',1)[1].isdigit() else int(r.headers.get('Content-Length',0)) if r.status==200 else None
            return dict(status='binary_reachable',checked_at=datetime.now(timezone.utc).isoformat(),
                http_status=r.status,bytes=size,etag=r.headers.get('ETag'),last_modified=r.headers.get('Last-Modified'),
                prefix_sha256=hashlib.sha256(body).hexdigest(),prefix_bytes=len(body))
    except Exception as e:return dict(status='unavailable',error=f'{type(e).__name__}: {e}')


def main():
    tree=json.loads(fetch('https://api.github.com/repos/open-mmlab/mmdetection3d/git/trees/main?recursive=1'))
    commit=tree['sha'];base=f'https://raw.githubusercontent.com/open-mmlab/mmdetection3d/{commit}/'
    @lru_cache(maxsize=None)
    def get(path):return fetch(base+path).decode()
    @lru_cache(maxsize=None)
    def config(path):
        if path.startswith('../') or not path.startswith('configs/'):raise ValueError('Config outside repository configs')
        text=get(path); env={};bases=[]
        for node in ast.parse(text).body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                key=node.targets[0].id
                try:env[key]=literal(node.value,env)
                except (ValueError,KeyError,TypeError,SyntaxError):continue
        base_paths=env.get('_base_',[])
        if isinstance(base_paths,str):base_paths=[base_paths]
        classes=None;files={path:dict(url=base+path,sha256=hashlib.sha256(text.encode()).hexdigest())}
        for b in base_paths:
            child=posixpath.normpath(posixpath.join(posixpath.dirname(path),b))
            inherited,more=config(child);files.update(more)
            if inherited:classes=inherited
        if isinstance(env.get('class_names'),list):classes=env['class_names']
        if isinstance(env.get('metainfo'),dict) and env['metainfo'].get('classes'):classes=env['metainfo']['classes']
        return classes,files
    candidates=[];excluded=[]
    for family in FAMILIES:
        path=f'configs/{family}/metafile.yml'
        try:meta=yaml.safe_load(get(path))
        except Exception as e:excluded.append(dict(source=path,error=str(e)));continue
        for entry in meta.get('Models',[]):
            try:
                results=entry.get('Results',[])
                result=next(r for r in results if r.get('Task')=='3D Object Detection')
                url=entry['Weights'];classes,files=config(entry['Config'])
                if not classes or not all(isinstance(c,str) for c in classes):raise ValueError('No explicit resolved class list')
                if not url.startswith('https://download.openmmlab.com/'):raise ValueError('Unexpected weight host')
                name=entry['Name'];identifier='mmdet3d-'+hashlib.sha256((name+'|'+url).encode()).hexdigest()[:16]
                metadata={**entry.get('Metadata',{}),**entry.get('metadata',{})}
                candidates.append(dict(id=identifier,name=name,architecture=entry.get('In Collection',family),
                    framework='mmdetection3d',dataset=result['Dataset'],classes=list(classes),data_type='3d',modality='point_cloud',
                    weights_url=url,config_path=entry['Config'],config_files=files,
                    source_url=f'https://github.com/open-mmlab/mmdetection3d/blob/{commit}/{path}',
                    repository='https://github.com/open-mmlab/mmdetection3d',revision=commit,
                    upstream_metrics=result.get('Metrics',{}),training_memory_gb=metadata.get('Training Memory (GB)'),
                    runtime_status='adapter_required',runtime_reason='MVP native adaptation hiện chỉ hỗ trợ ST3D SECOND-IoU/PV-RCNN; gói MMDetection3D cần adapter tương ứng.',
                    runtime_verified=False,publisher_sha256=None,
                    license_note='Repository Apache-2.0; weights/dataset terms must be checked at upstream before redistribution.',
                    classes_evidence='Explicit class_names/metainfo resolved statically from pinned config and bases'))
            except Exception as e:excluded.append(dict(source=path,name=entry.get('Name'),error=str(e)))
        print(family,'candidates',len(candidates),flush=True)
    # Same weights with multiple configs do not inflate checkpoint count.
    unique={}
    for c in candidates:unique.setdefault(c['weights_url'],c)
    candidates=list(unique.values())
    # Official OpenPCDet zoo includes VoxelNeXt / Voxel R-CNN / PV-RCNN++.
    # Google Drive /view pages are not downloads: probe the actual binary endpoint.
    try:
        pc=json.loads(fetch('https://api.github.com/repos/open-mmlab/OpenPCDet/commits/master'))['sha']
        pcbase=f'https://raw.githubusercontent.com/open-mmlab/OpenPCDet/{pc}/'
        readme=fetch(pcbase+'README.md').decode()
        @lru_cache(maxsize=None)
        def pcconfig(path):
            if not path.startswith('tools/cfgs/') or '..' in PurePosixPath(path).parts:raise ValueError('Invalid OpenPCDet config path')
            text=fetch(pcbase+path).decode();doc=yaml.safe_load(text)
            files={path:dict(url=pcbase+path,sha256=hashlib.sha256(text.encode()).hexdigest())}
            def bases(obj):
                if isinstance(obj,dict):
                    for k,v in obj.items():
                        if k=='_BASE_CONFIG_' and isinstance(v,str):
                            target=posixpath.normpath(posixpath.join('tools',v)) if not v.startswith('tools/') else v
                            _,more=pcconfig(target);files.update(more)
                        else:bases(v)
                elif isinstance(obj,list):
                    for v in obj:bases(v)
            bases(doc);return doc,files
        for line in readme.splitlines():
            if not line.startswith('|'):continue
            links=re.findall(r'\[([^\]]+)\]\(([^)]+)\)',line)
            configs=[u for _,u in links if u.startswith('tools/cfgs/') and u.endswith('.yaml')]
            weights=[u for label,u in links if 'drive.google.com' in u and 'model' in label.lower()]
            if not configs or not weights:continue
            for original in weights:
                try:
                    fileid=re.search(r'/d/([A-Za-z0-9_-]+)',original)
                    if not fileid:raise ValueError('Unsupported Drive link')
                    path=configs[0];doc,files=pcconfig(path);classes=doc['CLASS_NAMES']
                    dataset='nuScenes' if '/nuscenes_models/' in path else 'Waymo' if '/waymo_models/' in path else 'KITTI' if '/kitti_models/' in path else None
                    if not dataset:continue
                    name=PurePosixPath(path).stem
                    modality='camera' if doc['MODEL']['NAME']=='CaDDN' else 'lidar_camera' if doc['MODEL']['NAME']=='BevFusion' or 'multimodal' in path or 'images' in doc.get('DATA_CONFIG',{}).get('GET_ITEM_LIST',[]) else 'point_cloud'
                    url='https://drive.usercontent.google.com/download?id='+fileid[1]+'&export=download&confirm=t'
                    candidates.append(dict(id='openpcdet-'+hashlib.sha256((path+fileid[1]).encode()).hexdigest()[:16],
                        name=name,architecture=doc['MODEL']['NAME'],framework='OpenPCDet',dataset=dataset,
                        classes=classes,data_type='3d',modality=modality,weights_url=url,config_path=path,config_files=files,
                        source_url=f'https://github.com/open-mmlab/OpenPCDet/blob/{pc}/README.md',
                        original_weights_url=original,repository='https://github.com/open-mmlab/OpenPCDet',revision=pc,
                        upstream_metrics={},training_memory_gb=None,runtime_status='adapter_required',
                        runtime_reason='Checkpoint OpenPCDet cần kiểm tra config/head/features và adapter; không đồng nghĩa tương thích ST3D runtime.',
                        runtime_verified=False,publisher_sha256=None,
                        license_note='Repository Apache-2.0; upstream weights/dataset terms apply.',
                        classes_evidence='Explicit CLASS_NAMES in pinned official model config'))
                except Exception as e:excluded.append(dict(source='OpenPCDet',name=configs[0],error=str(e)))
        print('OpenPCDet total candidates',len(candidates),flush=True)
    except Exception as e:excluded.append(dict(source='OpenPCDet',error=str(e)))
    unique={}
    for c in candidates:unique.setdefault(c['weights_url'],c)
    candidates=list(unique.values())
    with ThreadPoolExecutor(max_workers=6) as pool:
        checks=list(pool.map(lambda c:probe(c['weights_url']),candidates))
    accepted=[]
    for c,check in zip(candidates,checks):
        if check['status']=='binary_reachable':accepted.append(dict(c,download_verification=check))
        else:excluded.append(dict(name=c['name'],url=c['weights_url'],**check))
    family_names={'PointPillar':'PointPillars','SECONDNet':'SECOND','PVRCNN':'PV-RCNN','PartA2Net':'Part-A^2','VoxelRCNN':'Voxel R-CNN','BevFusion':'BEVFusion','SECONDNetIoU':'SECOND-IoU'}
    for m in accepted:m['family']=family_names.get(m['architecture'],m['architecture'])
    OUT.parent.mkdir(parents=True,exist_ok=True)
    temporary=OUT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(dict(schema_version=1,generated_at=datetime.now(timezone.utc).isoformat(),
        models=accepted,source_revision=commit,policy='Only verified binary-reachable direct checkpoints, explicit config-derived classes; availability is a timestamped check, not permanent guarantee'),ensure_ascii=False,indent=2)+'\n')
    temporary.replace(OUT)
    report=ROOT/'data/model_registry_refresh';report.mkdir(parents=True,exist_ok=True)
    (report/'excluded.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2)+'\n')
    print('PUBLISHED',len(accepted),'EXCLUDED',len(excluded),flush=True)


if __name__=='__main__':main()
