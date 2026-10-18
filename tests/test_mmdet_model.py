"""Training configuration must survive construction, before head installation."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace,ModuleType
from unittest.mock import patch,Mock
import unittest
from annoloop.mmdet_model import build_model
from annoloop.mmdet_ignore import install


class Config(dict):
    @property
    def model(self):return self['model']


class ModelTests(unittest.TestCase):
    def test_downloaded_freeanchor_catalog_does_not_bind_regnet_anchor_config(self):
        path=Path(__file__).resolve().parents[1]/'src/annoloop/catalog/models.json'
        rows=json.loads(path.read_text())['models']
        row=next(r for r in rows if r['id']=='mmdet3d-a4f12c990b6cc4b8')
        self.assertEqual(row['architecture'],'FreeAnchor')
        self.assertEqual(row['config_path'],'configs/free_anchor/pointpillars_hv_fpn_head-free-anchor_sbn-all_8xb4-2x_nus-3d.py')
        self.assertIn('free-anchor',row['weights_url'])
        self.assertIn(row['config_path'],row['config_files'])

    def runtime(self):
        model=SimpleNamespace(to=Mock(),eval=Mock())
        def convert(config):config['norm_cfg']['type']='BN'
        build=Mock(return_value=model);scope=Mock()
        mods={name:ModuleType(name) for name in ('mmengine','mmengine.registry',
            'mmdet3d','mmdet3d.registry','mmdet3d.apis','mmdet3d.apis.inference')}
        mods['mmengine.registry'].init_default_scope=scope
        mods['mmdet3d.registry'].MODELS=SimpleNamespace(build=build)
        mods['mmdet3d.apis.inference'].convert_SyncBN=convert
        return mods,build,scope,model

    def test_train_components_kept_at_constructor_and_original_config_unchanged(self):
        cfg=Config(model=dict(type='CenterPoint',train_cfg=dict(pts=dict(code_weights=[1.]*10)),
                              norm_cfg=dict(type='naiveSyncBN')),default_scope='mmdet3d')
        before=deepcopy(cfg);mods,build,scope,model=self.runtime()
        with patch.dict('sys.modules',mods):self.assertIs(build_model(cfg,device='cpu'),model)
        built=build.call_args.args[0]
        self.assertEqual(built['train_cfg']['pts']['code_weights'],[1.]*10)
        self.assertEqual(built['norm_cfg']['type'],'BN')
        self.assertEqual(cfg,before);scope.assert_called_once_with('mmdet3d')
        model.to.assert_called_once_with('cpu');model.eval.assert_called_once()

    def test_missing_training_config_blocks_before_constructor(self):
        mods,build,_,_=self.runtime()
        for value in ({},dict(train_cfg=None)):
            with self.subTest(value=value),patch.dict('sys.modules',mods):
                with self.assertRaisesRegex(ValueError,'thiếu model.train_cfg'):
                    build_model(Config(model=value),'cpu')
        build.assert_not_called()

    def test_ignore_reports_missing_training_config_for_both_heads(self):
        for kind in ('CenterHead','Anchor3DHead'):
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'thiếu train_cfg'):
                install(SimpleNamespace(train_cfg=None),kind)


if __name__=='__main__':unittest.main()
