"""CPU regression tests: framework conversion must not weaken weight checks."""
from collections import OrderedDict
import unittest
import torch
from annoloop.mmdet_weights import load_weights


class VersionedKernel(torch.nn.Module):
    _version=2

    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.zeros(4,2,2,2,3))
        self.seen_version=None

    def _load_from_state_dict(self,state,prefix,metadata,*args):
        self.seen_version=metadata.get('version')
        if self.seen_version!=2:
            state[prefix+'weight']=state[prefix+'weight'].permute(4,0,1,2,3)
        super()._load_from_state_dict(state,prefix,metadata,*args)


class WeightTests(unittest.TestCase):
    def test_old_sparse_layout_uses_framework_conversion(self):
        model=VersionedKernel()
        old=torch.arange(96.).reshape(2,2,2,3,4)
        state=OrderedDict(weight=old);state._metadata={'':{'version':1}}
        load_weights(model,state)
        self.assertTrue(torch.equal(model.weight,old.permute(4,0,1,2,3)))
        self.assertEqual(model.seen_version,1)
        self.assertEqual(tuple(state['weight'].shape),(2,2,2,3,4))

    def test_new_checkpoint_metadata_prevents_double_conversion(self):
        source=VersionedKernel()
        with torch.no_grad():source.weight.copy_(torch.arange(96.).reshape_as(source.weight))
        target=VersionedKernel();load_weights(target,source.state_dict())
        self.assertEqual(target.seen_version,2)
        self.assertTrue(torch.equal(target.weight,source.weight))

    def test_ddp_prefix_preserves_wrapped_module_version(self):
        source=VersionedKernel();state=OrderedDict({'module.weight':source.weight.detach().clone()})
        state._metadata=OrderedDict([('module',{'version':2}),('',{'version':1})])
        target=VersionedKernel();load_weights(target,state)
        self.assertEqual(target.seen_version,2)
        self.assertEqual(state._metadata['module']['version'],2)

    def test_missing_unexpected_wrong_shape_and_nonfinite_fail(self):
        for state,pattern in [({},'missing='),
                ({'weight':torch.zeros(2,3),'extra':torch.ones(1)},'unexpected='),
                ({'weight':torch.zeros(3,2)},'framework load'),
                ({'weight':torch.full((2,3),float('nan'))},'NaN/Inf')]:
            with self.subTest(pattern=pattern),self.assertRaisesRegex(ValueError,pattern):
                load_weights(torch.nn.Linear(3,2,bias=False),state)

    def test_batchnorm_legacy_counter_is_only_missing_key_exception(self):
        source=torch.nn.BatchNorm1d(3);state=source.state_dict();del state['num_batches_tracked']
        load_weights(torch.nn.BatchNorm1d(3),state)
        del state['running_mean']
        with self.assertRaisesRegex(ValueError,'missing='):load_weights(torch.nn.BatchNorm1d(3),state)

    def test_prefix_collision_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'key trùng'):
            load_weights(torch.nn.Linear(3,2,bias=False),{'weight':torch.zeros(2,3),'module.weight':torch.ones(2,3)})


if __name__=='__main__':unittest.main()
