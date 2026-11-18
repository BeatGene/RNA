"""Run: python -m unittest discover -s tests -p 'test_modular_architecture.py' -v"""
import unittest
import torch
from etflow.models.model import BaseFlow
from etflow.data.confidence import confidence_features
from etflow.data.validation import validate_forward
from etflow.models.losses.safeguards import no_regret, protect


def example():
    pos = torch.tensor([[0.,0.,0.],[1.5,0.,0.],[0.,1.5,0.],[1.5,1.5,0.],
                        [0.,0.,0.],[1.4,0.,0.],[0.,1.4,0.],[1.4,1.4,.2]])
    batch = torch.arange(2).repeat_interleave(4)
    edges = torch.tensor([[0,1,1,2,2,3,4,5,5,6,6,7], [1,0,2,1,3,2,5,4,6,5,7,6]])
    d = dict(atomic_numbers=torch.tensor([6,7,8,15]*2), pos=pos,
             pos_pred=pos+.15*torch.randn_like(pos), batch=batch,
             edge_index=edges, edge_attr=torch.nn.functional.one_hot(torch.zeros(12,dtype=torch.long),7).float(),
             node_attr=torch.randn(8,3), geometry_bond_index=edges,
             ideal_bond_length=(pos[edges[0]]-pos[edges[1]]).norm(dim=-1),
             clash_exclusion_index=edges, residue_index=torch.tensor([0,0,1,1]*2),
             atom_name_id=torch.zeros(8,dtype=torch.long), target_mask=torch.ones(8,dtype=torch.bool),
             atom_to_token_idx=torch.tensor([0,0,1,1]*2), num_tokens=torch.tensor([2,2]),
             atom_plddt=torch.rand(8), token_pair_confidence=torch.rand(8,4),
             atom_mobility_attr=torch.rand(8,4))
    return d


def forward_args(d):
    return dict(z=d['atomic_numbers'], t=torch.tensor([[.2],[.7]]), pos=d['pos'],
                pos_source=d['pos_pred'], bond_index=d['edge_index'], edge_attr=d['edge_attr'],
                node_attr=d['node_attr'], batch=d['batch'],
                **{k:d[k] for k in ('atom_plddt','atom_to_token_idx','token_pair_confidence',
                                    'num_tokens','atom_mobility_attr','geometry_bond_index',
                                    'ideal_bond_length','clash_exclusion_index')})


def model(**kwargs):
    options=dict(hidden_channels=16, num_layers=1, num_heads=2, num_rbf=8,
                 node_attr_dim=3, cutoff_upper=5., dynamic_radius_cutoff=4.,
                 bond_loss_weight=.1, clash_loss_weight=.01, plane_loss_weight=0.,
                 use_source_augmentation=False)
    options.update(kwargs)
    m=BaseFlow(**options)
    m.log_helper=lambda *a, **kw: None
    return m


class ArchitectureTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        self.d=example()

    def test_all_backends_heads_forward_backward(self):
        for backend,head in [('torchmd','gated'),('torchmd','linear'),('abrefine','gated'),
                             ('abrefine','linear'),('abrefine','abrefine')]:
            with self.subTest(backend=backend,head=head):
                args=dict(representation_type=backend,output_head_type=head,confidence_conditioning=True,
                          use_mobility_v1=True)
                if backend=='abrefine':
                    args['representation_args']=dict(lmax_list=[1],mmax_list=[1],grid_resolution=8,
                                                     alpha_drop=0.,drop_path_rate=0.)
                m=model(**args)
                out=m(**forward_args(self.d))
                self.assertEqual(out.shape,(8,3)); self.assertTrue(torch.isfinite(out).all())
                loss=m.generic_step(self.d,0,'train')
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
                self.assertIsNotNone(next(m.mobility_head.parameters()).grad)

    def test_flow_paths_and_residual(self):
        for objective,path in [('flow','deterministic'),('flow','stochastic'),('residual','deterministic')]:
            m=model(training_objective=objective,flow_path=path,use_mobility_v1=True,
                    confidence_conditioning=False,use_source_augmentation=True)
            loss=m.generic_step(self.d,0,'train'); loss.backward()
            self.assertTrue(torch.isfinite(loss))

    def test_sampler_modes_steps_and_fixed_source(self):
        m=model(); seen=[]
        def field(**kw):
            seen.append((kw['pos_source'].clone(),kw['t'].clone()))
            return torch.ones_like(kw['pos'])*torch.tensor([1.,0.,0.])
        m.forward=field
        d=self.d; args=forward_args(d); args.pop('t'); args['pos_pred']=args.pop('pos_source'); args.pop('pos')
        for mode,calls in [('ode',3),('heun',6),('sde',3)]:
            seen.clear(); result=m.sample(**args,n_timesteps=3,sampler_type=mode)
            self.assertEqual(len(seen),calls)
            self.assertTrue(all(torch.equal(seen[0][0],s[0]) for s in seen))
            self.assertTrue(torch.isfinite(result).all())
        with self.assertRaises(ValueError): m.sample(**args,n_timesteps=0)

    def test_confidence_off_is_independent(self):
        m=model(confidence_conditioning=False,use_mobility_v1=True).eval()
        args=forward_args(self.d); first=m(**args)
        args.update(atom_plddt=None,atom_mobility_attr=None,token_pair_confidence=None)
        self.assertTrue(torch.allclose(first,m(**args)))
        d=dict(residue_index=self.d['residue_index'][:4])
        result=confidence_features(d,d['residue_index'])
        self.assertEqual(result['token_pair_confidence'].shape,(4,4))
        with self.assertRaises(ValueError): confidence_features(d,d['residue_index'],'required')

    def test_equivariance_translation_and_source_conditioning(self):
        for backend in ['torchmd','abrefine']:
            m=model(representation_type=backend,output_head_type='abrefine' if backend=='abrefine' else 'gated',
                    representation_args=dict(lmax_list=[1],mmax_list=[1],grid_resolution=8,
                                             alpha_drop=0.,drop_path_rate=0.) if backend=='abrefine' else None).eval()
            args=forward_args(self.d)
            initial=m(**args)
            q,_=torch.linalg.qr(torch.randn(3,3)); q=q*torch.linalg.det(q)
            moved=dict(args,pos=args['pos']@q+3.,pos_source=args['pos_source']@q+3.)
            torch.testing.assert_close(m(**moved),initial@q,atol=3e-4,rtol=3e-3)
            changed=dict(args,pos_source=args['pos_source']+.5*torch.randn_like(args['pos_source']))
            self.assertFalse(torch.allclose(initial,m(**changed)))

    def test_augmentations_and_empty_penalties(self):
        m=model(identity_pair_probability=1.,near_native_pair_probability=0.)
        source,identity,near=m.augment_residual_source(self.d['pos_pred'],self.d['pos'],self.d['batch'])
        torch.testing.assert_close(source,self.d['pos']); self.assertEqual(float(identity),1.)
        mask=torch.zeros(2,dtype=torch.bool); x=torch.zeros(2,requires_grad=True)
        loss=no_regret(x,x,mask); loss.backward(); self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(float(protect(self.d['pos'],self.d['pos_pred'],torch.zeros(8,dtype=torch.bool))),0.)

    def test_default_abrefine_deterministic_sampling_and_lmax2(self):
        m=model(representation_type='abrefine',output_head_type='abrefine')
        args=forward_args(self.d)
        m.eval()
        initial=m(**args)
        q,_=torch.linalg.qr(torch.randn(3,3)); q=q*torch.linalg.det(q)
        rotated=dict(args,pos=args['pos']@q,pos_source=args['pos_source']@q)
        torch.testing.assert_close(m(**rotated),initial@q,atol=3e-4,rtol=3e-3)
        m.train()
        args.pop('t'); args['pos_pred']=args.pop('pos_source'); args.pop('pos')
        a=m.sample(**args,n_timesteps=2)
        b=m.sample(**args,n_timesteps=2)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertTrue(m.training)

    def test_validation_metrics_and_masked_mobility(self):
        m=model(use_mobility_v1=True)
        self.d['target_mask'][4:]=False
        loss=m.generic_step(self.d,0,'val')
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_dataset_featurization_without_confidence(self):
        import tempfile
        from pathlib import Path
        from etflow.data.dataset import EuclideanDataset
        from torch_geometric.loader import DataLoader
        d={k:v[:4].clone() for k,v in self.d.items()
           if k in ('pos','pos_pred','atomic_numbers','target_mask','residue_index','atom_name_id')}
        d.update(edge_index=self.d['edge_index'][:,:6],edge_attr=self.d['edge_attr'][:6],
                 geometry_bond_index=self.d['geometry_bond_index'][:,:6],
                 ideal_bond_length=self.d['ideal_bond_length'][:6],
                 rnafm_embedding=torch.randn(2,640),residue_type_id=torch.tensor([0,1]))
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'train'; folder.mkdir()
            torch.save(d,folder/'a.pt'); torch.save(d,folder/'b.pt')
            dataset=EuclideanDataset(tmp,confidence_mode='off')
            batch=next(iter(DataLoader(dataset,batch_size=2)))
            self.assertEqual(batch.node_attr.shape,(8,674))
            self.assertEqual(batch.token_pair_confidence.shape,(8,4))
            m=model(node_attr_dim=674,use_mobility_v1=True)
            loss=m.generic_step(batch,0,'train'); loss.backward()
            self.assertTrue(torch.isfinite(loss))
            # Exercise real Lightning hooks, optimizer, validation logging and clipping.
            from lightning.pytorch import Trainer
            from etflow.models.base import BaseModel
            m.log_helper=BaseModel.log_helper.__get__(m)
            trainer=Trainer(fast_dev_run=1,accelerator='cpu',devices=1,logger=False,
                            enable_checkpointing=False,enable_progress_bar=False,
                            enable_model_summary=False,gradient_clip_val=1.)
            trainer.fit(m,train_dataloaders=DataLoader(dataset,batch_size=2),
                        val_dataloaders=DataLoader(dataset,batch_size=2))
            self.assertEqual(trainer.global_step,1)

    def test_base_loss_equal_graph_weighting(self):
        from etflow.models.losses.regression import flow_matching_loss
        pred=torch.tensor([[3.,0.,0.],[0.,4.,0.],[10.,0.,0.]])
        loss=flow_matching_loss(pred,torch.zeros_like(pred),torch.tensor([0,0,1]),
                                torch.tensor([True,True,False]))
        self.assertEqual(float(loss),3.5)

    def test_static_graph_without_base_edge_attributes(self):
        m=model(dynamic_graph=False,edge_attr_dim=0,confidence_conditioning=True)
        args=forward_args(self.d); args['edge_attr']=None; args['t']=args['t'].view(-1)
        self.assertTrue(torch.isfinite(m(**args)).all())

    def test_cross_graph_validation_and_pair_offsets(self):
        m=model(); e=torch.tensor([[0,4],[4,0]])
        with self.assertRaises(ValueError):
            m(**dict(forward_args(self.d),bond_index=e))
        pairs=torch.arange(32).view(8,4)
        found=m.build_confidence_edge_attr(torch.tensor([[0,4],[2,6]]),self.d['batch'],
                self.d['atom_to_token_idx'],pairs,self.d['num_tokens'])
        torch.testing.assert_close(found,torch.stack([pairs[1],pairs[5]]))


if __name__=='__main__': unittest.main()
