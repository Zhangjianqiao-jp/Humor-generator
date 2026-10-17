import importlib.util
from pathlib import Path
import unittest
import torch
spec=importlib.util.spec_from_file_location('alignment',Path(__file__).resolve().parents[1]/'scripts/statebridge_alignment.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class AlignmentTests(unittest.TestCase):
    def test_reflection_distinguishes_paper_and_official_constraint(self):
        cross=torch.diag(torch.tensor([3.,2.,-1.],dtype=torch.float64))
        reflected=a.orthogonal_map(cross,False);proper=a.orthogonal_map(cross,True)
        self.assertAlmostEqual(float(torch.linalg.det(reflected)),-1)
        self.assertAlmostEqual(float(torch.linalg.det(proper)),1)
        self.assertGreater(float(torch.trace(reflected.T@cross)),float(torch.trace(proper.T@cross)))
    def test_identity_recovers_reference_direction_and_calibrated_norm(self):
        e=torch.tensor([[1.,0.],[0.,2.],[-1.,0.],[0.,-2.]],dtype=torch.float64)
        result,diag=a.align_message(e,e,e,anchor_weight=0)
        target=e/e.norm(dim=-1,keepdim=True)*1.5
        torch.testing.assert_close(result,target,rtol=1e-8,atol=1e-8)
        self.assertLess(diag['rotation_orthogonality_max_error'],1e-10)
    def test_nearest_search_chunk_invariant_and_first_tie(self):
        vocab=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[-1.,0.]])
        query=torch.tensor([[2.,0.],[0.,3.]])
        self.assertEqual(a.nearest_vocab(query,vocab,1).tolist(),[0,1])
        self.assertEqual(a.nearest_vocab(query,vocab,4).tolist(),[0,1])
    def test_anchor_one_returns_exact_vocab_vectors(self):
        e=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]])
        result,diag=a.align_message(e,e,e,anchor_weight=1,chunk_size=1)
        torch.testing.assert_close(result,e[diag['nearest_vocab_ids']])
    def test_degenerate_message_finite_but_not_declared_informative(self):
        h=torch.ones(3,2);e=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]])
        result,diag=a.align_message(h,e,e)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(diag['scope'],'alignment_only_not_semantic_or_quality_guarantee')
    def test_alignment_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):a.align_message(torch.ones(3,2),torch.ones(2,2),torch.ones(3,2))

if __name__=='__main__':unittest.main()
