# Local ABRefine port

Source inspected: `/Users/fudanxth/Learning/Laboratory/Admis/graduate_first/ABRefine/model/equiformer_v2_model_FM.py` and `model/equiformerv2/`, on 2026-11-18.

`abrefine_ops/` preserves the local EquiformerV2 spherical/SO(2) operators and Jd.pt. Existing source attribution comments are retained. `abrefine_core.py` adapts construction to common scalar/vector channel counts, removes OCP imports, protein-only embeddings, energy/distogram branches and graph generation, and uses the local Gaussian radial basis. `abrefine.py` supplies RNA node features, current/source conditioning and the model interface. `heads/abrefine.py` separates the original equivariant attention output block. The random local-frame construction was replaced with a deterministic least-aligned-axis gauge. Jd.pt loads on CPU with weights_only=True.

The operators themselves reference EquiformerV2 and e3nn; the wigner helper retains its e3nn 0.4 attribution. No license file was found in the supplied local ABRefine directory. This provenance note records source and modifications, and does not assign a new license or imply permission for redistribution beyond the supplied project's terms.
