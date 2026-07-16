# SAE analysis (active)

## Policy-seed attribution (canonical)

```bash
./analyze/sae/run_attribution_policy_seeds.sh
```

Per aligned seed (ReCord / Reactive / Self-play):

1. **Collect** activations (`SAVE_OBS`) — skip if already on disk  
2. **Train SAE** for all three methods on that seed’s data — skip if ckpt exists  
3. **Matching** (retrieval → mutual-NN triples → semantics)  
4. **Projection attribution** validation  
5. **Seed-level stats** → `$OUT_ROOT/seed_level_summary.json`

Outputs: `/data/puffer/sae/runs/attribution_policy_seeds/seed*/`

Helpers: `aggregate_seed_attribution.py`, `projection_attribution_validation.py`, `stats_utils.py`

## Building blocks (optional standalone)

| Script | Role |
|--------|------|
| `run_onpolicy_sae_pipeline.sh` | Shared-obs collect only |
| `run_train_sae.sh` | Train SAEs on a fixed `SAE_ROOT` |

## Archived

Prior tracks (occupancy / Ego-CF / onset / belief / review) live in `_archive/`.
