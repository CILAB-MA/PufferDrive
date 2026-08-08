# SAE analysis (active)

## Policy-seed attribution (canonical)

```bash
./analyze/sae/run_attribution_policy_seeds.sh
```

Per aligned seed (ReCord / Reactive / Self-play):

1. **Collect** activations (`SAVE_OBS`) — skip if already on disk  
   - Default: **moving partners only** (`other_speed > 0.5 m/s`), **no distance cap**  
   - Override: `MAX_MIN_DIST_M=25 MIN_OTHER_SPEED_MPS=0.5 FORCE_COLLECT=1`
2. **Train SAE** for all three methods on that seed’s data — skip if ckpt exists  
3. **Matching** (retrieval → mutual-NN triples → semantics)  
4. **Projection attribution** validation  
5. **Seed-level stats** → `$OUT_ROOT/seed_level_summary.json`

Outputs: `/data/puffer/sae/runs/attribution_policy_seeds/seed*/`

Helpers: `aggregate_seed_attribution.py`, `projection_attribution_validation.py`, `stats_utils.py`

Primary attribution metrics: `attr_p_brake`, `attr_steer`, `attr_p_throttle`, `attr_brake_minus_throttle`

## Building blocks (optional standalone)

| Script | Role |
|--------|------|
| `run_onpolicy_sae_pipeline.sh` | Shared-obs collect only |
| `run_train_sae.sh` | Train SAEs on a fixed `SAE_ROOT` |
