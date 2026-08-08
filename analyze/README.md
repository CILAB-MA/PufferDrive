# analyze/

Drive PBT population을 평가/디버깅하기 위한 스크립트 모음입니다. zero-shot / self-play /
human-replay / WOSAC 평가 실행, 결과 집계, 맵 서브셋 큐레이션, 실패 케이스 시각화 등을 다룹니다.

공통 규칙:
- 대부분의 `.sh` 스크립트는 `GPU_ID`(또는 `POPULATION_MODE`/`MODE`)를 위치 인자로 받고
  기본값이 있습니다. 예: `./script.sh [GPU_ID=0] [MODE=nominal]`
- `puffer eval`/`puffer zeroshot`을 호출하는 스크립트 중 아래 표시된 것들은 `GPU_ID` 자리에
  `cpu`를 넣으면 CPU 전용으로 실행됩니다 (`CUDA_VISIBLE_DEVICES=""` + `--train.device cpu`).
- 결과는 `/data/puffer/results/<exp_or_population>/...` 아래에 저장되고,
  체크포인트는 `/data/puffer/<population>/` 또는 `/data/puffer/experiments/<exp>/`에서 읽습니다.

## 맵 큐레이션

- **`find_worst_maps.py`** — `<id>_selfplay.json` 시나리오 로그에서 failure score(score,
  collision/offroad/dnf rate 기반)로 맵을 랭킹/필터링합니다.
  ```
  python3 analyze/find_worst_maps.py <path_to_selfplay.json> --filter "score<1" --sort-by collision_rate --top 200 --out /workspace/analyze/worst.csv

  # 예시
  python3 analyze/find_worst_maps.py <path> --filter "score<1" "collision_rate>0" --out worst.csv
  ```
- **`build_map_subset.py`** — CSV의 `map_id` 컬럼(예: `find_worst_maps.py` 출력)이 가리키는
  `map_*.bin` 파일들을 새 폴더로 복사하면서 `map_000.bin`, `map_001.bin`, ... 순서로
  재넘버링합니다. 그대로 `--env.map-dir`로 쓸 수 있습니다.
  ```
  python3 analyze/build_map_subset.py <csv from find_worst_maps.py> --dest-name <new_folder_name> --limit 100

  # 예시
  python3 analyze/build_map_subset.py analyze/worst.csv --dest-name worst_100 --limit 100
  ```
  새 폴더 안에 `source_map_ids.csv`(새 인덱스 ↔ 원본 map_id 매핑)도 같이 남겨서
  나중에 원본을 추적할 수 있습니다.

## Self-play / human-replay 평가 파이프라인

population/experiment 폴더의 체크포인트마다 평가를 돌리고, 결과를 mean/std로 집계한 뒤
필요하면 LaTeX 표까지 만드는 흐름입니다.

- **`population_selfplay.sh`** — 각 체크포인트가 자기 자신과 대결 (reactive-play zero-shot).
  `[GPU_ID=0|cpu] [POPULATION_MODE=popul_lane]`. `.../selfplay/zeroshot_reactive.json`을
  `"<model>_vs_selfplay"` 키로 저장한 뒤 `aggregate_selfplay.py`를 실행합니다.
- **`population_selfplay_human_eval.sh`** — 각 체크포인트를 self-play가 아니라 기록된 사람
  주행 궤적과 대결시킵니다 (`--eval.human-replay-eval`). `[GPU_ID=0|cpu]
  [POPULATION_MODE=popul_lane]` (`NUM_MAPS` 환경변수, 기본 10000). `.../logreplay.json`을
  `_vs_selfplay` 접미사 없이 순수 model id 키로 저장합니다 — **이 파일에는
  `aggregate_selfplay.py`를 돌리면 안 됩니다.** 키 형식이 안 맞아서 `No *_vs_selfplay
  entries` 에러가 납니다.
- **`population_play.sh`** — `population_selfplay.sh`와 같은 개념(심볼릭 링크 기반
  reactive self-play)의 별도 진입점입니다.
- **`logreplay.sh`** — `population_selfplay_human_eval.sh`의 더 오래되고 단순한 버전으로,
  `/data/puffer/experiments/<MODE>/` 전용입니다. 결과 경로 관리, 시나리오 로깅, 집계 단계가
  없습니다. `population_selfplay_human_eval.sh` 사용을 권장합니다.
- **`aggregate_selfplay.py`** — `zeroshot_reactive.json`에서 `*_vs_selfplay` 항목을
  mean/std로 집계해서 `population_selfplay_summary.json`을 씁니다.
  ```
  python3 analyze/aggregate_selfplay.py /data/puffer/results/<pop>/selfplay/zeroshot_reactive.json [--ego-only]
  ```
- **`population_selfplay_latex.py`** — 여러 population variant의
  `population_selfplay_summary.json`을 모아 LaTeX 표(mean ± std)를 만듭니다.
  ```
  python3 analyze/population_selfplay_latex.py --populations popul_lane popul_lane_nominal --out table.tex
  ```

## Cross-play / zero-shot 평가 파이프라인

- **`zero_shot.sh`** — cross-play(ego vs. 다른 population)를 모든 ego/other 쌍에 대해
  실행합니다. `[GPU_ID=0] [FOLDER] [POPULATION_MODE] [MODE=unseen_other_rewards]
  [SCENARIO_LOG_DIR]`. `zeroshot(.json|_reactive.json)`과 쌍별
  `<ego>_vs_<other>_<mode>.json` 시나리오 로그를 씁니다.
- **`heatmap.py`** / **`heatmap.sh`** — `zeroshot(.json|_reactive.json)` 매치업 파일로
  agent-vs-agent 점수 히트맵을 그리고, nominal self-play baseline으로 정규화합니다.
  `heatmap.sh [GPU_ID=0]`는 모든 mode × condition-type 조합에 대해 `heatmap.py`를 돌립니다.
- **`analyze.py`** — WOSAC 지표 대비 self-play 개선폭 상관관계 곡선을 그립니다
  (nominal vs. 선택한 long-tail 조건).
  ```
  python3 analyze/analyze.py --long-tail lane_breaker --mode replay --wosac all
  ```
- **`analyze_scene.py`** — `wosac_scene*.json`에서 지표별 최악(하위 십분위) WOSAC 시나리오를
  찾아 `wosac_tail_{score,collision,distance}.json`을 씁니다.
- **`compare_exps.py`** — 여러 experiment 폴더의 `logreplay.json` / `wosac.json` /
  `zeroshot_reactive.json`을 비교해서 CSV 표와 bar/subplot 그림을 출력합니다.
  ```
  python3 analyze/compare_exps.py --exps exp_a exp_b --format both --out-dir /tmp/compare
  ```
- **`wosac.sh`** — experiment `MODE` 폴더의 모든 모델에 대해 WOSAC realism 평가를
  실행합니다. `[GPU_ID=0] [MODE=nominal]`. `wosac.json`을 생성하고,
  `analyze.py` / `compare_exps.py`가 이를 소비합니다.

## Linear-probe 파이프라인

순서대로 실행합니다:
1. **`generate_lp_dataset.sh`** — nominal 모델마다 `MODE` 아래의 모든 모델을 상대로
   linear-probe 데이터셋을 생성합니다. `[GPU_ID=0] [MODE=nominal]`
2. **`lp_train.sh`** — nominal 모델마다 linear probe를 학습합니다.
   `[GPU_ID=0] [MODE=nominal]`
3. **`lp_evaluate.sh`** — 학습된 probe를 (nominal ego, MODE other) 모든 쌍에 대해
   평가합니다. `[GPU_ID=0] [MODE=nominal]`

## Rollout / replay 데이터 수집

- **`collect_replay_rollout.sh`** — GPU/인덱스 구간에 대한 replay buffer shard를
  수집합니다: `./collect_replay_rollout.sh <cuda_device> <collect_start_idx>
  <collect_end_idx> [num_collect_rollout=50]`. `<population_path>/splits/` 아래에
  `actions_<start>_<end>.npy` (+ agent_offsets/map_ids/global_ids) shard를 씁니다.
- **`data_concat.py`** — 위 shard들을 (연속성 검증 후) 하나의 데이터셋으로 합칩니다.
  ```
  python3 analyze/data_concat.py --population-path PATH --total-rollouts N
  ```
- **`generate_replay_dataset.sh`** — `puffer_drive_pbt`용 zero-shot replay 데이터셋 생성을
  실행하는 얇은 wrapper입니다. `[GPU_ID=0] [MODE=nominal]`

## 실패 디버깅 / 시각화

- **`failure_visualize.py`** — zero-shot 시나리오 로그(또는 라이브 롤아웃 스캔)에서
  충돌 실패 장면을 찾아 시각화합니다. 실패한 맵마다 GIF를 렌더링하는 셸 스크립트를
  생성할 수도 있습니다.
  ```
  python3 analyze/failure_visualize.py --from-scenario-log DIR --map-dir DIR --ego-collision-threshold 0.01 --out /tmp/failures.json
  python3 analyze/failure_visualize.py --emit-viz-commands-only /tmp/failures.json --model CKPT.pt --write-viz-script /tmp/run_failure_viz.sh
  ```
- **`viz.py`** — C `./visualize` 바이너리를 통해 하나의 정책 롤아웃을 하나의 맵에서
  GIF/MP4로 렌더링합니다. `failure_visualize.py`가 생성한 스크립트가 맵마다 이를 호출합니다.
  ```
  python3 analyze/viz.py --model CKPT.pt --map-bin /path/to/map_XXX.bin --out viz.gif
  ```
- **`debug_minimum_distance.py`** — `Drive_PBT` agent-sampling 정합성 검증 도구입니다
  (index/LUT identity, `minimum_distance` 계산, episode-reset 엣지 케이스 등).
  ```
  python3 analyze/debug_minimum_distance.py --population-path /data/puffer/popul_lane_nominal
  ```

## 스크립트 간 파이프라인 관계

```
find_worst_maps.py -> build_map_subset.py -> 새 map_dir 폴더

population_selfplay.sh / population_play.sh -> zeroshot_reactive.json
  -> aggregate_selfplay.py -> population_selfplay_summary.json
  -> population_selfplay_latex.py

population_selfplay_human_eval.sh -> logreplay.json (순수 model-id 키;
  이 파일에 aggregate_selfplay.py를 돌리면 안 됨)

generate_lp_dataset.sh -> lp_train.sh -> lp_evaluate.sh

collect_replay_rollout.sh -> actions_<start>_<end>.npy shard -> data_concat.py

zero_shot.sh -> zeroshot(.json|_reactive.json), scenario_logs/*.json
  -> heatmap.py / heatmap.sh, analyze.py, compare_exps.py, failure_visualize.py

wosac.sh -> wosac.json -> analyze.py, compare_exps.py
  (analyze_scene.py는 wosac_scene*.json에서 wosac_tail_*.json을 도출)

failure_visualize.py --write-viz-script -> 생성된 스크립트가 맵마다 viz.py 호출
```
