# RWT Stage 3B preparation — CPU audit and GPU pilot handoff

**Kết luận: GPU-PILOT-READY cho một pilot có kiểm soát trên split A/B/D hiện
tại; chưa sẵn sàng cho full model-based RL benchmark.** Data integrity,
checkpoint save/load/resume, train-only normalization, state-only training và
rollout evaluator đã chạy trên CPU. CUDA chưa có trên máy local, nên pilot GPU
chưa được kiểm chứng thực tế. NEO upstream không thay đổi.

## Kiến trúc và protocol

`StateDynamics` là residual MLP 17D state + 6D action → 17D normalized delta,
với loss chỉ trên delta. `RewardPredictor` là MLP riêng 17D + 6D → scalar
reward, có parameters và train-only statistics riêng. `SeparatedWorldModel`
chỉ ghép hai predictions tại interface MPC hiện có; không chia sẻ trunk hoặc
optimizer. Pooled và history models không nhận regime ID. Oracle one-hot A/B/D
chỉ là diagnostic; C không có oracle code và không dùng để train/chọn model.

Giữ nguyên 1.800 train + 200 validation transitions mỗi A/B/D, seeds và fixed
rollout/MPC initial states từ final CPU diagnostic. Dùng ba training seeds
41/42/43, width 64, 35 epochs, batch 256, train-only normalization. C chỉ
được mở sau khi checkpoints đã chọn bằng A/B/D validation. Ridge dùng cùng
train transitions, penalty 1.0. Baselines dynamics **cùng state-only objective**.

## Kết quả trên split cũ

Mean ± sample SD qua ba training seeds, tính trên validation A/B/D:

| State dynamics | Raw MSE | Normalized MSE |
|---|---:|---:|
| Linear ridge | 0,4228 | 0,0894 |
| Pooled state-only | **0,3653 ± 0,0062** | **0,0654 ± 0,0013** |
| Oracle-conditioned state-only | **0,3249 ± 0,0152** | **0,0599 ± 0,0028** |
| History-conditioned state-only | 0,3680 ± 0,0124 | 0,0651 ± 0,0013 |

| Regime | Ridge raw / norm | Pooled raw / norm | Oracle raw / norm | History raw / norm |
|---|---:|---:|---:|---:|
| A | 0,4842 / 0,1041 | 0,4186 / 0,0743 | 0,4204 / 0,0736 | 0,4370 / 0,0769 |
| B | 0,4970 / 0,1081 | 0,4519 / 0,0810 | 0,4137 / 0,0784 | 0,4669 / 0,0817 |
| D | 0,2873 / 0,0559 | 0,2253 / 0,0409 | 0,1406 / 0,0278 | 0,1999 / 0,0365 |

Reward network riêng đạt validation reward MSE **0,0040** trên A/B/D
(A 0,00522; B 0,00381; D 0,00305). Pooled state-only giảm raw MSE so với
shared dynamics/reward trunk cũ **0,4637 → 0,3653** trên cùng split. Các raw
dimension giảm nhiều nhất là 15, 12, 14, 16, 13, 11. `per_dimension.csv`
ghi đủ 17 dimension × regime × seed. Oracle hơn pooled 0,0404 raw MSE (11%),
đặc biệt ở D; history đơn giản vẫn chưa vượt pooled. So sánh lợi ích regime
conditioning ở đây dùng **cùng state-only loss**, tránh confound reward loss
của Stage 3A.1.

Rollout endpoint MSE trên cùng fixed simulator states/actions, trung bình A/B/D:

| Model | h1 | h3 | h5 |
|---|---:|---:|---:|
| Ridge | 1,679 | 1,918 | 0,773 |
| Pooled state-only | 1,689 | 1,743 | 0,802 |
| Oracle state-only | 1,394 | 1,339 | 0,790 |
| History state-only | 0,994 | 1,594 | 0,986 |

Không có NaN/Inf. C held-out có h3 MSE ridge **2,199**, pooled **1,100**,
history **1,770**; đây chỉ là evaluation, không dùng chọn seed/hyperparameter.
History cải thiện h1 nhưng không nhất quán tại h5. Endpoints không đơn điệu do
dao động động lực học, và horizon 5 vẫn chưa chính xác để tin planned values.

## Matched physics và coverage

Trên cùng 16 train-like `(qpos,qvel,action)` probes, A–B separation L2 của
physics hiện tại chỉ **0,338**; A–D là **2,273**. Tín hiệu friction yếu hơn
lỗi mô hình trên cùng probes. Additive closure MuJoCo hiện tại có mean L2
**0,276**, relative **8,3%** (đo ở final CPU diagnostic).

Đề xuất physics mới được **chọn từ A/B/D בלבד** và lưu cố định trong
`proposed_regimes.yaml`: B friction **0,3** thay 0,6; D motor gain **0,5** thay
0,7; A giữ nguyên. Trước khi mở C, matched separation A–B tăng **0,338 →
0,708**, A–D **2,273 → 3,796**; ba seeds × 200 bước random/oscillator trên
A/B/D đều hữu hạn. Sau khi config đã cố định, evaluator kiểm tra C=(0,3;
0,5): trajectories hữu hạn, nhưng closure tăng thành L2 **0,589**, relative
**11,0%**. Config này **chưa dùng** để train các baseline hoặc GPU pilot hiện
tại; dùng nó sẽ cần một dataset A/B/D mới và một validation protocol đã khóa.

Dataset cũ chủ yếu ở vận tốc thấp: train fraction `x velocity > 0,5` là
**15,0% / 15,1% / 7,5%** cho A/B/D; mean absolute action khoảng 0,34. Đã
thu thêm **600 train transitions/regime** từ bounded oscillator và learned MPC
controller (không nhận regime metadata), lưu NPZ chỉ có transition/reward/
termination, không có C. Addendum có checksum và seeds riêng; validation cũ
không đổi. Mean absolute action tăng lên khoảng 0,38 khi gộp, nhưng fraction
velocity > 0,5 giảm xuống **13,2% / 13,2% / 7,1%**. Đây là **negative coverage
result**: controller hợp lệ và action đa dạng hơn vẫn không tạo đủ locomotion
nhanh. Không dùng addendum trong GPU pilot mặc định.

Train lại cùng bốn baseline trên A/B/D train+addendum, validation vẫn khóa:

| Predictor | Base raw MSE | Augmented raw MSE |
|---|---:|---:|
| Ridge | 0,4228 | 0,4571 |
| Pooled | 0,3653 | 0,3915 |
| Oracle | 0,3249 | 0,3449 |
| History | 0,3680 | 0,3769 |

Reward MSE tăng **0,0040 → 0,0049**. Addendum có thể giảm OOD plan rate
trong một số smoke cases nhưng làm xấu validation. Ngưỡng nearest-train OOD
cũng được tính lại khi đổi train set, nên OOD rates giữa hai datasets không
phải so sánh tuyệt đối. Không chọn addendum chỉ vì một metric MPC thuận lợi.

## MPC với hai mạng riêng

MPC-CEM dùng pooled state-only checkpoint và reward checkpoint riêng. Evaluator
replay **đúng plan CEM** từ cùng `(qpos,qvel)` ở A/B/D/C, ba seeds, horizons
3/5, so với shared-trunk Stage 3A.1 dưới cùng protocol:

| MPC metric | Shared trunk cũ h3 / h5 | Separate networks h3 / h5 |
|---|---:|---:|
| Planned-return MAE | 0,578 / 0,869 | **0,271** / 1,041 |
| State MSE mỗi bước | 1,941 / 1,354 | **1,066** / 1,423 |
| Predicted states ngoài validation-99% train support | 41,7% / 45,0% | **30,6%** / 40,0% |

Horizon 3 cải thiện, **horizon 5 return MAE xấu hơn**. Reward model riêng
không loại bỏ lỗi rollout hoặc OOD. Predicted/actual returns và từng reward,
state, action nằm trong `mpc_plans.csv`/`mpc_plan_steps.csv`. Không có invalid
CEM candidates. Không so sánh với SAC vì đây chưa phải policy benchmark.

## Readiness và tái hiện

CPU runner lưu best/latest checkpoint cho mọi training seed, optimizer/RNG
state, dataset hashes, resolved config, training curves, validation/rollout/
MPC metrics. Resume trên CPU đã kiểm tra: dùng lại checkpoint/dataset, metrics
không đổi. Sau load, runner đối chiếu **means và scales** của state/action/
delta/context/reward với A/B/D train arrays. Addendum có kiểm tra checksum,
field schema, seed và exact transition overlap với base train/validation.
GPU-pilot config giữ `pooled`, `history`, `oracle` diagnostic, reward predictor
và ridge reference. Oracle được train/evaluate chỉ trên A/B/D với regime labels
đặc quyền và không tham gia dự đoán C hoặc MPC. Ba training workers chạy ba
seed độc lập trên CUDA; dataset nhỏ được chuyển trực tiếp lên device, nên không
có DataLoader workers để nhân bản dữ liệu hay tăng overhead.

Máy local: Python 3.11.9, PyTorch 2.13.0+cpu, Gymnasium 1.3.0, MuJoCo 3.3.7;
`cuda_available=false`. GPU training chưa được chạy. Trên GPU VM, từ gốc repo:

```powershell
pip install -e ".[rwt-mujoco,test]"
$env:PYTHONPATH='src'
python experiments/rwt/prepare_stage3b_dataset.py --config experiments/rwt/configs/stage3a/cpu_smoke.yaml
python experiments/rwt/run_stage3b_pilot.py --check-gpu
python experiments/rwt/check_stage3b_cuda.py
python experiments/rwt/run_stage3b_pilot.py --config experiments/rwt/configs/stage3b/gpu_pilot.yaml --device cuda
```

Trên Linux VM, thay dòng môi trường bằng `export PYTHONPATH=src`; các lệnh
`python` giữ nguyên. Nếu dùng `python3`, thay đồng nhất cả ba lệnh `python`.

Nếu bị ngắt, thêm `--resume` vào lệnh cuối. `prepare_stage3b_dataset.py`
chỉ xác thực dataset nếu đã có; nếu chưa có, nó thu thập A/B/D với đúng
seed/count/protocol khóa, không train Stage 3A model. Để so sánh số liệu chính
xác giữa hai máy, sao chép NPZ/manifest từ máy local; chạy lại simulator có
thể khác chút do phiên bản/hardware. GPU pilot mặc định chỉ train pooled,
history và reward predictor, đồng thời đánh giá ridge; oracle đã được train
và báo cáo **riêng trong CPU diagnostic**, không tham gia GPU pilot, MPC hay
model selection C.

Lệnh CPU đã chạy: `run_stage3b_pilot.py --config
experiments/rwt/configs/stage3b/cpu_smoke.yaml --device cpu`; ablation addendum
dùng `cpu_augmented.yaml`. Kết quả ở `results/rwt/stage3b_cpu_smoke/`,
`results/rwt/stage3b_cpu_augmented/`, `results/rwt/stage3b_coverage_addendum/`
và `results/rwt/stage3b_proposed_physics.json`. Không sửa NEO upstream, không
chạy full benchmark.

Kiểm thử cuối: **63 RWT CPU tests passed**; `compileall` trên `src/rwt`,
`experiments/rwt` và RWT tests; `git diff --check` không báo lỗi. Những phần
chưa kiểm chứng là CUDA training/resume, độ ổn định dài hạn của proposed
physics và locomotion nhanh bằng một controller mạnh hơn.
