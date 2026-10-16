# RWT Stage 3A.1 — final CPU diagnostic

**Kết luận: NOT GPU-READY cho full world-model MPC training.** Pipeline chạy đúng
và không thấy rò rỉ C, nhưng mô hình dùng trong MPC còn sai số rollout lớn,
chọn nhiều trạng thái ngoài support và cache chưa cải thiện nhất quán. Đây là
diagnostic CPU, không phải full benchmark hay kết quả Stage 3B.

## Protocol và mô hình

Dùng lại đúng 1.800 train + 200 validation transitions cho từng A/B/D; seed
train/validation tách biệt, không trùng transition. C chỉ mở trong evaluator
**sau** khi đã chọn checkpoint bằng validation A/B/D. C không đi vào
normalization, model selection hay threshold support. Ba training seeds
41/42/43, cùng width 64, batch 256, tối đa 35 epochs, early-stopping patience
7 (cả ba đều tới epoch 35). Các mô hình:

- **Pooled:** MLP Stage 3A gốc, chung trunk dự đoán state delta và reward.
- **Oracle:** cùng MLP cộng one-hot regime A/B/D; đây là diagnostic có nhãn đặc
  quyền. Không thể dùng oracle để đánh giá C hay làm main agent.
- **History:** cùng MLP cộng ba `(action, observed delta, mask)` trước thời điểm
  dự đoán; reset context ở ranh giới episode, không nhận regime ID.
- **State-only ablation:** MLP pooled giữ nguyên kiến trúc và seed, chỉ tối ưu
  normalized state-delta loss. Reward output của ablation này không được train
  và **không** được dùng trong MPC.

Số parameters lần lượt là 6.866 / 7.058 / 11.474 / 6.866. Toàn bộ loss,
checkpoint và normalization của mô hình được tính từ A/B/D. Trên validation,
MSE trung bình qua ba training seeds là:

| Predictor | Raw state MSE | Normalized state MSE | Reward MSE |
|---|---:|---:|---:|
| Persistence (Stage 3A.1) | 6,9181 | 0,9887 | — |
| Linear ridge (Stage 3A.1) | 0,4228 | 0,0894 | — |
| Pooled shared reward trunk | 0,4637 | 0,0866 | 0,0046 |
| Oracle-conditioned shared trunk | 0,4313 | 0,0802 | 0,0035 |
| History-conditioned shared trunk | 0,4689 | 0,0865 | 0,0056 |
| Pooled state-only ablation | **0,3668** | **0,0657** | — |

Per-regime raw / normalized MSE:

| Regime | Pooled | Oracle | History | State-only |
|---|---:|---:|---:|---:|
| A | 0,5287 / 0,0994 | 0,5231 / 0,0947 | 0,5550 / 0,1021 | **0,4182 / 0,0753** |
| B | 0,5777 / 0,1083 | 0,5438 / 0,1032 | 0,5939 / 0,1083 | **0,4567 / 0,0813** |
| D | 0,2846 / 0,0520 | 0,2269 / 0,0428 | 0,2579 / 0,0491 | **0,2256 / 0,0407** |

Pooled thua ridge về raw MSE chủ yếu vì **negative transfer từ reward loss
chung trunk** trong budget này: với cùng seed/kiến trúc/data, bỏ reward loss
giảm raw MSE 0,0969 (20,9%) và vượt ridge. Pooled mất nhiều raw MSE ở các
dimension 12, 13, 11, 16, 14; trong khi normalized loss lại tốt hơn ridge ở
dimension 9, 10, 8, 0, 1. Đây là bằng chứng về objective trade-off, không phải
chứng minh reward training luôn gây hại ở mọi budget. Toàn bộ **17 dimension**
theo regime và seed có ở `per_dimension.csv`.

Oracle giảm raw MSE 0,0324 (7,0%) so với pooled, và giảm ở cả ba seed, nhưng
history đơn giản không cải thiện. Regime conditioning giúp một phần, rõ nhất
ở D, song **không giải thích phần lớn lỗi**. Oracle là upper-bound diagnostic
cho seen regimes, không phải model deployable.

## Rollout và matched physics

Evaluator lấy ba trạng thái MuJoCo cố định mỗi regime, quan sát ba transitions
đầu để khởi tạo history, rồi rollout tự do trên cùng chuỗi 10 actions. MSE
endpoint trung bình trên A/B/D với checkpoint chọn bằng A/B/D:

| Predictor | h=1 | h=3 | h=5 | h=10 |
|---|---:|---:|---:|---:|
| Pooled | 1,720 | 1,809 | 0,824 | 1,301 |
| Oracle | 1,349 | 1,708 | 0,770 | 1,049 |
| History | 1,652 | 2,424 | 1,253 | 1,346 |
| State-only | 1,692 | 1,840 | 0,765 | 1,178 |

Endpoint error không đơn điệu vì HalfCheetah dao động; không có NaN/Inf. C có
trong `rollout.csv` cho pooled/history/state-only, không có oracle C. Sai số
rollout trên trạng thái gần reset lớn hơn validation ngẫu nhiên; độ chính xác
horizon 3–5 hiện **chưa đủ tin cậy** để dùng planned value như đánh giá return.
Theo regime, MSE endpoint h3 / h5 của checkpoint đã chọn:

| Regime | Pooled | Oracle | History | State-only |
|---|---:|---:|---:|---:|
| A | 1,133 / 0,895 | 1,621 / 0,973 | 2,359 / 1,459 | 1,777 / 0,903 |
| B | 1,144 / 0,790 | 1,690 / 0,896 | 2,351 / 1,332 | 1,709 / 0,797 |
| D | 3,149 / 0,787 | 1,813 / 0,443 | 2,563 / 0,966 | 2,033 / 0,593 |
| C, held-out | 2,365 / 0,947 | — | 2,071 / 0,975 | 1,301 / 0,577 |

One-step gain không bảo đảm rollout gain: oracle/state-only h3 còn kém pooled
trên A/B. C chỉ là phép đo held-out, không dùng để train, chọn seed hoặc tune.

Với 32 matched `(qpos, qvel, action)` probes, evaluator chạy cùng input dưới
A/B/D/C. 16 probes theo policy thu thập train, 16 probes dùng action rộng và
velocity mở rộng. Mean pairwise next-state L2 ở vùng train-like:

| Pair | A–B | A–D | B–D | D–C |
|---|---:|---:|---:|---:|
| Separation | 0,338 | 2,273 | 2,389 | 0,372 |
| Separation / pooled error | 0,141 | 0,981 | 1,004 | 0,134 |

Ở vùng held-out, A–B và D–C có separation/error chỉ 0,045 và 0,050. Mô hình
pooled có lỗi lớn hơn tín hiệu phân biệt friction. Oracle tạo dự đoán khác nhau
cho A/B/D, nhưng separation dự đoán A–D chỉ 1,554 so với thực 2,273 ở vùng
train-like. State-only giảm model error trên matched probes, song không loại
bỏ sự chồng lấp này.

MuJoCo closure `F_C-F_B-F_D+F_A` có mean L2 **0,276**, tương đương 8,3% tổng
độ lớn hai hiệu ứng A→B và A→D; ở vùng train-like là 10,2%. Vì vậy additive
composition **không chính xác tuyệt đối** cho perturbations MuJoCo này. Lỗi
closure gần độ lớn A–B/D–C; không nên giả định executor cộng tuyến tính là
ground truth ở Stage 3.

## MPC: đúng cùng initial state và action sequence

Với ba seeds × bốn regimes × horizons 3/5, evaluator lưu `qpos/qvel`, lấy đúng
plan CEM rồi replay **chính plan đó** trong MuJoCo từ cùng state. Reward
aggregation của planner bằng tổng từng predicted reward tới sai số số học;
không thấy lệch timestep hoặc normalization. Mỗi plan được so thêm với zero
và random từ cùng simulator state. Mô hình pooled, không adapter, dùng cho
phép kiểm này để cô lập world model.

| MPC horizon | Predicted return | Actual return | Return MAE | Teacher-forced reward MAE | State MSE/bước |
|---|---:|---:|---:|---:|---:|
| 3 | 1,324 | 1,589 | 0,578 | 0,305 | 1,941 |
| 5 | 0,725 | 0,925 | 0,869 | 0,236 | 1,354 |

Mean predicted minus actual return là −0,266 (h3) và −0,200 (h5); 11/24 plans
overpredict. **Không thấy overprediction có hệ thống** trong matched replay
này, nên chưa thể khẳng định MPC đang cố ý khai thác lỗi reward. Tuy nhiên
return MAE lớn hơn teacher-forced MAE 1,9×/3,7×: lỗi dynamics khi free rollout
là nguồn lớn. MPC thực tế hơn zero-action ở 21/24 matched plans ngắn, nhưng
kết quả này không thay thế recurrent episode test Stage 3A.1, nơi MPC không
thắng zero một cách ổn định.

OOD: khoảng cách nearest train `(state,action)` được chuẩn hóa **chỉ bằng train
A/B/D**; ngưỡng 4,393 là percentile 99 trên validation A/B/D. Tất cả 16
held-out-state probes nằm ngoài ngưỡng, không probe train-like nào nằm ngoài.
MPC có **42/96** predicted plan states ngoài ngưỡng, random **27/96**, zero
**0/96**. Cờ đơn giản `|z|>6` bỏ sót tất cả trong matched replay; ở recurrent
episode trước đó đã xuất hiện một plan vượt 6 SD. Đây là dấu hiệu MPC chọn
vùng model ít được train, nhưng chưa chứng minh nó khai thác sai số để tăng
return dự đoán. Kết thúc plan không có next action nên support distance ở bước
cuối dùng lại action cuối. `mpc_plan_steps.csv` lưu từng action, reward và state error.

## Recurrence memory

Hai adapters được học từ 20 transitions **đã quan sát** của A/C. Retrieval dùng
ba transitions đầu của lần quay lại, không có nhãn. Sau đó đánh giá trên fixed
16-step probe sets độc lập, trước **mọi** online update mới:

| Fixed probes | Frozen MSE | Current adapter MSE | Retrieved adapter MSE |
|---|---:|---:|---:|
| A | **1,0412** | 1,1745 | 1,4188 |
| C | 0,2898 | 0,3972 | **0,2351** |

Retrieval chọn đúng cache A/C trên hai context nhưng **làm xấu A** trên cùng
probe distribution. Cache chưa đạt điều kiện reuse đáng tin cậy; kết quả C là
diagnostic có segmentation biết trước do evaluator tạo snapshot, không phải
online regime discovery hoàn chỉnh.

## Trạng thái trước GPU

Không tìm thấy rò rỉ C vào train, validation, checkpoint hoặc normalization;
không tìm thấy lỗi reward aggregation/timestep alignment. Diagnostic CEM giờ
lưu plan actions, predicted states và rewards để replay đúng cùng quỹ đạo.
History builder reset tại episode boundary; support threshold lấy từ A/B/D.
Không sửa NEO upstream hay artifacts Stage 1–3A.

**Kiến trúc đề xuất để thử trên GPU sau khi sửa pipeline:** residual state
dynamics MLP train bằng state loss riêng, với ridge làm baseline/linear skip;
reward predictor dùng trunk riêng; context regime nếu cần phải suy ra từ
transitions đã quan sát, không dùng oracle ID. Giữ diagnostic oracle làm upper
bound, và kiểm tra rollout calibration/OOD trước khi bật MPC dài hơn. Naive
history concat và additive physics executor chưa có bằng chứng đủ tốt để làm
world model chính. Một GPU pilot có thể phục vụ kiểm định các sửa đổi này;
**không nên chạy full GPU benchmark với pooled shared-trunk checkpoint hiện tại**.

Lệnh tái hiện trên CPU:

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
python experiments/rwt/run_stage3a_final_cpu.py --config experiments/rwt/configs/stage3a/final_cpu.yaml
```

Artifacts: `results/rwt/stage3a1_final_cpu/` gồm `summary.json`, 12
checkpoints, `training_curves.csv`, `one_step.csv`, `per_dimension.csv`,
`rollout.csv`, `matched_physics.csv`, `mpc_plans.csv`, `mpc_plan_steps.csv`,
`memory_retrieval.csv`, `memory_fixed_probes.csv` và memory NPZ.

Kiểm thử cuối: **57 RWT CPU tests passed**; `compileall` trên `src/rwt`,
`experiments/rwt` và các RWT tests; `git diff --check` không báo lỗi. CUDA
không được kiểm tra trong lượt diagnostic này; không có GPU training.
