# Báo cáo CLSM: triển khai và kết quả chạy CPU

**Ngày:** 22-10-2026  
**Nhánh:** `duc`  
**Commit mã và kết quả:** `fd433e4` (nguồn checkpoint sạch)  
**Commit báo cáo này:** được ghi riêng trên cùng nhánh sau khi hoàn tất.

## Tóm tắt

Đã chạy được đường đi CPU của kiến trúc CLSM cho ba track COMP, EVOL và MEM trên HalfCheetah: tạo dữ liệu smoke, huấn luyện approximate generalized EM, lưu và replay checkpoint, chạy evaluator C1/C2/C3, và tổng hợp kết quả có kiểm tra provenance. Bộ kiểm thử phần mềm đạt **53 passed**.

Các kết quả hiện tại **chưa đạt nghiệm thu khoa học đầy đủ**. Đây là smoke với 32/16/16 transition và một seed/world. C1 nghiêng về pooled MLP; C2 chỉ dương tại một horizon; C3 đổi dấu theo ngân sách và vi phạm ngưỡng mass-loss. Không dùng các số này để kết luận CLSM tổng quát hóa hay học được luật tiến hóa.

## Môi trường và protocol đã chạy

- Windows, Python 3.11.9, PyTorch 2.13.0 CPU, Gymnasium 1.3.0, MuJoCo 3.3.7, NumPy 2.4.6, SciPy 1.17.1.
- CUDA không khả dụng cho PyTorch. Bộ smoke dùng CPU.
- Mỗi track dùng HalfCheetah-v5, seed 41, 32 train / 16 validation / 16 test transition, world length 8, một vòng approximate GEM và tối đa 128 thành phần posterior.
- Chọn checkpoint theo validation prequential NLL. Checkpoint ghi `source_dirty=false`, commit `fd433e4828240fe0ec8570d3e026790b9a9da3f3`, fingerprint mã `1ce778f8323b252814b26985568152343d4956ffd580b9194eb47a68b11f17a0`.
- Dữ liệu train COMP có SHA-256 `7d567ff2ad5918a3f8d415b0d6b1e146bfafa9bce2ad12ec405795a250035627`; train EVOL/MEM có SHA-256 `9c212ae134d63b5b36703623d14af4f0a74be933397bf03425d0ab2d7da528c7`. Hash validation và reference được ghi trong manifest từng track.

## Kết quả thực thi CPU

| Track / phép kiểm | Kết quả | Đọc kết quả |
|---|---:|---|
| COMP validation prequential NLL | 9.4406775463 | Smoke, không phải full-data fit. |
| EVOL validation prequential NLL | 9.4454564488 | Smoke, không phải full-data fit. |
| MEM validation prequential NLL | 9.4454564488 | Smoke, không phải full-data fit. |
| Maximum dropped mass khi train | 6.66e-16 | Qua gate số học trong các world 8 bước. |
| Replay checkpoint COMP / EVOL / MEM | 9.4406775463 / 9.4454564488 / 9.4454564488; dropped mass tối đa 4.44e-16 | Load và replay checkpoint sạch thành công. |
| Pooled MLP validation NLL | 9.1431884104 | Tốt hơn CLSM trên split smoke này. |
| Test suite `python -m pytest tests/clsm -q` | 53 passed trong 12.11 giây | Các kiểm tra đơn vị, synthetic, interface và checkpoint hiện có đều đạt. |

## Kết quả theo ba câu hỏi nghiên cứu

### C1 — compositional generalization

Evaluator dùng cùng fixed probe bank và bốn held-out structural codes 3, 5, 6, 7, với bốn transition thích nghi công khai. Hiệu số được báo cáo là `ES_pooled - ES_CLSM`:

| Held-out code | 3 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|
| `ES_pooled - ES_CLSM` | -0.36746 | -0.46912 | -0.32320 | -0.28461 |

Số âm nghĩa là pooled MLP có Energy Score thấp hơn, do đó tốt hơn trong phép thử này. Cả bốn code đều nghiêng về pooled. Chỉ có một seed và một evaluator world; không có khoảng tin cậy. **C1 chưa đạt.**

### C2 — dự báo phân phối luật tương lai

So sánh `ES_frozen - ES_CLSM` tại horizon 0, 1, 8, 24, 64 lần lượt là:

| Horizon | 0 | 1 | 8 | 24 | 64 |
|---|---:|---:|---:|---:|---:|
| Skill | 0.00000 | -0.00415 | +0.04824 | -0.00079 | -0.01935 |

Chỉ horizon 8 dương; các horizon còn lại không vượt frozen-current. **Kết quả này chưa chứng minh CLSM học được law evolution và C2 chưa đạt.**

### C3 — lợi ích và rủi ro của memory

Hiệu số là `ES_scratch - ES_reuse`; số dương nghiêng về reuse:

| Adaptation budget | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| Gain | ~0 | 0.00000 | -0.02245 | +0.02797 | +0.01469 |
| Reuse dropped mass tối đa | 2.22e-16 | 2.22e-16 | 4.201% | 4.681% | 4.854% |
| Scratch dropped mass tối đa | 0 | 2.22e-16 | 3.33e-16 | 3.472% | 4.203% |

Gain đổi dấu theo budget. Từ budget 16 trở đi, ít nhất một filter vượt ngưỡng dropped mass 0.1%; ở budget 64 cả hai vượt ngưỡng. Scratch chưa được matched compute. **Inference degraded; không tính đây là bằng chứng đạt C3.** Thời gian đo ở lượt 64-budget run: reuse 171.48 giây, scratch 162.62 giây.

## Đối chiếu với yêu cầu trong hai file

| Yêu cầu | Trạng thái |
|---|---|
| Kiểm tra toán học/cấu trúc: causal timing, discrete Weibull và age, sparse structural edits, reset không làm mới structural age, Gaussian update/support/mass, memory scratch/reuse, leakage, checkpoint/resume, terminal/truncation | Bộ test hiện có đạt 53/53. Đây là kiểm tra code và synthetic; không chứng minh hiệu quả khoa học. |
| C1: held-out combinations, pooled/history/switching/factorization ablation và oracle upper bound | Chỉ có C1 smoke so với pooled MLP. History MLP, switching experts, factorization ablation và oracle upper bound chưa có trong panel. Kết quả pooled tốt hơn CLSM trên bốn mask smoke. |
| C2: future-law Energy Score, same-belief frozen, stationary/geometric controls và switch calibration | Có forecast h=0/1/8/24/64 so với frozen-current cho một world. Stationary/geometric-duration controls và calibration confirmatory chưa chạy. Chỉ h=8 dương. |
| C3: reuse so scratch với cùng data/budget, memory risk, negative transfer, recovery và compute | Có diagnostic reuse/scratch theo budget. Chưa matched-compute; long-world mass gate thất bại; chưa đủ seed/world để ước lượng paired uncertainty. |
| Secondary action-value/ranking/return | Chưa chạy; không suy ra từ next-state hoặc law score. |
| Full benchmark | Chưa chạy. Protocol yêu cầu 150k/30k/30k transitions mỗi environment/track, train seeds 41–45 và evaluator worlds 801–820. Full collection cần nominal SAC controller fit ít nhất 100k bước. |
| Môi trường | HalfCheetah smoke đã chạy. Swimmer không có damping dương trên controlled DOFs theo schema đã cài, nên intervention damping không hợp lệ. ManiSkill chưa được cài/kiểm chứng PD adapter. |

Vì vậy, **phần mềm CPU và pipeline diagnostic chạy được; kết quả C1/C2/C3 chưa đạt yêu cầu confirmatory của hai file**. Gate long-world mass-loss là blocker số học cần giải quyết trước khi chạy full world. Baseline panel và controller/data full budget cũng còn thiếu.

## Artifact và tái lập

Artifact kết quả sinh trong `results/rwt/clsm_cpu_verification/` và bị Git ignore nên chỉ nằm trên máy chạy. Báo cáo này được đưa lên GitHub; checkpoint, dữ liệu, predictions và diagnostics thô không nằm trong commit.

- Tổng hợp: `results/rwt/clsm_cpu_verification/diagnostic_summary.json` (14 records / 14 per-slice effects, khoảng tin cậy để null).
- Posterior và luật tiến hóa: `inference_diagnostics.csv` và `evolution_diagnostics.csv` trong từng track/seed directory.
- Raw C1/C2/C3 scores, predictions, checkpoint hash và probe hash nằm trong thư mục evaluator tương ứng.
- Mã sinh diagnostics và lọc kết quả evaluator cũ: commit `fd433e4`; tài liệu audit chi tiết: `docs/clsm_implementation_audit.md`.

Các lệnh chính:

```powershell
python -m pytest tests/clsm -q
python experiments/rwt/clsm/build_datasets.py --env HalfCheetah-v5 --track comp --config experiments/rwt/clsm/configs/cpu_smoke.yaml --output-dir results/rwt/clsm_cpu_verification
python experiments/rwt/clsm/run_train.py --env HalfCheetah-v5 --track comp --config experiments/rwt/clsm/configs/cpu_smoke.yaml --output-dir results/rwt/clsm_cpu_verification --device cpu
python experiments/rwt/clsm/summarize.py --results-dir results/rwt/clsm_cpu_verification
```

EVOL và MEM dùng lần lượt `configs/evol_cpu_smoke.yaml` và `configs/mem_cpu_smoke.yaml`. Không dùng cấu hình smoke để báo cáo như full benchmark.
