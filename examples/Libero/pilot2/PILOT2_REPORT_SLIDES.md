# Pilot 2 — Goal-Conditioned Funnel Pullback trên GR00T-N1.5

Nội dung đề xuất cho 7 slide. Các dòng “Placeholder hình” là hướng dẫn chèn ảnh, không cần đưa vào phần chữ của slide.

---

## Slide 1 — Mục tiêu và dữ liệu

**Câu hỏi:** Goal image có giúp VLA điều khiển robot trở lại quỹ đạo hợp lệ sau khi bị lệch không?

- Base model: **NVIDIA GR00T-N1.5-3B**; benchmark: **LIBERO-Long**, 10 task (`libero_10`).
- Train và eval trên cùng bộ task: **in-domain fine-tuning**.
- Cache thực tế: **379 demonstration → 349 train + 30 reference**; giữ riêng 3 demo/task để lấy goal khi eval.
- Reference demo không tham gia cập nhật trọng số hoặc tính normalization.
- Hai mục tiêu: học một policy hỗ trợ cả language/goal conditioning; kiểm tra khả năng **perturb-and-recover**.

> **Placeholder hình:** Một task LIBERO, gồm ảnh quan sát hiện tại và ảnh milestone từ reference demo.

*Phạm vi: pilot chẩn đoán; chưa dùng đủ 500 demo theo thiết kế ban đầu. Chưa tích hợp uncertainty gate của Pilot 1.*

---

## Slide 2 — Training dual-mode

**Input:** camera ngoài + camera cổ tay + proprioception; bổ sung goal image khi bật goal conditioning.

- Frozen Eagle vision/language backbone; train projector, DiT action head và các embedding điều kiện mới.
- Goal image đi qua một nhánh encode riêng, dùng chung Eagle; nối goal tokens vào DiT conditioning, thêm **goal-view embedding + mode token**.
- Goal training: **50% cùng demo**, **50% demo khác cùng task**, căn chỉnh bằng tiến độ tương đối; lấy ảnh tương lai **16–32 bước**.
- Với cross-demo samples: thêm state noise **σ = 0.02** trong đơn vị state gốc, giữ nguyên action label.

| Tỷ lệ mẫu train | Language | Goal image | Mode ID |
|---|---|---|---:|
| 40% | Có | Null | 0 |
| 40% | Có | Có | 1 |
| 20% | Prompt rỗng | Có | 1 |

**Baseline đối chứng:** 100% language-only, cùng train split và training budget.

**Cả hai:** 5,000 optimizer steps ≈ **6.87 epochs**; effective batch 128; AdamW, LR 10⁻⁴, cosine schedule, warmup 5%; seed 42. Checkpoint eval: step 5,000.

> **Placeholder hình:** `Current views + language → Eagle → current tokens` và `Goal image → cùng Eagle → goal tokens`; ghép với mode token → DiT → action chunk.
>
> Nếu cần training curves: [training_curves.png](../../../outputs/pilot2/run_2k_20261026/training_review/training_curves.png).

*40/40/20 là recipe của pilot, lấy cảm hứng từ modality masking của Octo.*

---

## Slide 3 — Eval 1: Standard rollout và multi-mode sanity

- **10 task × 5 initial states = 50 rollout/condition**; cùng initial states giữa các condition.
- Mỗi query dự đoán và thực thi **16 actions**; **4 Flow Matching Euler steps**; tối đa 1,000 control steps/rollout.
- Goal retrieval: nearest frame theo cosine similarity của frozen Eagle features, **chỉ trong 3 reference demo cùng task**, rồi lấy milestone **+16 bước**; cập nhật mỗi chunk.

| Policy / inference mode | Task success |
|---|---:|
| Vanilla baseline — language-only | **47/50 = 94%** |
| Dual — language-only | **45/50 = 90%** |
| Dual — goal-only | **39/50 = 78%** |
| Dual — language + goal | **46/50 = 92%** |

**Nhận xét:** Dual hoạt động được ở cả ba mode; thêm goal chỉ tăng 1/50 episode so với dual language-only. Chưa có bằng chứng về cải thiện ổn định hoặc semantic goal grounding.

- Probe 1,000 states, cùng sampling noise: goal/null sensitivity **mean 5.37%, median 1.83%** → action có thay đổi, nhưng chưa chứng minh thay đổi đúng hướng.
- Model latency language-only / language+goal: **72 / 97 ms/chunk**; goal retrieval thêm khoảng **110 ms/query** trong implementation hiện tại.

> **Placeholder hình:** [screening_overview.png](../../../outputs/pilot2/screen_5k_taskfix/analysis/screening_overview.png).

*Dùng kết quả đã sửa task mapping. Goal-only vẫn được benchmark cung cấp task ID cho retrieval. Goal/null probe thay đổi cả mode token và goal input.*

---

## Slide 4 — Eval 2: Thiết kế perturb-and-recover

**Perturbation:** Cộng Cartesian command offset trong **4 control steps**, bắt đầu tại **t = 25 hoặc t = 60**.

- **1×:** vector offset tổng yêu cầu có norm **5–8 cm**; **4×:** cùng vector gốc, nhân bốn → **20–32 cm**.
- Đây là nhiễu vào **lệnh điều khiển**, không phải teleport hay lực đẩy ngoài; displacement thực phụ thuộc controller, clipping và tiếp xúc.
- Trước và trong perturbation, cả A/B/C đều dùng **dual language-only**.

| Strategy sau perturbation | Cách điều khiển |
|---|---|
| **A — Language-only** | Tiếp tục dual language-only đến hết episode |
| **B — Naive action copy** | Retrieve nearest reference state, chép 1 expert chunk **16 bước**, rồi về language-only |
| **C — Goal pullback** | Retrieve milestone +16; giữ cùng goal cho **2 chunks = 32 bước**, rồi về language-only |

- **5 trials/task**, initial-state indices 5–9; cùng initial state, flow seed và disturbance giữa A/B/C. Trạng thái simulator tại đầu recovery đã được kiểm tra khớp.
- Tổng: **10 task × 5 trials × 2 triggers × 3 strategies × 2 mức = 600 rollout**.

> **Placeholder hình:** Timeline `Language-only → Perturb 4 bước → A / B / C → Language-only`; hoặc chèn clip rollout ở cửa sổ perturb/recovery.

---

## Slide 5 — Kết quả chính: Goal pullback chưa vượt language-only ổn định

**Recovery rate = tỷ lệ hoàn thành toàn task sau perturbation. Mỗi ô có 50 rollout.**

| Mức perturb | Trigger | A: Language-only | B: Action copy | C: Goal pullback | C − A |
|---|---:|---:|---:|---:|---:|
| 1× | 25 | 84% | 80% | **88%** | +4 pp |
| 1× | 60 | **88%** | 74% | **88%** | 0 pp |
| 4× | 25 | **84%** | 52% | 78% | −6 pp |
| 4× | 60 | **86%** | 80% | 84% | −2 pp |

- **C tốt hơn B ở cả bốn điều kiện**, nhưng **chưa cải thiện ổn định so với A**.
- Action copy dễ gây hại: ở 4×/t25, B đạt **52%**, trong khi A đạt **84%**.
- Language-only vốn khá robust trong protocol hiện tại; thắng action copy chưa đủ để xác nhận lợi ích của goal conditioning.

> **Placeholder hình chính:** [recovery_success.png](../../../outputs/pilot2/recovery_scale_comparison_20261026/recovery_success.png).

*pp = điểm phần trăm. Đây là kết quả pilot với 5 initial states/task và một training seed; chưa kết luận ưu thế có ý nghĩa thống kê.*

---

## Slide 6 — Chẩn đoán: Goal ảnh hưởng yếu; 4× bị clipping

**1. Chưa thấy contraction riêng biệt của goal pullback**

- Đo khoảng cách EE tới tập reference cùng task tại **0 / 8 / 16 / 32 bước sau khi perturbation kết thúc**.
- Đường A và C gần trùng nhau; tại +32, chênh lệch khoảng cách trung bình chỉ **0.01–0.12 cm** giữa các điều kiện.
- Từ cùng trạng thái đầu recovery và cùng flow seed: median chênh lệch Cartesian commands A/C ở chunk đầu khoảng **2.3–3.4%**; median khoảng cách EE giữa hai rollout sau 16 bước khoảng **2 mm**.

**2. Requested perturbation không bằng displacement thực**

| Trigger | Offset tổng yêu cầu TB, 1× → 4× | EE motion trong 4 bước, 1× → 4× | Trial bị clipping ở 4× |
|---|---:|---:|---:|
| 25 | 6.74 → 26.98 cm | 3.02 → 5.06 cm | 76% |
| 60 | 6.74 → 26.98 cm | 2.56 → 4.36 cm | 70% |

**Diễn giải:** Kết quả phù hợp với ảnh hưởng thực tế còn yếu của goal; chưa tách được việc policy ít dùng goal khỏi việc milestone retrieval chưa phù hợp.

> **Placeholder hình:** [paired_contraction.png](../../../outputs/pilot2/recovery_scale_comparison_20261026/paired_contraction.png). Có thể chỉ chọn hai panel 4× để slide gọn hơn.

*Khoảng cách EE là proxy hình học, không phải semantic sub-goal success hay bằng chứng Lyapunov. EE motion trong bảng gồm cả chuyển động nominal của policy và đáp ứng tiếp xúc.*

---

## Slide 7 — Kết luận và bước tiếp theo

**Đã làm được**

- Train một dual-mode policy và baseline đối chứng trên cùng dữ liệu/budget.
- Hoàn tất **200 standard rollout + 600 perturb-and-recover rollout**, cùng sensitivity probes và latency measurement.
- Policy chạy được với goal-only; open-loop expert action copy kém robust hơn closed-loop policy trong các điều kiện đã thử.

**Kết luận hiện tại**

> **Chưa có bằng chứng goal image tạo ra lợi ích funnel pullback vượt language-only một cách ổn định.**

- Khác biệt A/C nhỏ về hành động và quỹ đạo; tăng perturb lên 4× chưa làm xuất hiện lợi thế rõ của C.
- Giới hạn: cache 379 demo; 5 trials/task; một training seed; chưa có **0× control** khớp initial states và lịch query để đo riêng tác động của perturbation.

**Bước tiếp theo ưu tiên**

1. Từ cùng post-perturbation state và cùng noise, so sánh **null goal / retrieved goal / same-task wrong-phase goal / distractor goal**; kiểm tra goal có đúng phase thao tác không.
2. Bổ sung 0× control với cùng lịch query; nếu cần thử stress mạnh hơn, ưu tiên perturb tại **grasp/carry events** và đo displacement thực.
3. Dùng kết quả chẩn đoán để quyết định sửa retrieval/conditioning hoặc tăng training budget.

---

## Nguồn số liệu và ảnh — không cần đưa thành slide

- [Training review](../../../outputs/pilot2/run_2k_20261026/training_review/review.md): 5,000 steps, 6.869 epochs, matched recipes.
- [Corrected screening review](../../../outputs/pilot2/screen_5k_taskfix/analysis/review.md): 4 inference conditions, sensitivity, latency.
- [Recovery comparison review](../../../outputs/pilot2/recovery_scale_comparison_20261026/review.md): 1×/4× results, pairing audit và diagnostics.
- [Recovery summary CSV](../../../outputs/pilot2/recovery_scale_comparison_20261026/summary.csv).
- [Disturbance delivery CSV](../../../outputs/pilot2/recovery_scale_comparison_20261026/disturbance_delivery.csv).
- Video ví dụ 1×: [A/t25/task00, slow motion](../../../outputs/pilot2/recovery_5k_5trials/preview/strategy_A_t25_task00_init005_slow.mp4).
- Minh họa scale từ smoke test: [1×/2×/4× side-by-side](../../../outputs/pilot2/perturb_scale_smoke_20261026/scale_comparison_slow.mp4). Chỉ dùng minh họa; **2× chưa có full recovery evaluation**.

**Lưu ý khi báo cáo:** Không dùng kết quả goal retrieval trước khi sửa task mapping. Không so trực tiếp với con số RA-VLA công bố để tuyên bố cải thiện vì training/eval protocol khác nhau. Initial states lấy từ benchmark; chưa xác minh chúng hoàn toàn không xuất hiện trong demonstrations. Chưa có nhãn semantic cho từng goal frame, nên không trình bày EE proximity như semantic sub-goal success.
