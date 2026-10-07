# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Lê Phước Tiến  
**MSSV**: 2A202602616  
**Ngày**: 07/10/2026  
**Tier**: `T4`  
**Base model**: `unsloth/Qwen3.5-4B`  
**GPU thực tế**: `NVIDIA T4 16GB`

> Mọi con số dưới đây được lấy từ các artifact trong `results/`.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 256 — p95 đo được là 98 tokens (`results/token_stats.json`) |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | `max_steps = 30` |

Thống kê độ dài token trên 250 mẫu cho thấy mean = 93.1, p50 = 93, p95 = 98, p99 = 100 và maximum = 101 tokens. Tôi sử dụng `max_length = 256`, lớn hơn đáng kể p95 và maximum của tập dữ liệu, nên đủ không gian cho các sequence mà không cần chọn context length quá lớn.

**Template có giữ khối `<think>` không?** Có.

Kết quả `results/template_check.json` cho thấy `open_tag_present = true`, `body_present = true` và `ok = true`. Verdict của template check là `reasoning preserved — safe to train on traces`. Vì vậy tôi giữ nguyên template và không cần thực hiện bước sửa reasoning template.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Mask mode được sử dụng là `assistant-only`. Có 39 supervised tokens trên tổng cộng 94 tokens của mẫu kiểm tra. Điều này tương ứng supervised fraction bằng 0.4149.

Đoạn đầu của phần được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Kết quả mask proof xác nhận `answer_is_supervised = true` và `question_is_masked = true`. Như vậy loss được tập trung vào phần output của assistant thay vì bắt model học lại system prompt và câu hỏi của user.

---

## 3. Ba baseline (NB2/NB5)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3214.2 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.000 | 1010.9 |
| (c) LoRA fine-tune | **0.970** | **0.6778** | **1.000** | 1413.2 |

**(b) có thật sự mạnh hơn (a) không?** Có.

Chỉ riêng optimized prompt đã làm target score tăng từ 0.000 lên 0.765, đồng thời format score tăng từ 0.000 lên 1.000. Regression score vẫn giữ nguyên ở 0.7911. Điều này cho thấy trước khi fine-tune, prompt engineering đã giải quyết được một phần lớn yêu cầu của tác vụ.

Tôi không sửa `OPTIMIZED_PROMPT`; gatekeeper xác nhận `baseline (b) prompt unmodified`. Điều này giúp baseline (b) vẫn là mốc so sánh chuẩn của lab thay vì cố tình làm prompt yếu đi để khiến fine-tune có vẻ tốt hơn.

LoRA tiếp tục cải thiện target từ 0.765 lên 0.970, tương ứng tăng 0.205. Tuy nhiên sự cải thiện trên target đi kèm với regression score giảm từ 0.7911 xuống 0.6778.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | target (NB5 §4) | s | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6257 | **0.970** | 393.3 | 8.78 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | **0.5368** | **0.970** | 263.2 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.000** | 395.4 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.940** | 462.2 | **3.86** |

### 4.1 — `attn_only`

`attn_only` và `correct` hòa nhau trên target evaluation với score đều bằng 0.970. Tuy nhiên thứ tự theo training loss lại khác: `attn_only` đạt loss 0.5368, thấp hơn `correct` là 0.6257. Nếu chỉ nhìn training loss, tôi có thể kết luận sai rằng `attn_only` tốt hơn, trong khi task metric cho thấy chúng hòa nhau.

Đáng chú ý, rank của `attn_only` đã được tăng lên 283 để số trainable parameters gần bằng `correct`: 32,456,704 so với 32,464,896. Kết quả cho thấy tăng rank để bù số tham số không tự động tạo ra lợi thế trên task metric. Vì vậy rank và số lượng trainable parameters không nên được xem độc lập với vị trí gắn adapter. Adapter placement và metric của tác vụ thực tế quan trọng hơn việc chỉ tối ưu training loss.

### 4.2 — `wrong_lr`

`wrong_lr` chỉ thay learning rate từ `1e-4` xuống `1e-5`, trong khi giữ text-linear placement, rank 16 và cùng số trainable parameters với cấu hình `correct`. Final training loss tăng mạnh từ 0.6257 lên 1.5702 và target score giảm từ 0.970 xuống 0.000.

Nếu chỉ nhìn loss mà không biết learning rate, tôi có thể kết luận sai rằng model, dataset hoặc LoRA configuration có vấn đề. Thực tế thí nghiệm kiểm soát này cho thấy chỉ một thay đổi về learning rate đã tạo ra khác biệt rất lớn. Với step budget chỉ 30 bước, learning rate `1e-5` không giúp adapter học đủ nhanh, trong khi `1e-4` hoạt động tốt hơn rõ rệt trong thiết lập của lab.

### 4.3 — `qlora`

QLoRA giảm peak VRAM từ 8.78 GB xuống 3.86 GB, tức giảm khoảng 4.92 GB, tương đương khoảng 56% so với cấu hình `correct`. Đổi lại, thời gian train tăng từ 393.3 giây lên 462.2 giây, final loss tăng từ 0.6257 lên 0.7058 và target score giảm từ 0.970 xuống 0.940.

Trong thí nghiệm này, QLoRA mang lại lợi ích rõ ràng về memory nhưng không cải thiện chất lượng hay tốc độ. Vì GPU T4 vẫn chạy được cấu hình LoRA 16-bit với peak VRAM 8.78 GB, tôi không nhận được lợi ích thực tế đủ lớn để bù cho target score thấp hơn và thời gian train dài hơn. Do đó số đo của tôi nhìn chung ủng hộ khuyến nghị không cần sử dụng QLoRA cho model này khi GPU có đủ VRAM, mặc dù QLoRA vẫn có giá trị trong môi trường bị giới hạn bộ nhớ.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`

`target Δ = +0.205` · `regression Δ = -0.113` · `valid_trace_rate = 0.00`

Fine-tuning tạo ra cải thiện rõ rệt trên tác vụ mục tiêu. Target score tăng từ 0.765 của optimized-prompt baseline lên 0.970 sau LoRA, tương ứng mức tăng 0.205. Format score vẫn giữ ở 1.000. Tuy nhiên regression score giảm từ 0.7911 xuống 0.6778, tức giảm khoảng 0.1133. Mức giảm này lớn hơn nhiều so với tolerance 0.020 của regression gate, vì vậy verdict cuối cùng là `FAILED`.

Kết quả này cho thấy fine-tuning đã làm model chuyên biệt hóa rất tốt cho bài toán phân loại ticket nhưng đồng thời gây suy giảm đáng kể khả năng trên regression set. Vì vậy target score cao không đủ để kết luận model đã tốt hơn toàn diện. Nếu triển khai ngay phiên bản này, hệ thống có thể xử lý tác vụ triage tốt hơn nhưng đánh đổi quá nhiều general capability. Hướng cải thiện tiếp theo hợp lý là bổ sung một lượng nhỏ replay data từ dữ liệu tổng quát vào quá trình fine-tuning, sau đó đánh giá lại cả target và regression gate.

---

## 6. Định tính — bắt buộc có cả ca THUA

`qualitative.json` lưu các trường hợp theo `ft_score`. Artifact hiện tại không lưu prediction từng mẫu của baseline (b), vì vậy phần dưới phân tích trực tiếp các trường hợp fine-tune tốt và chưa hoàn hảo thay vì tự suy diễn prediction không được lưu.

| # | Ticket (rút gọn) | Nhãn/tín hiệu chính | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Ốp lưng điện thoại — “Giá bao nhiêu” | `hoi_thong_tin` | Không lưu per-sample prediction | score 1.00 | ✅ FT xử lý đầy đủ |
| 2 | Ốp lưng điện thoại — “Sai màu” | `san_pham_loi` | Không lưu per-sample prediction | score 1.00 | ✅ FT xử lý đầy đủ |
| 3 | Bình giữ nhiệt — “Chưa thấy tiền” | `hoan_tien` | Không lưu per-sample prediction | score 0.75 | ❌ FT chưa hoàn hảo |
| 4 | Nồi chiên không dầu — “Thiếu phụ kiện” | `san_pham_loi` | Không lưu per-sample prediction | score 0.75 | ❌ FT chưa hoàn hảo |
| 5 | Áo khoác gió — “Bị lỗi” | `san_pham_loi` | Không lưu per-sample prediction | score 0.75 | ❌ FT chưa hoàn hảo |

Có một mẫu chung đáng chú ý: các trường hợp kém nhất vẫn nhận diện đúng tín hiệu intent chính như `hoan_tien` hoặc `san_pham_loi`, nhưng chỉ đạt field accuracy 0.75. Điều đó có nghĩa lỗi tập trung ở một trong các field còn lại của JSON triage thay vì model hoàn toàn không hiểu yêu cầu.

Trong 50 target examples, 44 mẫu đạt `ft_score = 1.00` và 6 mẫu đạt `0.75`. Trung bình vì vậy là:

`(44 × 1.00 + 6 × 0.75) / 50 = 0.970`

Kết quả định tính do đó nhất quán với target score 0.970 trong `verdict.json`. Các ca thua cho thấy model vẫn cần cải thiện khả năng phân biệt những thuộc tính phụ như urgency hoặc sentiment trong các ticket ngắn hoặc có tín hiệu ngôn ngữ không rõ ràng.

---

## 7. Kết luận & điều tôi học được

Kết quả của lab cho thấy fine-tuning không nên được đánh giá chỉ bằng training loss hoặc metric của tác vụ mục tiêu. Optimized prompt đã tạo ra bước cải thiện rất lớn trước khi fine-tune: target score tăng từ 0.000 lên 0.765 mà không làm giảm regression score. LoRA tiếp tục tăng target score lên 0.970, nhưng regression giảm từ 0.7911 xuống 0.6778. Vì mức regression -0.1133 vượt xa tolerance 0.020, tôi **không nên deploy phiên bản fine-tune hiện tại** dù target accuracy rất cao.

Các thí nghiệm NB4 cũng cho thấy không có một chỉ số đơn lẻ quyết định chất lượng model. `attn_only` có training loss thấp hơn `correct` nhưng target score chỉ hòa. `wrong_lr` cho thấy learning rate không phù hợp có thể khiến kết quả tác vụ sụp từ 0.970 xuống 0.000. QLoRA giảm khoảng 56% peak VRAM nhưng target thấp hơn và thời gian train dài hơn. Ngoài adapter placement và learning rate, mask cũng là điều kiện quan trọng vì nó đảm bảo loss tập trung vào output của assistant. Với kết quả hiện tại, bước cải thiện quan trọng tiếp theo là chất lượng và thành phần dữ liệu, đặc biệt bổ sung replay data để giảm regression mà vẫn giữ mức target performance cao.

**Ba điều tôi học được:**

1. Prompt engineering phải được đo trước fine-tuning. Trong thí nghiệm này, chỉ optimized prompt đã tăng target từ 0.000 lên 0.765 mà không gây regression.
2. Training loss không phải metric cuối cùng. `attn_only` có loss 0.5368 tốt hơn `correct` 0.6257 nhưng cả hai đều đạt target 0.970.
3. Fine-tuning có thể làm model chuyên biệt tốt hơn nhưng làm giảm general capability. LoRA tăng target thêm 0.205 nhưng regression giảm 0.1133, khiến model không vượt deployment gate.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** thêm khoảng 1–5% replay data đại diện cho general capability vào training set, giữ nguyên evaluation sets và các hyperparameter chính, sau đó train lại cùng step budget. Tôi sẽ so sánh target delta và regression delta với run hiện tại để kiểm tra liệu replay data có thể giữ target gần 0.970 nhưng đưa regression degradation về trong tolerance 0.020 hay không.

---

## Phụ lục — thưởng đã làm
- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link: https://huggingface.co/phtien/lab21-2A202602616-lora