# Member Role Report — Day 9: Multi Agent A2A

> **BẢN NHÁP** — mọi số liệu kỹ thuật dưới đây đều lấy từ artifact thật trong repo
> (`logging/trace.jsonl`, `logging/metadata.json`, `output/`) và kiểm chứng được bằng
> các lệnh ghi trong bài. Phần trong dấu `[ ]` là thông tin cá nhân và phần tự đánh
> giá — bạn phải tự điền, và sửa lại bất cứ chỗ nào không phản ánh đúng hiểu biết
> của mình trước khi nộp.

## 1. Thông tin cá nhân

| Thông tin       | Nội dung                                      |
| --------------- | --------------------------------------------- |
| Họ và tên       | [Họ và tên]                                   |
| MSSV            | [MSSV]                                        |
| Khóa/Lớp        | K3                                            |
| Vai trò chính   | Làm cá nhân — sở hữu toàn bộ pipeline         |
| Ngày hoàn thành | [YYYY-MM-DD]                                  |

## 2. Vai trò và phạm vi công việc

Bài làm cá nhân, không chia việc theo nhóm. Toàn bộ module dưới đây do một người thực hiện.

### Phần việc sở hữu

| Module/deliverable | File/hàm phụ trách | Input nhận vào | Output bàn giao | Trạng thái |
| --- | --- | --- | --- | --- |
| Tầng dữ liệu tất định | `agents/dataset.py` — `OlistDataset`, `OrderBundle` | 4 file CSV Olist | Bundle đã join, tổng tài chính, mốc giao hàng | Hoàn thành |
| Bộ luật EC_POLICY_V1 | `agents/policy.py` — `evaluate_rules`, `build_reference_output` | `OrderBundle` | Kết quả 6 luật, evidence ID, output tham chiếu | Hoàn thành |
| Sáu agent node | `agents/nodes.py` | State của case | `order_facts`, `payment_facts`, `delivery_facts`, `policy_decision`, `verifier_report`, `final_output` | Hoàn thành |
| Graph và handoff | `agents/graph.py` — `build_graph` | — | Graph LangGraph đã compile | Hoàn thành |
| Client LLM và điều tiết token | `agents/llm.py`, `agents/ratelimit.py` | Prompt | JSON đã validate, tôn trọng trần TPM | Hoàn thành |
| Runner và artifact | `run.py`, `agents/tracing.py` | 50 file `input/EC_*.json` | 50 file `output/`, `trace.jsonl`, `metadata.json` | Hoàn thành |
| Bộ kiểm chứng | `scripts/smoke_graph.py`, `score_outputs.py`, `compare_runs.py`, `make_zip.py` | output đã sinh | Báo cáo hard gate, đối chiếu baseline, file nộp | Hoàn thành |

## 3. Kết quả theo vai trò

| Nhiệm vụ | Artifact liên quan | Kết quả bàn giao | Cách xác minh |
| --- | --- | --- | --- |
| Chạy đủ 50 case qua 6 agent | `logging/trace.jsonl` (364 sự kiện) | 50/50 case, 0 case lỗi | `uv run run.py` |
| Kiểm tra hard gate | `scripts/score_outputs.py` | 50/50 case đạt, mọi ID và số tiền truy ngược được về CSV | `uv run scripts/score_outputs.py` |
| Kiểm tra wiring không tốn quota | `scripts/smoke_graph.py` | 12/12 case mẫu đúng nhánh; Verifier chặn được kết luận sai | `uv run scripts/smoke_graph.py` |
| Đóng gói bài nộp | `output.zip` | Đúng 50 JSON trong thư mục `output/`, không file lạ | `uv run scripts/make_zip.py` |

Số liệu lượt chạy thật, đọc từ `logging/metadata.json`:

- 50 case, 207 lượt gọi LLM, 133.644 token (118.312 vào / 15.332 ra)
- Thời gian model xử lý: 92,5s. Wall-clock: 1628s — **94% thời gian là nằm chờ hạn mức**, không phải tính toán
- Verifier bác Policy Agent **7 lần**; 43 case đúng ngay lần đầu; **0 case** phải dùng fallback tất định

## 4. Giải thích phần kỹ thuật đã thực hiện

### Vấn đề cần giải quyết

Mỗi case chỉ cho một `claimed_order_id` và lời than phiền của khách. Hệ thống phải tự
đối chiếu 4 bảng CSV để xác định chuyện gì thực sự xảy ra, quy trách nhiệm, tính khoản
hoàn và dẫn ra bằng chứng — **không được tin lời khiếu nại**. Ví dụ cùng một câu "giao
trễ" nhưng có thể là lỗi seller, lỗi vận chuyển, hoặc khiếu nại vô căn cứ.

### Cách triển khai

Nguyên tắc xuyên suốt: **LLM phán đoán, code tính toán.**

Thang chấm so khớp chính xác từng con số và từng ID. Một model 8B tự cộng `price` của
3 item rồi làm tròn là cách nhanh nhất mất trọn hạng mục tài chính, và một `seller_id`
bịa ra bị tính false positive. Vì vậy `dataset.py` và `policy.py` lo toàn bộ số học và
việc áp luật; agent LLM nhận dữ kiện đã tính sẵn, đưa nhận định nghiệp vụ và bàn giao.

Topology: coordinator kiểm tra order tồn tại → fan-out song song 3 agent chuyên môn
(order/seller, payment, delivery) vì chúng đọc 3 domain không giao nhau → barrier tại
Policy Agent, nơi bằng chứng ba nguồn được đối chiếu lần đầu → Verifier tất định →
nếu lệch luật thì trả ngược về Policy Agent kèm lý do, tối đa 2 vòng → finalize.

Điểm khiến các agent thực sự **bàn giao** thay vì chia sẻ một prompt khổng lồ:
**Policy Agent không có quyền đọc CSV**, nó buộc phải dùng bằng chứng ba agent kia đưa sang.

### Input, output và contract

| Thành phần | Mô tả |
| --- | --- |
| Input | `input/EC_XXX.json` — `case_id`, `opened_at`, `customer_request.claimed_order_id`, `policy_version` |
| Output | `output/EC_XXX.json` theo schema mục 6 của đề |
| Module phụ thuộc | `agents/dataset.py` (dữ liệu), `agents/policy.py` (luật), `agents/llm.py` (model) |
| Module sử dụng output | `scripts/score_outputs.py`, `scripts/make_zip.py` |
| Điều kiện lỗi cần xử lý | order không tồn tại trong CSV; order không có item row; model trả JSON hỏng; lỗi 429 vượt hạn mức |

### Cách xác minh

```bash
uv run scripts/smoke_graph.py
uv run run.py
uv run scripts/score_outputs.py
```

- **Kết quả mong đợi:** 50/50 case đạt hard gate; mọi `item_id`, `payment_id`, `seller_id` khớp một dòng có thật trong CSV; mọi số tiền cộng đúng.
- **Kết quả thực tế:** đúng như trên, 50/50.
- **Artifact/log:** `logging/trace.jsonl`, `logging/metadata.json`, `output/` — không chứa secret.

## 5. Một quyết định kỹ thuật quan trọng

- **Bối cảnh:** Ràng buộc model ≤ 10B. Câu hỏi là để LLM quyết định tới đâu.
- **Các phương án đã cân nhắc:**
  1. LLM làm hết — đọc dữ liệu thô, tự tính tiền, tự kết luận.
  2. LLM phán đoán trên dữ kiện đã tính sẵn, code lo số học, Verifier tất định chốt chặn.
- **Phương án đã chọn:** phương án 2.
- **Lý do:** thang chấm so khớp chính xác, không có điểm cho "gần đúng". Một con số model bịa là mất trọn hạng mục đó.
- **Bằng chứng quyết định phù hợp:** trace cho thấy Delivery Agent lệch với dữ liệu **26/50 lần** — nó luôn trả `logistics_provider`, kể cả khi đơn không hề trễ (18 lần) lẫn khi lỗi thuộc seller (8 lần). Order & Seller Agent lệch 6 lần. Payment Agent 0 lần. Nếu tin thẳng kết luận của Delivery Agent thì hơn nửa số case đã sai. Tầng tất định là thứ giữ cho kết quả đúng.

## 6. Một lỗi hoặc blocker đã xử lý

- **Triệu chứng/lỗi nguyên văn:**
  `Error code: 429 - Rate limit reached for model llama-3.1-8b-instant ... on tokens per minute (TPM): Limit 6000, Used 5876, Requested 1316. Please try again in 11.92s`
- **Bước tái hiện:** `uv run run.py --workers 4` trên 12 case → 5/12 case chết.
- **Nguyên nhân gốc:** gói free giới hạn theo **token mỗi phút**, không phải số request. Mỗi case tiêu ~2.600 token nên trần thực tế chỉ hơn 2 case/phút. Backoff 1-2-4s cũng quá ngắn khi Groq yêu cầu chờ ~12s, và các thread khác vẫn lao vào làm hỏng lẫn nhau.
- **Cách xử lý:** viết `agents/ratelimit.py` — token bucket sliding-window dùng chung cho mọi thread: xin trước hạn mức, ghi bù số token thật sau khi gọi, và khi dính 429 thì **đọc thời gian chờ Groq trả về rồi chặn toàn bộ thread khác** trong khoảng đó. Kèm hạ `max_tokens` 1200→400 và lược mảng chi tiết khỏi prompt của Policy Agent.
- **Cách xác minh sau khi sửa:** chạy lại 12 case → 12/12 thành công; sau đó 50/50 case, `cases_failed: 0` trong `metadata.json`.
- **Điều học được:** phải đọc đúng loại hạn mức mà provider áp, và trong môi trường nhiều thread thì backoff cục bộ từng thread là vô nghĩa — cần một bộ điều tiết dùng chung.

## 7. Hiểu biết về luồng end-to-end

> Lưu ý: các câu hỏi in sẵn trong template gốc hỏi về Crossref và vector index —
> không thuộc phạm vi lab này. Phần dưới trả lời luồng end-to-end thực tế của bài.
> **Nên hỏi lại giảng viên** xem có phải template bị dùng nhầm không.

1. **Dữ liệu đi từ ticket đến kết luận thế nào?** `run.py` đọc `input/EC_XXX.json`, lấy
   `claimed_order_id`. Coordinator tra order trong `orders`; nếu không có thì xuất bản
   bác claim với confidence thấp. Nếu có, ba agent chuyên môn chạy song song trên ba
   domain, mỗi agent ghi khoá state riêng. Policy Agent gộp ba nguồn, chọn `primary_issue`.
   Verifier áp lại luật một cách tất định và đối chiếu. Coordinator ghi file.

2. **Grounding được bảo đảm bằng gì?** Mọi ID trong output đều dựng từ một dòng CSV có
   thật, theo 5 định dạng đề cho phép. Verifier kiểm tra sự tồn tại của từng
   `item_id`, `payment_id`, `seller_id` trước khi ghi file, nên evidence bịa không lọt được.

3. **Handoff khác gì với việc gọi model nhiều lần?** Mỗi agent chỉ ghi vào khoá state của
   riêng mình và chỉ đọc khoá của agent thượng nguồn. Policy Agent không có quyền đọc CSV
   nên buộc phải dùng bằng chứng được bàn giao. Khoá `trace` dùng reducer `operator.add`
   để ba nhánh song song không ghi đè log của nhau.

4. **Verifier khác gì với việc kiểm tra schema?** Nó kiểm ba lớp: đối chiếu kết luận của
   LLM với luật tất định; kiểm tra sự tồn tại của mọi ID trong CSV; kiểm tra schema và
   các giới hạn 5/10/3/3/5. Không đạt thì trả case ngược về Policy Agent **kèm lý do bị bác**,
   tối đa 2 vòng.

5. **Căn cứ nào nói hệ thống chạy đúng?** `metadata.json`: 50/50 case, 0 lỗi, 0 fallback.
   `score_outputs.py`: 50/50 đạt hard gate với ID và số tiền đối chiếu ngược về CSV.
   Và bằng chứng mạnh nhất về vòng sửa lỗi: Verifier bác Policy Agent đúng **7 lần**, tất cả
   cùng một kiểu — model chọn `unsupported_late_claim` thay vì `canceled_order_paid` ở
   7/8 case đơn bị hủy (EC_003, 007, 015, 021, 026, 041, 045). Đó đều là case khớp đồng
   thời hai luật; model không tôn trọng thứ tự ưu tiên. Vòng sửa lỗi kéo lại được cả 7,
   tương đương 14% tổng điểm.

## 8. Cam kết của thành viên

Đánh dấu sau khi tự kiểm tra:

- [ ] Nội dung báo cáo phản ánh đúng phần việc và mức hiểu của tôi.
- [ ] Tôi có thể giải thích luồng end-to-end, không chỉ module mình phụ trách.
- [ ] Tôi không ghi "đã chạy thành công" cho phần chưa được kiểm chứng.
- [ ] Báo cáo không chứa `.env`, API key, token hoặc secret.
- [ ] Báo cáo này không phải bản sao nguyên văn của báo cáo nhóm hoặc báo cáo thành viên khác.

**Họ và tên:** [Họ và tên]
**Ngày xác nhận:** [YYYY-MM-DD]
