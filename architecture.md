# Kiến trúc hệ multi-agent — EC Dispute Resolution

Framework: **LangGraph** · Model mỗi agent: **`llama-3.1-8b-instant`** (8B ≤ 10B, provider Groq)
Khai báo model: [`agents/llm.py`](agents/llm.py) → `MODEL_NAME`, đồng thời ghi vào `logging/metadata.json`.

## 1. Nguyên tắc thiết kế: LLM phán đoán, code tính toán

Thang chấm so khớp chính xác từng con số tiền và từng ID. Một model 8B tự cộng
`price` của 3 item rồi làm tròn là cách nhanh nhất để mất trọn 20% điểm hạng mục
financial resolution, và một `seller_id` bị bịa ra bị tính là false positive.

Vì vậy ranh giới trách nhiệm được đặt như sau:

| Việc | Do ai làm |
| --- | --- |
| Đọc CSV, join bảng, cộng tiền, so mốc thời gian | `agents/dataset.py` — code thuần |
| Đánh giá 6 điều kiện luật EC_POLICY_V1 | `agents/policy.py` — code thuần |
| Nhận định nghiệp vụ trên từng domain, giải thích căn cứ | 3 agent LLM chuyên môn |
| Chọn `primary_issue` và đặt `confidence` | Policy Agent (LLM) |
| Chặn kết luận sai trước khi ghi file | Verifier Agent — code thuần |

Agent LLM **không bao giờ được cấp quyền tự sinh số**. Mọi con số trong output
đều truy ngược được về một dòng CSV cụ thể.

## 2. Sơ đồ agent và luồng handoff

```
                        ┌─────────────────────┐
                        │ coordinator_intake  │  (không gọi LLM — chỉ định tuyến)
                        └──────────┬──────────┘
                    order tồn tại? │
                  ┌────────────────┴────────────────┐
                 có                                không
                  │                                  │
            ┌─────▼─────┐                   ┌────────▼─────────┐
            │ dispatch  │                   │ coordinator_abort│
            └─────┬─────┘                   └────────┬─────────┘
      ┌───────────┼───────────┐                      │
      ▼           ▼           ▼                     END
┌──────────┐ ┌─────────┐ ┌──────────┐
│  Order & │ │ Payment │ │ Delivery │   ← 3 agent chạy SONG SONG
│  Seller  │ │  Agent  │ │  Agent   │     (3 domain dữ liệu độc lập)
└─────┬────┘ └────┬────┘ └────┬─────┘
      └───────────┼───────────┘
                  ▼                        ← barrier: chờ đủ cả 3 bằng chứng
          ┌───────────────┐
          │ Policy Agent  │◄──────────────┐
          └───────┬───────┘               │
                  ▼                       │ vòng sửa lỗi
          ┌───────────────┐               │ (tối đa 2 lần,
          │ Verifier Agent│───────────────┘  kèm lý do bị bác)
          └───────┬───────┘
                  │ đạt, hoặc hết vòng sửa
                  ▼
       ┌──────────────────────┐
       │ coordinator_finalize │
       └──────────┬───────────┘
                 END
```

Ba agent chuyên môn chạy song song vì chúng đọc ba domain không giao nhau.
`policy_agent` là **điểm gộp**: LangGraph chờ đủ cả ba nhánh mới kích hoạt, và đây
là nơi bằng chứng từ ba nguồn được đối chiếu với nhau lần đầu tiên.

## 3. Vai trò, quyền truy cập dữ liệu và contract

| Agent | Đọc bảng nào | Đọc khóa state nào | Ghi khóa state nào | Gọi LLM |
| --- | --- | --- | --- | :---: |
| **Coordinator (intake)** | `orders` (chỉ kiểm tra tồn tại) | `claimed_order_id` | `order_found`, `fatal_error` | ✗ |
| **Order & Seller Agent** | `orders`, `order_items`, `sellers` | — | `order_facts` | ✓ |
| **Payment Agent** | `order_payments`, `order_items` | — | `payment_facts` | ✓ |
| **Delivery Agent** | `orders`, `order_items` | `customer_message` | `delivery_facts` | ✓ |
| **Policy Agent** | — (chỉ dùng bằng chứng đã bàn giao) | cả 3 khóa `*_facts`, `verifier_report` | `policy_decision` | ✓ |
| **Verifier Agent** | toàn bộ (để đối chiếu ID) | `policy_decision` | `verifier_report`, `repair_count` | ✗ |
| **Coordinator (finalize)** | — | `verifier_report` | `final_output` | ✗ |

Mỗi agent chỉ ghi vào khóa của riêng mình. Đây là điều khiến các agent thực sự
**bàn giao** công việc thay vì cùng chia sẻ một prompt khổng lồ — Policy Agent
không có quyền đọc CSV, nó buộc phải dùng bằng chứng ba agent kia đưa sang.

Khóa `trace` dùng reducer `operator.add` để ba nhánh song song không ghi đè log của nhau.

## 4. Cơ chế kiểm chứng

Verifier là **chốt chặn tất định**, không gọi LLM. Nó kiểm ba lớp:

1. **Đối chiếu kết luận** — áp lại 6 luật theo thứ tự ưu tiên và so với
   `primary_issue` mà Policy Agent chọn.
2. **Kiểm tra sự tồn tại của ID** — mọi `item_id`, `payment_id`, `seller_id` phải
   khớp một dòng có thật trong CSV; mọi `evidence_id` phải đúng 1 trong 5 tiền tố cho phép.
3. **Kiểm tra schema và giới hạn** — `confidence ∈ [0,1]`, `case_status` khớp với
   issue, các giới hạn 5/10/3/3/5, `currency = BRL`, và trường hợp order không có
   item row thì `item_ids`/`seller_ids` rỗng và tổng tiền bằng `0.0`.

Nếu không đạt, case được **trả ngược về Policy Agent kèm danh sách lý do bị bác**,
tối đa 2 vòng. Hết vòng mà vẫn lệch thì `coordinator_finalize` ghi kết quả áp luật
tất định và hạ `confidence` xuống `0.55` — vì lúc đó kết luận do luật quyết định
chứ không phải do model đồng thuận.

**Đây là đánh đổi có chủ ý:** nộp file sai schema bị hard gate 0 điểm, nên hệ thống
ưu tiên tính chắc chắn hơn là bảo toàn ý kiến của model. Trace ghi lại đầy đủ
những case phải dùng fallback (`deterministic_fallback_used: true`) để đánh giá
được model 8B thực sự đúng bao nhiêu phần.

## 5. Điều tiết token (ràng buộc vận hành)

Gói free của Groq giới hạn **6.000 tokens/phút**, không phải giới hạn số request.
Mỗi case tiêu khoảng 2.300 token, nên trần thực tế chỉ hơn 2 case/phút — chạy 4 case
song song thì lần chạy đầu tiên hỏng 5/12 case vì lỗi 429.

`agents/ratelimit.py` giải quyết bằng token bucket sliding-window dùng chung cho mọi thread:

- Trước mỗi lần gọi, agent phải **xin trước** một lượng token ước lượng; hết hạn mức thì chờ.
- Sau khi gọi xong, số token thật được ghi bù để cửa sổ bám sát thực tế.
- Khi dính 429, hệ thống **đọc thời gian chờ Groq trả về** (`try again in 11.92s`) và
  chặn toàn bộ các thread khác trong khoảng đó — nếu không, các thread còn lại sẽ
  lao vào và cùng ăn 429.

Kèm hai tối ưu giảm token: `max_tokens` hạ từ 1200 xuống 400 (các agent chỉ trả JSON ngắn),
và Policy Agent nhận bằng chứng đã lược bỏ mảng `items[]`/`payments[]` chi tiết.

Nếu dùng gói trả phí hoặc provider khác, đặt biến môi trường `GROQ_TPM_LIMIT` để nới trần.

## 6. Trace

`logging/trace.jsonl` — mỗi dòng một sự kiện, ghi đè mỗi lượt chạy (không append):

```json
{"ts":"...","case_id":"EC_005","agent":"policy_agent","event":"decision_ready",
 "handoff_to":"verifier_agent","llm":{"prompt_tokens":812,"completion_tokens":96,"latency_ms":430},
 "payload":{"primary_issue":"late_delivery_seller","confidence":0.94,"rationale":"..."}}
```

Mỗi agent chuyên môn còn ghi cờ `llm_disagreed_with_data` — dùng để đo trực tiếp
mức tin cậy của model 8B trên từng domain.

## 7. Chạy

```bash
uv sync
cp .env.example .env    # điền GROQ_API_KEY
uv run run.py
```

| Lệnh | Mục đích |
| --- | --- |
| `uv run run.py` | Chạy 50 case qua đầy đủ 6 agent |
| `uv run run.py --dry-run` | Chỉ áp luật tất định, không gọi LLM — dùng làm baseline đối chiếu |
| `uv run run.py --cases EC_005` | Chạy một case để soi trace |
| `uv run scripts/smoke_graph.py` | Kiểm tra wiring của graph bằng LLM stub, không tốn quota |
| `uv run scripts/compare_runs.py` | Đối chiếu kết quả agent với baseline, kiểm tra điều kiện nộp |

Baseline dùng để đối chiếu:

```bash
uv run run.py --dry-run --output-dir output_baseline
```
