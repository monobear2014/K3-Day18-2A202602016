# Failure Analysis — Lab 18: Production RAG

**Học viên:** Nguyễn Hoài Nam · **Lớp:** AICB-K34 · **Ngày 18**
**Corpus:** 26 documents (24 .md + 1 .pdf có text layer; 2 PDF scan bị bỏ vì chưa OCR) → 108 child chunks
**Test set:** 20 câu · **Model:** bge-m3 (embed) · bge-reranker-v2-m3 (rerank) · gpt-4o-mini (generate + RAGAS judge)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8125 | 0.7976 | **−0.0149** |
| Answer Relevancy | 0.7208 | 0.7764 | **+0.0555** |
| Context Precision | 0.9250 | 0.9333 | **+0.0083** |
| Context Recall | 0.9000 | 0.8417 | **−0.0583** |

> Baseline = `chunk_basic` (paragraph ~500 ký tự) + dense-only, top-3.
> Production = `chunk_hierarchical` (child 256) + enrichment + BM25/Dense/RRF + CrossEncoder top-20→3.

### Kết luận quan trọng nhất: pipeline "xịn" hơn nhưng KHÔNG thắng baseline

Đây là kết quả phản trực giác nhất của bài lab, và nó có lý do rõ ràng:

1. **Retrieval vốn đã không phải nút thắt.** Context Precision 0.93 và Context Recall 0.84 nghĩa là
   retriever gần như luôn lấy đúng tài liệu. Corpus chỉ 21K ký tự / 26 file — quá nhỏ để hybrid search
   và reranking phát huy. M2+M3 sinh ra để cứu recall/precision trên corpus hàng chục nghìn chunk,
   ở đây chúng chỉ cải thiện được +0.008 precision.
2. **Nút thắt thật nằm ở tầng generation.** 4/5 câu tệ nhất có `context_precision` và
   `context_recall` = 1.0 nhưng `faithfulness` ≤ 0.5 — tức là **context đúng, LLM trả lời sai**.
   Không có thay đổi nào ở M1/M2/M3 chạm được vào lỗi này.
3. **Context Recall tụt 0.90 → 0.84 là cái giá của child chunk.** Baseline trả context 500 ký tự;
   production index child 256 ký tự và trả thẳng child, **không expand ngược lên parent**.
   Ý tưởng parent-child là "retrieve child để chính xác, trả parent để đủ ngữ cảnh" — `pipeline.py`
   mới làm nửa đầu. Đây là fix có giá trị cao nhất (chi tiết ở mục "Nếu có thêm 1 giờ").

---

## Bottom-5 Failures

### #1 — Multi-hop bị cắt cụt vì top-3
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 ngày cơ bản + 3 ngày thâm niên = 18 ngày phép. Lương Senior (P3-P4): 20–35 triệu VNĐ/tháng.
- **Got:** "…được nghỉ 18 ngày phép năm. Lương … không có thông tin cụ thể về lương trong context."
- **Worst metric:** `answer_relevancy = 0.0` (faithfulness 0.667 · precision 1.0 · **recall 0.0**)
- **Error Tree:**
  1. Output sai? → **Sai một nửa** — phần ngày phép đúng, phần lương trống.
  2. Context đúng? → **Không đủ.** Câu hỏi cần 2 tài liệu (`nghi_phep_nam_v2024.md` + `bang_luong_2024.md`),
     nhưng sau rerank `top_k=3` cả 3 slot đều bị chunk về nghỉ phép chiếm — chunk lương bị đẩy khỏi top-3.
  3. Query OK? → Query là **một câu hỏi ghép 2 ý**; embedding của nó nghiêng hẳn về "nghỉ phép"
     nên dense search cũng ưu tiên nhánh đó.
  4. Fix ở bước: **M3 / query understanding**
- **Root cause:** `RERANK_TOP_K = 3` là hằng số cứng, không phân biệt câu hỏi 1 nguồn hay nhiều nguồn.
  Reranker làm đúng việc của nó — xếp hạng theo độ liên quan với *toàn bộ* query — nhưng với query
  ghép thì "liên quan nhất" và "đủ để trả lời" là hai chuyện khác nhau.
- **Suggested fix:** Query decomposition (tách thành "ngày phép Senior 9 năm" + "lương Senior") rồi
  union context; hoặc nâng `RERANK_TOP_K` lên 5 cho câu hỏi có liên từ "và"; hoặc MMR để ép đa dạng nguồn.

### #2 — Context đúng 100% nhưng LLM từ chối trả lời
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** "Không tìm thấy."
- **Worst metric:** `faithfulness = 0.0` (**precision 1.0 · recall 1.0** · relevancy 0.0)
- **Error Tree:**
  1. Output sai? → **Sai hoàn toàn** (từ chối trả lời).
  2. Context đúng? → **Đúng tuyệt đối.** Precision = recall = 1.0, chunk `mua_sam.md` chứa đúng
     bảng ngưỡng phê duyệt.
  3. Query OK? → Có.
  4. Fix ở bước: **Generation (prompt)**
- **Root cause:** Context ghi ngưỡng dạng "trên 50.000.000 VNĐ → CEO", câu hỏi hỏi "55 triệu".
  Cần một phép **so sánh số học** (55 > 50) mà prompt hiện tại lại ra lệnh
  *"Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"* — model diễn giải
  "không có dòng nào ghi 55 triệu" thành "không có thông tin". Prompt chống hallucination
  đã siết chặt đến mức chặn luôn suy luận hợp lệ.
- **Suggested fix:** Sửa system prompt thành *"Chỉ dùng context, nhưng ĐƯỢC PHÉP suy luận số học,
  so sánh ngưỡng và đối chiếu khoảng giá trị. Chỉ nói 'Không tìm thấy' khi context hoàn toàn
  không đề cập chủ đề."*

### #3 — Version conflict v1/v2 (lỗi "kinh điển" mà corpus cố tình gài)
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0) là **120 ngày**; v1.0 (90 ngày) đã bị thay thế.
- **Got:** "Có hai quy định… 1. mỗi **90 ngày**. 2. mỗi **120 ngày**. Tùy vào từng phương thức xác thực…"
- **Worst metric:** `faithfulness = 0.3333` (precision 0.5833 · recall 1.0 · relevancy 0.7927)
- **Error Tree:**
  1. Output sai? → **Sai** — liệt kê cả 2 phiên bản rồi bịa lý do "tùy phương thức xác thực"
     (câu này không có trong bất kỳ chunk nào → chính là phần làm faithfulness rơi xuống 0.33).
  2. Context đúng? → **Đúng nhưng lẫn.** Recall 1.0 (có 120 ngày), precision 0.58
     (chunk `mat_khau_v1.md` cũ cũng lọt vào top-3).
  3. Query OK? → Có, nhưng **query không mang thông tin thời gian** nên không có tín hiệu nào
     để hệ thống biết nên ưu tiên v2.
  4. Fix ở bước: **M2/M3 (metadata filter) — không phải prompt**
- **Root cause:** Cả `mat_khau_v1.md` và `mat_khau_v2.md` đều nói về "chu kỳ đổi mật khẩu" nên
  **cosine similarity của chúng gần như bằng nhau** — dense search không có cách nào phân biệt.
  Cross-encoder cũng không, vì nó chấm độ liên quan query↔chunk chứ không biết chunk nào mới hơn.
  Đây là lỗi mà việc "tăng chất lượng retrieval" không bao giờ sửa được: cả hai chunk đều *đúng chủ đề*.
- **Suggested fix:** Bơm version vào metadata (`version`, `effective_date`, `superseded_by`) ngay từ M5
  `extract_metadata`, rồi lọc `superseded == False` trước khi rerank. Rẻ hơn và chắc ăn hơn mọi
  cách chỉnh model.

### #4 — Faithfulness bị phạt vì suy luận bắc cầu qua 2 chunk
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất 20.000.000 → thử việc = 85% × 20.000.000 = **17.000.000 VNĐ/tháng**.
- **Got:** "17.000.000 VNĐ (85% của 20.000.000 VNĐ)" — **đáp án đúng**.
- **Worst metric:** `faithfulness = 0.5` (precision 0.5833 · recall 1.0 · relevancy 0.8241)
- **Error Tree:**
  1. Output sai? → **Không, đúng.** Nhưng RAGAS vẫn phạt.
  2. Context đúng? → Đúng, nhưng **hai mảnh nằm ở hai chunk khác nhau**: bậc lương Junior ở
     `bang_luong_2024.md`, tỷ lệ 85% ở `thu_viec.md`.
  3. Query OK? → Có.
  4. Fix ở bước: **M1 (chunk size) + evaluation design**
- **Root cause:** `faithfulness` của RAGAS tách câu trả lời thành các *claim* rồi hỏi
  "claim này có được câu nào trong context chứng minh trực tiếp không?". Claim
  "lương thử việc Junior = 17.000.000" **không xuất hiện nguyên văn ở đâu cả** — nó là kết quả phép nhân.
  Chunk 256 ký tự càng làm nặng thêm: hai tiền đề bị đẩy ra xa nhau.
- **Suggested fix:** Expand child → parent trước khi đưa vào LLM (hai tiền đề có cơ hội nằm chung
  một context); và yêu cầu model **trình bày từng bước** ("Junior cao nhất: 20tr [nguồn A];
  tỷ lệ thử việc 85% [nguồn B]; ⇒ 17tr") để mỗi claim trung gian đều truy vết được về context.

### #5 — Sai ở phép tính pro-rata
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; phí 2%/tháng trên 15.000.000 = 300.000/tháng → pro-rata ≈ **50.000 VNĐ**.
- **Got:** "15 triệu × 2% = 300.000 VNĐ. Vậy bị phạt 300.000 VNĐ."
- **Worst metric:** `faithfulness = 0.2857` (precision 0.8333 · recall 1.0 · relevancy 0.8526)
- **Error Tree:**
  1. Output sai? → **Sai về lượng** — lấy phí cả tháng gán cho 5 ngày quá hạn.
  2. Context đúng? → Đúng (recall 1.0): có cả thời hạn 15 ngày lẫn mức 2%/tháng.
  3. Query OK? → Có.
  4. Fix ở bước: **Generation (reasoning)**
- **Root cause:** Context nói "2%/**tháng**", câu hỏi ngụ ý quá hạn **5 ngày**. Model bỏ qua đơn vị
  thời gian và không tự nghĩ tới pro-rata. Corpus cũng không viết rõ "tính theo ngày" nên đây là
  vùng xám — nhưng ground truth thì kỳ vọng có pro-rata.
- **Suggested fix:** Với câu hỏi numeric, thêm chỉ dẫn *"chú ý đơn vị thời gian, quy đổi pro-rata
  khi kỳ tính phí khác kỳ thực tế, và ghi rõ công thức"*; hoặc tách một nhánh tool-calling
  cho phép LLM gọi máy tính thay vì nhẩm.

---

## Tổng hợp Diagnostic Tree — lỗi tập trung ở đâu?

| Tầng | Số case trong bottom-5 | Bằng chứng |
|---|---|---|
| Retrieval (M1/M2) | 1 (#1 recall 0.0) | 4/5 case còn lại có recall = 1.0 |
| Ranking (M3) | 1 (#3 precision 0.58) | version cũ lọt top-3 |
| **Generation (prompt/reasoning)** | **3 (#2, #4, #5)** | context precision = recall = 1.0 nhưng faithfulness ≤ 0.5 |

**Đây là lý do production không thắng baseline:** 60% lỗi nằm ở tầng mà M1–M3 không chạm tới.
Đầu tư thêm vào chunking/hybrid/rerank sẽ cho lợi tức gần bằng 0 trên corpus này.

---

## Case Study (cho presentation)

**Question chọn phân tích:** *"Bao lâu phải đổi mật khẩu một lần?"* (#3)

**Error Tree walkthrough:**
1. Output đúng? → **Không.** Trả về cả 90 và 120 ngày, kèm một mệnh đề bịa
   ("tùy phương thức xác thực") không tồn tại trong context.
2. Context đúng? → **Có 120 ngày (recall 1.0) nhưng lẫn 90 ngày (precision 0.58).**
   Cả `mat_khau_v1.md` lẫn `mat_khau_v2.md` đều lọt vào top-3 sau rerank.
3. Query rewrite OK? → Query hợp lệ, nhưng **không chứa tín hiệu thời gian** — không có gì để
   hệ thống suy ra "phiên bản hiện hành".
4. Fix ở bước: **M5 `extract_metadata` (gắn version) + filter trước M3**, không phải ở prompt.

**Vì sao case này đáng nói:** nó cho thấy giới hạn bản chất của similarity-based retrieval.
Hai tài liệu mâu thuẫn nhau nhưng **giống nhau về ngữ nghĩa** — mọi embedding model đều xếp chúng
cạnh nhau, mọi cross-encoder đều chấm chúng điểm gần bằng nhau. Không có lượng
"tăng chất lượng retrieval" nào sửa được, vì cả hai đều *đúng chủ đề*; chỉ một tài liệu là *đúng thời điểm*.
Tính thời sự là **metadata**, không phải ngữ nghĩa — và phải xử lý bằng filter chứ không phải bằng model.

**Nếu có thêm 1 giờ, sẽ optimize (xếp theo lợi tức giảm dần):**
1. **Child → parent expansion** (~15 phút): trong `run_query`, sau rerank thì tra `parent_id`
   trong metadata để trả về parent 2048 ký tự thay vì child 256. Kỳ vọng kéo Context Recall
   0.84 → ≥0.90 (bằng hoặc hơn baseline) và giúp luôn case #4 vì hai tiền đề có thể vào chung context.
   Đây là bước duy nhất hiện thực hoá đúng ý tưởng hierarchical chunking.
2. **Sửa system prompt cho phép suy luận số học** (~5 phút): cứu trực tiếp case #2 (0.0 → ~1.0)
   và #5. Rẻ nhất, tác động lớn nhất trên faithfulness.
3. **Version metadata filter** (~20 phút): thêm `version`/`superseded` vào `extract_metadata`,
   lọc trước khi rerank. Cứu #3 và các câu version-conflict khác trong test set.
4. **Nâng `RERANK_TOP_K` 3 → 5 cho câu multi-hop** (~10 phút): cứu #1.

---

## Latency Breakdown

### Per-query (trung bình 10 câu hỏi, Apple Silicon **CPU**, chưa bật MPS)

| Bước | avg (ms) | p50 (ms) | max (ms) | % tổng |
|---|---:|---:|---:|---:|
| BM25 search (top-20) | 96.9 | 71.9 | 222.9 | 0.3% |
| Dense search bge-m3 (top-20) | 8 947.7 | 7 276.7 | 18 995.8 | 32.3% |
| RRF fusion | **0.22** | 0.11 | 0.75 | ~0% |
| CrossEncoder rerank (20→3) | 18 621.6 | 18 150.7 | 24 281.6 | 67.3% |
| **Tổng retrieval** | **~27 666** | | | 100% |

Nhận xét:
- **RRF gần như miễn phí (0.22ms)** — rẻ hơn BM25 440 lần. Hybrid search về mặt chi phí
  là "cho không": toàn bộ giá phải trả nằm ở hai nhánh search, không nằm ở khâu gộp.
- **Rerank chiếm 2/3 thời gian** vì cross-encoder phải chạy 20 lần forward pass full attention
  trên cặp (query, doc) — không cache được như bi-encoder. Đây đúng là trade-off precision↔latency
  của bài giảng, và trên corpus này ta **trả 18.6s để đổi lấy +0.008 context precision** — không đáng.
- Con số cao bất thường vì chạy CPU. Bật MPS/GPU thường giảm 5–10 lần (rerank → ~2s).
  `flashrank` (đã implement sẵn ở `FlashrankReranker`) là phương án <5ms nếu cần production thật.

### Per-stage (toàn pipeline, 108 chunks / 20 câu hỏi)

| Bước | Thời gian | Ghi chú |
|---|---:|---|
| M1 Chunking | 0.1s | 26 docs → 26 parents / 108 children |
| M5 Enrichment | **50 130s** ⚠️ | 108 API call — **bất thường, xem bên dưới** |
| M2 Indexing (BM25 + bge-m3 → Qdrant) | 28.7s | 108 chunks |
| M3 Reranker load | ~0s | model đã cache trong RAM |
| M4 RAGAS (20 câu × 4 metric) | 37.9s | |
| **Tổng** | **50 764s (14.1h)** | |

⚠️ **Cảnh báo về số 50 130s:** trung bình 464 giây cho **một** call `gpt-4o-mini` là vô lý —
bình thường ~1–2s. Nguyên nhân là mạng lúc chạy cực yếu (~100 KB/s, có lúc DNS fail),
khiến mỗi request bị treo rồi retry. **Đây là artifact môi trường, không phải đặc tính của M5.**
Với mạng bình thường, 108 chunks × ~1.5s ≈ **3 phút**.

Dù vậy nó cho thấy đúng bài học chi phí của enrichment: **combined mode (1 call/chunk) thay vì
4 technique riêng lẻ đã tiết kiệm 75%** — nếu làm 4 call riêng thì con số trên sẽ là ~56 giờ.
Bước tiếp theo hiển nhiên là gọi song song (`asyncio` / `ThreadPoolExecutor`), vì 108 call này
hoàn toàn độc lập với nhau.
