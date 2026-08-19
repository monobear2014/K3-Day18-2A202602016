# Individual Reflection — Lab 18: Production RAG

**Tên:** Nguyễn Hoài Nam
**Lớp:** AICB-K34 · **Ngày 18** · **Modules phụ trách:** M1 → M5 (toàn bộ, bài cá nhân)
**Kết quả:** `pytest tests/ -v` → **37/37 passed** · `grep "# TODO" src/m*.py` → **0** · pipeline chạy end-to-end

---

## Phần 1: Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation từ chính số liệu của mình |
|----------------|--------|-------------|---------------------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 trên `all-MiniLM-L6-v2`. Vấn đề phát sinh: heading markdown ("## Nghỉ ốm", 11 ký tự) luôn có similarity thấp với đoạn sau nên bị tách thành chunk riêng — vô nghĩa để embed. Phải thêm bước gộp chunk < 100 ký tự vào chunk kề bên. Bài học: semantic chunking thuần cosine **không đủ**, luôn cần guard về kích thước tối thiểu. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Parent 2048 / child 256 → 26 parents, 108 children. Ràng buộc bất đối xứng dễ sai: parent lưu id trong `metadata["parent_id"]`, child lưu ở field `parent_id`. Quan trọng hơn: `pipeline.py` **chỉ index child rồi trả thẳng child**, không expand lên parent → Context Recall tụt 0.90 → 0.84 so với baseline. Tức là làm đúng một nửa ý tưởng thì **tệ hơn không làm**. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Split theo `^#{1,3}\s+`, giữ header trong text và lưu vào `metadata["section"]`. Trên corpus policy tiếng Việt (toàn markdown có heading rõ) đây là strategy hợp lý nhất — nhưng đề bài mặc định dùng hierarchical cho pipeline. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF chỉ dùng **thứ hạng**, không dùng score gốc — nhờ đó không phải normalize giữa BM25 (score không chặn trên) và cosine (0..1). Đo được: RRF tốn **0.22ms**, trong khi BM25 96.9ms và dense 8947ms. Khâu gộp gần như miễn phí; toàn bộ chi phí nằm ở hai nhánh search. |
| Vietnamese segmentation | M2 | `segment_vietnamese()` | underthesea nối từ ghép bằng `_` ("nghỉ_phép"). Nếu không `replace("_", " ")` thì corpus có token `nghỉ_phép` còn query tách thành `nghỉ` + `phép` → BM25 **không khớp gì cả**. Một dòng code quyết định BM25 hoạt động hay không. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Latency đo được **18 621ms** cho 20 cặp (CPU), chiếm 67% thời gian retrieval. Đổi lại chỉ +0.008 Context Precision trên corpus 108 chunk. Kết luận thẳng: **reranking không đáng trên corpus nhỏ** — nó sinh ra cho corpus hàng chục nghìn chunk nơi top-20 thực sự lẫn nhiều rác. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Metric thấp nhất là `faithfulness` 0.7976. Điều bất ngờ: 3/5 câu tệ nhất có `context_precision = context_recall = 1.0` — **context hoàn hảo mà vẫn sai**. RAGAS tách answer thành các claim rồi hỏi "có câu nào trong context chứng minh trực tiếp không", nên mọi suy luận số học bắc cầu đều bị phạt dù đáp án đúng (case #4: trả lời đúng 17.000.000 vẫn chỉ được faithfulness 0.5). |
| Contextual embeddings | M5 | `contextual_prepend()` / `_enrich_single_call()` | Prepend 1 câu mô tả chunk nằm ở đâu trong tài liệu, giữ nguyên vẹn text gốc. Chọn combined mode: 1 API call/chunk thay vì 4 → tiết kiệm 75% chi phí. Với 108 chunks thì đó là chênh lệch giữa 108 và 432 call. |
| Failure analysis / Error Tree | M4 | `failure_analysis()` | Diagnostic Tree map metric thấp nhất → tầng hỏng. Áp dụng lên bottom-5 cho ra phân bố: retrieval 1 case, ranking 1 case, **generation 3 case**. Chính con số này giải thích vì sao production không thắng baseline. |

---

## Phần 2: Khó khăn & cách giải quyết

### 2.1 Môi trường Python — mất nhiều thời gian nhất

**Lỗi:** `pip install -r requirements.txt` trên Python 3.13 (anaconda mặc định) không cài được
`ragas<0.2`. Máy sẵn có `ragas 0.3.9`, mà API 0.3 đã đổi tên cột dataset
(`question/answer/contexts/ground_truth` → `user_input/response/retrieved_contexts/reference`),
nên pseudo-code trong scaffold sẽ fail nếu chạy thẳng.

**Debug:** So `.python-version` (3.11) với `python3 --version` (3.13.9) → phát hiện lệch.
Kiểm tra `importlib.metadata.version("ragas")` để biết bản thực tế đang cài.

**Giải quyết:** Tạo venv sạch bằng `uv venv --python 3.11 .venv`, cài đúng `requirements.txt`
→ `ragas 0.1.22`, code bám sát scaffold, không phải port sang API mới.

### 2.2 Mạng đứt liên tục khi tải dependency

**Exact errors gặp phải:**
```
× Failed to download `pyarrow==25.0.1`
╰─▶ Failed to download distribution due to network timeout. Try increasing
    UV_HTTP_TIMEOUT (current value: 30s).
```
```
× Failed to download `pandas==3.0.5`
├─▶ client error (Connect)
╰─▶ dns error: failed to lookup address information: nodename nor servname provided, or not known
```
```
× Download failed after 6 attempts because not enough bytes were received (37.7 MB/111.2 MB)
  URL: .../torch-2.13.0-cp311-cp311-macosx_14_0_arm64.whl
```

**Giải quyết:** `UV_HTTP_TIMEOUT=600`, `UV_CONCURRENT_DOWNLOADS=2`, và bọc lệnh cài trong vòng lặp
retry 5 lần có kiểm tra điều kiện dừng (`python -c "import ragas, sentence_transformers, ..."`).
Cache của uv giữ lại wheel đã tải nên retry không mất công tải lại từ đầu.

### 2.3 Tải model: sai cách làm mất 18 phút cho model 90MB

**Triệu chứng:** `snapshot_download("sentence-transformers/all-MiniLM-L6-v2")` báo
`Fetching 30 files`, file thứ 8 treo 17 phút 49 giây, ETA nhảy lên `1:58:18`, hai file
`.incomplete` đứng ở 0 byte.

**Debug:** `find ~/.cache/huggingface/hub -name "*.incomplete"` để thấy file nào đang treo, rồi
đối chiếu danh sách file trong repo HF.

**Nguyên nhân:** `snapshot_download` kéo **toàn bộ** repo — gồm cả `tf_model.h5` (TensorFlow),
`rust_model.ot`, thư mục `onnx/` và `openvino/` — trong khi sentence-transformers chỉ dùng
`model.safetensors` + config + tokenizer. Với `bge-m3` thì repo đầy đủ hơn 8GB.

**Giải quyết:** Gọi thẳng `SentenceTransformer(name)` / `CrossEncoder(name)` để loader tự resolve
đúng file cần. MiniLM tải lại xong trong **15 giây**. Tổng khối lượng từ ~12GB xuống ~4.6GB.

### 2.4 Qdrant — API đã thay đổi

**Lỗi tiềm ẩn:** scaffold hướng dẫn `client.recreate_collection(...)`, nhưng hàm này đã bị bỏ ở
`qdrant-client 1.19`. Phát hiện trước khi chạy nên không mất thời gian debug runtime.

**Giải quyết:** `collection_exists()` → `delete_collection()` → `create_collection()`.
Ngoài ra `search()` cũ được thay bằng `query_points()` (trả về object có `.points`).

Về container: máy đã có sẵn image `qdrant/qdrant:v1.18.3` từ lab 17, còn `docker-compose.yml`
gọi `:latest`. Thay vì pull thêm 120MB qua mạng yếu, chỉ cần
`docker tag qdrant/qdrant:v1.18.3 qdrant/qdrant:latest` — không sửa file được chấm.

### 2.5 Hai cái bẫy nhỏ nhưng làm hỏng test / pipeline

1. **`json.loads` vỡ vì markdown fence.** `extract_metadata()` và `_enrich_single_call()` parse JSON
   từ output LLM. gpt-4o-mini hay bọc trong ```` ```json ```` → `JSONDecodeError`.
   Fix: `response_format={"type": "json_object"}`.
2. **`test_summarize_shorter_than_original` fail khi có API key.** Test bắt
   `len(summary) <= 2 × len(original)`, mà đoạn mẫu chỉ 65 ký tự — LLM tóm "2-3 câu" chắc chắn dài hơn 130.
   Fix: chỉ nhận summary khi nó **ngắn hơn bản gốc**, ngược lại rơi về extractive.
   Đây cũng là logic đúng về mặt sản phẩm: tóm tắt dài hơn bản gốc thì tóm tắt làm gì.

### 2.6 Kiến thức còn thiếu → cách bổ sung

- **Cơ chế chấm của RAGAS.** Ban đầu mình tưởng faithfulness thấp = model bịa. Đọc kỹ mới hiểu nó
  tách answer thành claim rồi kiểm từng claim, nên **suy luận số học đúng vẫn bị phạt**.
  → Cần đọc paper RAGAS + thử tự viết lại metric để hiểu sâu.
- **Khi nào KHÔNG cần reranking.** Bài giảng dạy cách bật rerank, không dạy cách quyết định có nên bật.
  Số liệu của mình cho câu trả lời: +0.008 precision đổi lấy 18.6s/query là lỗ.
  → Cần học ước lượng ngưỡng corpus size mà rerank bắt đầu có lãi.
- **Serving trên Apple Silicon.** Toàn bộ số latency của mình chạy CPU. Chưa biết cấu hình MPS cho
  sentence-transformers. → Việc tiếp theo phải tìm hiểu.

---

## Phần 3: Action Plan cho project

> ⚠️ Thay `[Tên project]` và phần "Hiện tại" bằng project thật của mình trước khi nộp.

## Project: [Tên project] — RAG hỏi đáp tài liệu nội bộ

### Hiện tại
- RAG pipeline hiện tại: chunking cố định + dense-only search + trả top-k thẳng cho LLM.
- Known issues:
  1. Tài liệu có nhiều phiên bản (quy định cũ/mới) → bot trả lời theo bản đã hết hiệu lực.
  2. Câu hỏi cần ghép thông tin từ 2 tài liệu thì trả lời thiếu.
  3. Chưa có bộ test set nào để biết sửa xong có tốt lên không.

### Plan áp dụng (xếp theo lợi tức, rút ra từ chính số liệu lab này)

1. [ ] **Evaluation TRƯỚC — làm đầu tiên, không phải cuối cùng.**
   Bài học đắt nhất hôm nay: mình đã build đủ hybrid + rerank + enrichment rồi mới đo, và phát hiện
   **production thua baseline** ở 2/4 metric. Nếu đo trước thì đã biết retrieval không phải nút thắt.
   → Dựng test set 30–50 câu (đủ 6 loại: lookup, version, negation, multi-hop, numeric, ambiguous)
   và chạy RAGAS trên pipeline hiện tại để có baseline **trước khi sửa bất cứ thứ gì**.

2. [ ] **Chunking: hierarchical parent-child, và PHẢI expand child → parent.**
   Retrieve child 256 để chính xác, trả parent 2048 để đủ ngữ cảnh. Lab này chứng minh làm nửa vời
   (chỉ retrieve child) khiến Context Recall tụt 0.90 → 0.84 — tệ hơn cả chunking cơ bản.

3. [ ] **Metadata versioning — ưu tiên cao vì đúng known issue #1.**
   Gắn `version`, `effective_date`, `superseded_by` vào metadata ngay khi ingest, lọc
   `superseded == False` trước khi rerank. Lab cho thấy đây là lỗi mà **không model nào sửa được**:
   hai tài liệu mâu thuẫn nhưng giống hệt nhau về ngữ nghĩa, mọi embedding đều xếp chúng cạnh nhau.
   Tính thời sự là metadata, không phải ngữ nghĩa.

4. [ ] **Prompt cho phép suy luận số học.** Prompt "CHỈ dựa trên context, không có thì nói không tìm thấy"
   làm model từ chối cả những suy luận hợp lệ (câu "55 triệu cần ai duyệt" → "Không tìm thấy"
   dù context có ngưỡng "trên 50 triệu → CEO", faithfulness 0.0). Sửa prompt là fix rẻ nhất, tác động lớn nhất.

5. [ ] **Search: hybrid BM25 + Dense + RRF.** RRF tốn 0.22ms — rẻ đến mức không có lý do không dùng.
   BM25 đặc biệt cần cho tiếng Việt khi query chứa mã số, tên riêng, thuật ngữ chính xác.
   Nhớ `replace("_", " ")` sau underthesea.

6. [ ] **Reranking: CHƯA bật.** Chỉ bật khi corpus > ~10K chunks và số liệu chứng minh có lãi.
   Nếu bật thì dùng `flashrank` (<5ms) trước, cross-encoder 2GB chỉ khi flashrank không đủ.

7. [ ] **Enrichment: contextual prepend, combined 1 call/chunk, gọi song song.**
   108 call tuần tự đã là điểm nghẽn trong lab; ở quy mô vài nghìn chunk thì bắt buộc phải
   `ThreadPoolExecutor` vì các call hoàn toàn độc lập.

### Timeline

| Tuần | Việc | Định nghĩa "xong" |
|---|---|---|
| **Tuần 1** | Dựng test set 30–50 câu + chạy RAGAS lấy baseline | Có `baseline_report.json`, biết metric nào yếu nhất |
| **Tuần 2** | Sửa prompt (suy luận số học) + hierarchical parent-child expansion | Faithfulness và Context Recall đều tăng so với baseline |
| **Tuần 3** | Metadata versioning + filter superseded | 100% câu hỏi version-conflict trả lời theo bản hiện hành |
| **Tuần 4** | Hybrid BM25 + Dense + RRF | Context Recall ≥ 0.90; đo latency trước/sau |
| **Tuần 5** | Enrichment song song + đo lại toàn bộ; quyết định có bật rerank không **dựa trên số** | Bảng so sánh 4 metric qua từng tuần; quyết định rerank có/không kèm lý do định lượng |

### Nguyên tắc rút ra, áp dụng cho mọi thay đổi về sau
**Đo trước, sửa sau, và mỗi lần chỉ đổi một thứ.** Lab này bật cùng lúc hierarchical + enrichment +
hybrid + rerank, nên khi thấy faithfulness tụt 0.0149 thì không thể biết thành phần nào gây ra —
phải lần ngược qua failure analysis mới truy được về việc child chunk không expand lên parent.

---

## Tự đánh giá

> Điền lại theo cảm nhận thật của mình.

| Tiêu chí | Tự chấm (1-5) | Ghi chú |
|----------|---------------|---------|
| Hiểu bài giảng | | Map được cả 5 concept vào code và giải thích được bằng số liệu của chính mình |
| Code quality | | 37/37 test pass, 0 TODO; có xử lý fallback khi không có API key |
| Problem solving | | Tự phát hiện và xử lý: ragas 0.3 vs 0.1, `recreate_collection` bị gỡ, `snapshot_download` tải thừa, `json.loads` vỡ vì markdown fence |
| Phân tích kết quả | | Chỉ ra được production thua baseline và truy đúng nguyên nhân thay vì chỉ báo cáo con số |
