# Production Issue Report - 2025-12-15

## Tóm tắt

Phân tích dữ liệu production từ Analytics Service phát hiện nhiều vấn đề về data quality và configuration. Document này tổng hợp tất cả các issues được phát hiện từ file `data.json` và `.log`.

---

## Issue #1: ID bị mất precision (CRITICAL)

### Mô tả

TikTok post ID là số 19 chữ số, nhưng khi lưu/export dưới dạng JSON Number, JavaScript sẽ làm tròn do vượt quá `Number.MAX_SAFE_INTEGER` (9007199254740991 - 16 chữ số).

### Ví dụ

```json
// ID gốc từ TikTok: 7576276451171880962
// ID trong data.json: 7576276451171880000  ← 3 số cuối bị thành 000
```

### Impact

- Không thể trace back về post gốc trên TikTok
- Có thể gây duplicate hoặc mất data khi nhiều posts bị làm tròn về cùng ID
- Ảnh hưởng đến tất cả records

### Root Cause

- ID được lưu dưới dạng `BIGINT` trong PostgreSQL (đúng)
- Nhưng khi export ra JSON, client/tool có thể convert sang JavaScript Number

### Giải pháp

```python
# Option 1: Lưu ID dưới dạng string trong JSON response
class PostAnalyticsResponse(BaseModel):
    id: str  # Thay vì int

# Option 2: Sử dụng custom JSON encoder
import json

class BigIntEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, int) and obj > 2**53:
            return str(obj)
        return super().default(obj)
```

---

## Issue #2: NULL values được lưu dưới dạng string "NULL" (HIGH)

### Mô tả

Các trường có giá trị NULL trong database được export thành string `"NULL"` thay vì JSON `null`.

### Các trường bị ảnh hưởng

```json
{
  "job_id": "NULL", // Nên là: null
  "batch_index": "NULL", // Nên là: null
  "task_type": "NULL", // Nên là: null
  "keyword_source": "NULL", // Nên là: null
  "crawled_at": "NULL", // Nên là: null
  "pipeline_version": "NULL", // Nên là: null
  "fetch_error": "NULL", // Nên là: null
  "error_code": "NULL", // Nên là: null
  "error_details": "NULL" // Nên là: null
}
```

### Impact

- Client code phải handle cả `null` và `"NULL"` string
- Có thể gây lỗi khi parse/filter data
- Inconsistent data format

### Root Cause

Có thể do:

1. SQL export tool convert NULL thành string
2. Code serialization không handle None correctly
3. Data được insert với string "NULL" thay vì actual NULL

### Giải pháp

```python
# Kiểm tra và fix trong repository layer
def _serialize_value(value):
    if value is None or value == "NULL":
        return None
    return value
```

---

## Issue #3: Boolean được lưu dưới dạng string (MEDIUM)

### Mô tả

Các trường boolean được lưu dưới dạng string `"True"`/`"False"` thay vì JSON boolean `true`/`false`.

### Ví dụ

```json
{
  "is_viral": "False", // Nên là: false
  "is_kol": "True" // Nên là: true
}
```

### Impact

- Client code phải convert string to boolean
- Có thể gây lỗi logic khi so sánh: `"False" == true` trong JavaScript

### Giải pháp

```python
# Trong model hoặc serializer
class PostAnalytics(BaseModel):
    is_viral: bool
    is_kol: bool

    @field_validator('is_viral', 'is_kol', mode='before')
    @classmethod
    def parse_bool(cls, v):
        if isinstance(v, str):
            return v.lower() == 'true'
        return bool(v)
```

---

## Issue #4: SentimentAnalyzer không được cấu hình (HIGH)

### Mô tả

Module phân tích sentiment không hoạt động, tất cả posts đều trả về giá trị mặc định.

### Evidence từ Log

```
WARNING | SentimentAnalyzer is not configured; returning neutral defaults.
```

### Evidence từ Data

Tất cả 18 records đều có:

```json
{
  "overall_sentiment": "NEUTRAL",
  "overall_sentiment_score": 0,
  "overall_confidence": 0,
  "sentiment_probabilities": {}
}
```

### Impact

- Không có sentiment analysis thực sự
- Tất cả posts được đánh giá là NEUTRAL
- Mất giá trị business của analytics

### Giải pháp

1. Kiểm tra configuration cho SentimentAnalyzer
2. Verify model files/API credentials
3. Check environment variables

---

## Issue #5: Metadata từ Crawler bị mất (HIGH)

### Mô tả

Tất cả metadata fields từ crawler đều là "NULL", cho thấy data không được enrich đúng cách.

### Các trường bị ảnh hưởng

| Field            | Expected             | Actual |
| ---------------- | -------------------- | ------ |
| job_id           | UUID của crawl job   | "NULL" |
| batch_index      | Index của batch      | "NULL" |
| task_type        | "research_and_crawl" | "NULL" |
| keyword_source   | Keyword đã search    | "NULL" |
| crawled_at       | Timestamp crawl      | "NULL" |
| pipeline_version | "3.0"                | "NULL" |

### Impact

- Không thể trace data về job/batch gốc
- Không biết keyword nào đã được crawl
- Mất audit trail

### Root Cause

Có thể do:

1. Crawler không gửi metadata trong event payload
2. Analytics service không extract metadata từ batch data
3. Mapping sai giữa crawler output và analytics input

---

## Issue #6: Một số records thiếu impact_breakdown (LOW)

### Mô tả

2 records có `impact_breakdown: {}` rỗng và `impact_score: 0`.

### Records bị ảnh hưởng

- ID: `7583358705404562000` (published: 2025-12-13)
- ID: `7573421431627828000` (published: 2025-11-16)

### Possible Cause

- Error trong impact calculation
- Missing required fields cho calculation

---

## Issue #7: Unexpected batch size warning (LOW)

### Evidence từ Log

```
WARNING | Unexpected batch size: expected=%d, actual=%d, platform=%s, job_id=%s
```

### Impact

- Batch size không khớp với expected
- Có thể do partial batch hoặc data loss

---

## Tổng hợp theo Priority

| Priority | Issue                         | Impact              | Effort |
| -------- | ----------------------------- | ------------------- | ------ |
| P0       | #1 ID precision loss          | Data integrity      | Medium |
| P0       | #4 SentimentAnalyzer disabled | Core feature broken | Low    |
| P1       | #5 Metadata missing           | Traceability lost   | Medium |
| P1       | #2 NULL as string             | Data quality        | Low    |
| P2       | #3 Boolean as string          | Data quality        | Low    |
| P3       | #6 Missing impact_breakdown   | Minor data issue    | Low    |

---

## Action Items

### Immediate (Today)

1. [ ] Investigate SentimentAnalyzer configuration
2. [ ] Check crawler → analytics metadata flow
3. [ ] Verify database column types for boolean fields

### Short-term (This Week)

4. [ ] Fix ID serialization to use string format
5. [ ] Fix NULL value handling in export/serialization
6. [ ] Add data validation tests

### Long-term

7. [ ] Add data quality monitoring/alerts
8. [ ] Implement schema validation at ingestion
9. [ ] Add integration tests for full pipeline

---

## Files liên quan

### Analytics Service

- `repository/analytics_repository.py` - Data persistence
- `services/analytics/orchestrator.py` - Pipeline orchestration
- `internal/consumers/main.py` - Message processing

### Configuration

- `.env` - Environment variables
- `config/` - Service configuration
