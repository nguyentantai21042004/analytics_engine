# Analytics Engine API Documentation

REST API để truy vấn dữ liệu phân tích social media.

**Base URL:** `/`  
**Version:** v1  
**Swagger UI:** `/swagger/index.html`

---

## Mục lục

1. [Posts API](#1-posts-api)
2. [Summary API](#2-summary-api)
3. [Trends API](#3-trends-api)
4. [Keywords API](#4-keywords-api)
5. [Alerts API](#5-alerts-api)
6. [Errors API](#6-errors-api)
7. [Health API](#7-health-api)
8. [Common Schemas](#8-common-schemas)

---

## 1. Posts API

### 1.1 GET /posts

Lấy danh sách bài viết đã phân tích với phân trang, lọc và sắp xếp.

#### Request Parameters

| Parameter    | Type     | Required | Default       | Description                                          |
| ------------ | -------- | -------- | ------------- | ---------------------------------------------------- |
| `project_id` | UUID     | ✅       | -             | ID của project                                       |
| `brand_name` | string   | ❌       | null          | Lọc theo tên thương hiệu                             |
| `keyword`    | string   | ❌       | null          | Lọc theo từ khóa                                     |
| `platform`   | string   | ❌       | null          | Lọc theo nền tảng (facebook, tiktok, youtube...)     |
| `sentiment`  | string   | ❌       | null          | Lọc theo sentiment (POSITIVE, NEUTRAL, NEGATIVE)     |
| `risk_level` | string   | ❌       | null          | Lọc theo mức độ rủi ro (LOW, MEDIUM, HIGH, CRITICAL) |
| `intent`     | string   | ❌       | null          | Lọc theo intent                                      |
| `is_viral`   | boolean  | ❌       | null          | Lọc bài viral                                        |
| `is_kol`     | boolean  | ❌       | null          | Lọc bài từ KOL                                       |
| `from_date`  | datetime | ❌       | null          | Ngày bắt đầu (inclusive)                             |
| `to_date`    | datetime | ❌       | null          | Ngày kết thúc (inclusive)                            |
| `sort_by`    | string   | ❌       | "analyzed_at" | Cột sắp xếp                                          |
| `sort_order` | string   | ❌       | "desc"        | Thứ tự sắp xếp (asc/desc)                            |
| `page`       | int      | ❌       | 1             | Số trang (1-based)                                   |
| `page_size`  | int      | ❌       | 20            | Số item mỗi trang (1-100)                            |

#### Response

```json
{
  "success": true,
  "data": [
    {
      "id": "string",
      "platform": "string",
      "permalink": "string | null",
      "content_text": "string | null (max 300 chars)",
      "author_name": "string | null",
      "author_username": "string | null",
      "author_is_verified": "boolean | null",
      "overall_sentiment": "POSITIVE | NEUTRAL | NEGATIVE",
      "overall_sentiment_score": "float (-1 to 1)",
      "primary_intent": "string",
      "impact_score": "float (0-100)",
      "risk_level": "LOW | MEDIUM | HIGH | CRITICAL",
      "is_viral": "boolean",
      "is_kol": "boolean",
      "view_count": "int",
      "like_count": "int",
      "comment_count": "int",
      "published_at": "datetime (ISO 8601)",
      "analyzed_at": "datetime (ISO 8601)",
      "brand_name": "string | null",
      "keyword": "string | null",
      "job_id": "string | null"
    }
  ],
  "meta": {
    "page": "int",
    "page_size": "int",
    "total_items": "int",
    "total_pages": "int",
    "has_next": "boolean",
    "has_prev": "boolean"
  }
}
```

#### Response Fields Description

| Field                     | Type     | Description                                        |
| ------------------------- | -------- | -------------------------------------------------- |
| `id`                      | string   | ID duy nhất của bài viết                           |
| `platform`                | string   | Nền tảng (facebook, tiktok, youtube, instagram...) |
| `permalink`               | string   | Link trực tiếp đến bài viết gốc                    |
| `content_text`            | string   | Nội dung bài viết (cắt ngắn 300 ký tự)             |
| `author_name`             | string   | Tên hiển thị của tác giả                           |
| `author_username`         | string   | Username của tác giả                               |
| `author_is_verified`      | boolean  | Tài khoản đã xác minh                              |
| `overall_sentiment`       | string   | Sentiment tổng thể                                 |
| `overall_sentiment_score` | float    | Điểm sentiment (-1 đến 1)                          |
| `primary_intent`          | string   | Intent chính của bài viết                          |
| `impact_score`            | float    | Điểm ảnh hưởng (0-100)                             |
| `risk_level`              | string   | Mức độ rủi ro                                      |
| `is_viral`                | boolean  | Bài viết có viral không                            |
| `is_kol`                  | boolean  | Bài viết từ KOL không                              |
| `view_count`              | int      | Số lượt xem                                        |
| `like_count`              | int      | Số lượt thích                                      |
| `comment_count`           | int      | Số bình luận                                       |
| `published_at`            | datetime | Thời gian đăng bài                                 |
| `analyzed_at`             | datetime | Thời gian phân tích                                |
| `brand_name`              | string   | Tên thương hiệu liên quan                          |
| `keyword`                 | string   | Từ khóa tìm kiếm                                   |
| `job_id`                  | string   | ID của job crawl                                   |

---

### 1.2 GET /posts/all

Lấy tất cả bài viết không phân trang. Dùng cho export hoặc khi cần toàn bộ dữ liệu.

#### Request Parameters

| Parameter    | Type     | Required | Default       | Description               |
| ------------ | -------- | -------- | ------------- | ------------------------- |
| `project_id` | UUID     | ✅       | -             | ID của project            |
| `brand_name` | string   | ❌       | null          | Lọc theo tên thương hiệu  |
| `keyword`    | string   | ❌       | null          | Lọc theo từ khóa          |
| `platform`   | string   | ❌       | null          | Lọc theo nền tảng         |
| `sentiment`  | string   | ❌       | null          | Lọc theo sentiment        |
| `risk_level` | string   | ❌       | null          | Lọc theo mức độ rủi ro    |
| `intent`     | string   | ❌       | null          | Lọc theo intent           |
| `is_viral`   | boolean  | ❌       | null          | Lọc bài viral             |
| `is_kol`     | boolean  | ❌       | null          | Lọc bài từ KOL            |
| `from_date`  | datetime | ❌       | null          | Ngày bắt đầu              |
| `to_date`    | datetime | ❌       | null          | Ngày kết thúc             |
| `sort_by`    | string   | ❌       | "analyzed_at" | Cột sắp xếp               |
| `sort_order` | string   | ❌       | "desc"        | Thứ tự sắp xếp            |
| `limit`      | int      | ❌       | 1000          | Số lượng tối đa (1-10000) |

#### Response

```json
{
  "success": true,
  "data": [PostListItem],
  "meta": {
    "total": "int"
  }
}
```

---

### 1.3 GET /posts/{post_id}

Lấy chi tiết đầy đủ của một bài viết theo ID, bao gồm comments.

#### Path Parameters

| Parameter | Type   | Required | Description     |
| --------- | ------ | -------- | --------------- |
| `post_id` | string | ✅       | ID của bài viết |

#### Response

```json
{
  "success": true,
  "data": {
    "id": "string",
    "platform": "string",
    "permalink": "string | null",
    "content_text": "string | null",
    "content_transcription": "string | null",
    "hashtags": ["string"] | null,
    "media_duration": "int | null (seconds)",
    "author_id": "string | null",
    "author_name": "string | null",
    "author_username": "string | null",
    "author_avatar_url": "string | null",
    "author_is_verified": "boolean | null",
    "follower_count": "int",
    "overall_sentiment": "string",
    "overall_sentiment_score": "float",
    "overall_confidence": "float",
    "sentiment_probabilities": {
      "POSITIVE": "float",
      "NEUTRAL": "float",
      "NEGATIVE": "float"
    } | null,
    "primary_intent": "string",
    "intent_confidence": "float",
    "impact_score": "float",
    "risk_level": "string",
    "is_viral": "boolean",
    "is_kol": "boolean",
    "impact_breakdown": {
      "engagement_score": "float",
      "reach_score": "float",
      "platform_multiplier": "float",
      "sentiment_amplifier": "float",
      "raw_impact": "float"
    } | null,
    "aspects_breakdown": {
      "[aspect_name]": {
        "sentiment": "string",
        "score": "float",
        "confidence": "float",
        "keywords": ["string"]
      }
    } | null,
    "keywords": [
      {
        "keyword": "string",
        "aspect": "string",
        "sentiment": "string",
        "score": "float"
      }
    ] | null,
    "view_count": "int",
    "like_count": "int",
    "comment_count": "int",
    "share_count": "int",
    "save_count": "int",
    "published_at": "datetime",
    "analyzed_at": "datetime",
    "crawled_at": "datetime | null",
    "brand_name": "string | null",
    "keyword": "string | null",
    "job_id": "string | null",
    "comments": [
      {
        "id": "int",
        "comment_id": "string | null",
        "text": "string",
        "author_name": "string | null",
        "likes": "int | null",
        "sentiment": "string | null",
        "sentiment_score": "float | null",
        "commented_at": "datetime | null"
      }
    ],
    "comments_total": "int"
  }
}
```

#### Response Fields Description (Chi tiết bổ sung)

| Field                     | Type     | Description                             |
| ------------------------- | -------- | --------------------------------------- |
| `content_transcription`   | string   | Nội dung transcript từ video/audio      |
| `hashtags`                | array    | Danh sách hashtag trong bài             |
| `media_duration`          | int      | Thời lượng media (giây)                 |
| `follower_count`          | int      | Số follower của tác giả                 |
| `overall_confidence`      | float    | Độ tin cậy của phân tích sentiment      |
| `sentiment_probabilities` | object   | Xác suất cho từng loại sentiment        |
| `intent_confidence`       | float    | Độ tin cậy của phân tích intent         |
| `impact_breakdown`        | object   | Chi tiết cách tính impact score         |
| `aspects_breakdown`       | object   | Phân tích sentiment theo từng khía cạnh |
| `keywords`                | array    | Danh sách từ khóa được trích xuất       |
| `share_count`             | int      | Số lượt chia sẻ                         |
| `save_count`              | int      | Số lượt lưu                             |
| `crawled_at`              | datetime | Thời gian crawl dữ liệu                 |
| `comments`                | array    | Danh sách bình luận                     |
| `comments_total`          | int      | Tổng số bình luận                       |

---

## 2. Summary API

### 2.1 GET /summary

Lấy thống kê tổng hợp cho dashboard overview.

#### Request Parameters

| Parameter    | Type     | Required | Default | Description              |
| ------------ | -------- | -------- | ------- | ------------------------ |
| `project_id` | UUID     | ✅       | -       | ID của project           |
| `brand_name` | string   | ❌       | null    | Lọc theo tên thương hiệu |
| `keyword`    | string   | ❌       | null    | Lọc theo từ khóa         |
| `from_date`  | datetime | ❌       | null    | Ngày bắt đầu             |
| `to_date`    | datetime | ❌       | null    | Ngày kết thúc            |

#### Response

```json
{
  "success": true,
  "data": {
    "total_posts": "int",
    "total_comments": "int",
    "sentiment_distribution": {
      "POSITIVE": "int",
      "NEUTRAL": "int",
      "NEGATIVE": "int"
    },
    "avg_sentiment_score": "float",
    "risk_distribution": {
      "LOW": "int",
      "MEDIUM": "int",
      "HIGH": "int",
      "CRITICAL": "int"
    },
    "intent_distribution": {
      "[intent_name]": "int"
    },
    "platform_distribution": {
      "[platform_name]": "int"
    },
    "engagement_totals": {
      "views": "int",
      "likes": "int",
      "comments": "int",
      "shares": "int",
      "saves": "int"
    },
    "viral_count": "int",
    "kol_count": "int",
    "avg_impact_score": "float"
  }
}
```

#### Response Fields Description

| Field                    | Type   | Description                         |
| ------------------------ | ------ | ----------------------------------- |
| `total_posts`            | int    | Tổng số bài viết                    |
| `total_comments`         | int    | Tổng số bình luận                   |
| `sentiment_distribution` | object | Phân bố bài viết theo sentiment     |
| `avg_sentiment_score`    | float  | Điểm sentiment trung bình           |
| `risk_distribution`      | object | Phân bố bài viết theo mức độ rủi ro |
| `intent_distribution`    | object | Phân bố bài viết theo intent        |
| `platform_distribution`  | object | Phân bố bài viết theo nền tảng      |
| `engagement_totals`      | object | Tổng các chỉ số engagement          |
| `viral_count`            | int    | Số bài viral                        |
| `kol_count`              | int    | Số bài từ KOL                       |
| `avg_impact_score`       | float  | Điểm impact trung bình              |

---

## 3. Trends API

### 3.1 GET /trends

Lấy dữ liệu xu hướng theo thời gian cho biểu đồ dashboard.

#### Request Parameters

| Parameter     | Type     | Required | Default | Description                  |
| ------------- | -------- | -------- | ------- | ---------------------------- |
| `project_id`  | UUID     | ✅       | -       | ID của project               |
| `brand_name`  | string   | ❌       | null    | Lọc theo tên thương hiệu     |
| `keyword`     | string   | ❌       | null    | Lọc theo từ khóa             |
| `granularity` | string   | ❌       | "day"   | Độ chi tiết (day/week/month) |
| `from_date`   | datetime | ✅       | -       | Ngày bắt đầu                 |
| `to_date`     | datetime | ✅       | -       | Ngày kết thúc                |

#### Response

```json
{
  "success": true,
  "data": {
    "granularity": "day | week | month",
    "items": [
      {
        "date": "YYYY-MM-DD",
        "post_count": "int",
        "comment_count": "int",
        "avg_sentiment_score": "float",
        "avg_impact_score": "float",
        "sentiment_breakdown": {
          "POSITIVE": "int",
          "NEUTRAL": "int",
          "NEGATIVE": "int"
        },
        "total_views": "int",
        "total_likes": "int",
        "viral_count": "int",
        "crisis_count": "int"
      }
    ]
  }
}
```

#### Response Fields Description

| Field                         | Type   | Description                        |
| ----------------------------- | ------ | ---------------------------------- |
| `granularity`                 | string | Độ chi tiết thời gian              |
| `items`                       | array  | Danh sách data points              |
| `items[].date`                | string | Ngày (format YYYY-MM-DD)           |
| `items[].post_count`          | int    | Số bài viết trong khoảng thời gian |
| `items[].comment_count`       | int    | Số bình luận                       |
| `items[].avg_sentiment_score` | float  | Điểm sentiment trung bình          |
| `items[].avg_impact_score`    | float  | Điểm impact trung bình             |
| `items[].sentiment_breakdown` | object | Phân bố sentiment                  |
| `items[].total_views`         | int    | Tổng lượt xem                      |
| `items[].total_likes`         | int    | Tổng lượt thích                    |
| `items[].viral_count`         | int    | Số bài viral                       |
| `items[].crisis_count`        | int    | Số bài có intent CRISIS            |

---

## 4. Keywords API

### 4.1 GET /top-keywords

Lấy danh sách từ khóa phổ biến nhất với phân tích sentiment. Có thể kèm theo thông tin xếp hạng của nhiều keywords cụ thể.

#### Request Parameters

| Parameter          | Type     | Required | Default | Description                                                                    |
| ------------------ | -------- | -------- | ------- | ------------------------------------------------------------------------------ |
| `project_id`       | UUID     | ✅       | -       | ID của project                                                                 |
| `brand_name`       | string   | ❌       | null    | Lọc theo tên thương hiệu                                                       |
| `keyword`          | string   | ❌       | null    | Lọc theo từ khóa crawl                                                         |
| `from_date`        | datetime | ❌       | null    | Ngày bắt đầu                                                                   |
| `to_date`          | datetime | ❌       | null    | Ngày kết thúc                                                                  |
| `limit`            | int      | ❌       | 20      | Số lượng keywords trả về (1-50)                                                |
| `include_rank_for` | string   | ❌       | null    | Danh sách keywords cần tìm xếp hạng, phân cách bằng dấu phẩy (comma-separated) |

#### Response

```json
{
  "success": true,
  "data": {
    "keywords": [
      {
        "keyword": "string",
        "count": "int",
        "avg_sentiment_score": "float",
        "aspect": "string",
        "sentiment_breakdown": {
          "POSITIVE": "int",
          "NEUTRAL": "int",
          "NEGATIVE": "int"
        }
      }
    ],
    "input_keyword_ranks": [
      {
        "keyword": "string",
        "rank": "int | null",
        "count": "int",
        "avg_sentiment_score": "float",
        "in_top": "boolean"
      }
    ]
  }
}
```

#### Response Fields Description

| Field                                       | Type       | Description                                             |
| ------------------------------------------- | ---------- | ------------------------------------------------------- |
| `keywords`                                  | array      | Danh sách top keywords                                  |
| `keywords[].keyword`                        | string     | Từ khóa                                                 |
| `keywords[].count`                          | int        | Số lần xuất hiện                                        |
| `keywords[].avg_sentiment_score`            | float      | Điểm sentiment trung bình                               |
| `keywords[].aspect`                         | string     | Khía cạnh liên quan                                     |
| `keywords[].sentiment_breakdown`            | object     | Phân bố sentiment của keyword                           |
| `input_keyword_ranks`                       | array/null | Thông tin xếp hạng (chỉ có khi dùng `include_rank_for`) |
| `input_keyword_ranks[].keyword`             | string     | Keyword được tìm kiếm                                   |
| `input_keyword_ranks[].rank`                | int/null   | Vị trí xếp hạng (1-based), null nếu không tìm thấy      |
| `input_keyword_ranks[].count`               | int        | Số lần xuất hiện của keyword                            |
| `input_keyword_ranks[].avg_sentiment_score` | float      | Điểm sentiment trung bình                               |
| `input_keyword_ranks[].in_top`              | boolean    | Keyword có nằm trong top list không                     |

#### Example Usage

Lấy top 20 keywords và xếp hạng của "baocaosu" và "durex":

```
GET /top-keywords?project_id=xxx&limit=20&include_rank_for=baocaosu,durex
```

Response:

```json
{
  "success": true,
  "data": {
    "keywords": [
      {
        "keyword": "durex",
        "count": 500,
        "avg_sentiment_score": 0.7,
        "aspect": "product",
        "sentiment_breakdown": {
          "POSITIVE": 400,
          "NEUTRAL": 80,
          "NEGATIVE": 20
        }
      },
      {
        "keyword": "okamoto",
        "count": 450,
        "avg_sentiment_score": 0.65,
        "aspect": "product",
        "sentiment_breakdown": {
          "POSITIVE": 350,
          "NEUTRAL": 70,
          "NEGATIVE": 30
        }
      }
    ],
    "input_keyword_ranks": [
      {
        "keyword": "baocaosu",
        "rank": 5,
        "count": 320,
        "avg_sentiment_score": 0.65,
        "in_top": true
      },
      {
        "keyword": "durex",
        "rank": 1,
        "count": 500,
        "avg_sentiment_score": 0.7,
        "in_top": true
      }
    ]
  }
}
```

---

## 5. Alerts API

### 5.1 GET /alerts

Lấy các bài viết cần chú ý (critical, viral, crisis).

#### Request Parameters

| Parameter    | Type   | Required | Default | Description                 |
| ------------ | ------ | -------- | ------- | --------------------------- |
| `project_id` | UUID   | ✅       | -       | ID của project              |
| `brand_name` | string | ❌       | null    | Lọc theo tên thương hiệu    |
| `keyword`    | string | ❌       | null    | Lọc theo từ khóa            |
| `limit`      | int    | ❌       | 10      | Số item mỗi category (1-50) |

#### Response

```json
{
  "success": true,
  "data": {
    "critical_posts": [AlertPost],
    "viral_posts": [AlertPost],
    "crisis_intents": [AlertPost],
    "summary": {
      "critical_count": "int",
      "viral_count": "int",
      "crisis_count": "int"
    }
  }
}
```

#### AlertPost Schema

```json
{
  "id": "string",
  "content_text": "string | null (max 200 chars)",
  "risk_level": "string | null",
  "impact_score": "float",
  "overall_sentiment": "string",
  "primary_intent": "string | null",
  "is_viral": "boolean | null",
  "view_count": "int",
  "published_at": "datetime",
  "permalink": "string | null"
}
```

#### Response Fields Description

| Field                    | Type  | Description                       |
| ------------------------ | ----- | --------------------------------- |
| `critical_posts`         | array | Bài viết có risk_level = CRITICAL |
| `viral_posts`            | array | Bài viết viral                    |
| `crisis_intents`         | array | Bài viết có intent = CRISIS       |
| `summary.critical_count` | int   | Tổng số bài critical              |
| `summary.viral_count`    | int   | Tổng số bài viral                 |
| `summary.crisis_count`   | int   | Tổng số bài crisis                |

---

## 6. Errors API

### 6.1 GET /errors

Lấy danh sách lỗi crawl với phân trang.

#### Request Parameters

| Parameter    | Type     | Required | Default | Description               |
| ------------ | -------- | -------- | ------- | ------------------------- |
| `project_id` | UUID     | ✅       | -       | ID của project            |
| `job_id`     | string   | ❌       | null    | Lọc theo job ID           |
| `error_code` | string   | ❌       | null    | Lọc theo mã lỗi           |
| `from_date`  | datetime | ❌       | null    | Ngày bắt đầu              |
| `to_date`    | datetime | ❌       | null    | Ngày kết thúc             |
| `page`       | int      | ❌       | 1       | Số trang                  |
| `page_size`  | int      | ❌       | 20      | Số item mỗi trang (1-100) |

#### Response

```json
{
  "success": true,
  "data": [
    {
      "id": "int",
      "content_id": "string",
      "platform": "string",
      "error_code": "string",
      "error_category": "string",
      "error_message": "string | null",
      "permalink": "string | null",
      "job_id": "string",
      "created_at": "datetime"
    }
  ],
  "meta": {
    "page": "int",
    "page_size": "int",
    "total_items": "int",
    "total_pages": "int",
    "has_next": "boolean",
    "has_prev": "boolean"
  }
}
```

#### Response Fields Description

| Field            | Type     | Description            |
| ---------------- | -------- | ---------------------- |
| `id`             | int      | ID của error record    |
| `content_id`     | string   | ID của content bị lỗi  |
| `platform`       | string   | Nền tảng               |
| `error_code`     | string   | Mã lỗi                 |
| `error_category` | string   | Phân loại lỗi          |
| `error_message`  | string   | Thông báo lỗi chi tiết |
| `permalink`      | string   | Link đến content gốc   |
| `job_id`         | string   | ID của job crawl       |
| `created_at`     | datetime | Thời gian ghi nhận lỗi |

---

## 7. Health API

### 7.1 GET /health

Health check cơ bản.

#### Response

```json
{
  "status": "healthy"
}
```

---

### 7.2 GET /health/detailed

Health check chi tiết với trạng thái dependencies.

#### Response

```json
{
  "status": "healthy | unhealthy",
  "version": "string",
  "service": "string",
  "dependencies": {
    "database": "healthy | unhealthy"
  },
  "uptime": "string"
}
```

---

## 8. Common Schemas

### 8.1 Error Response

Tất cả API trả về format lỗi thống nhất khi có exception:

```json
{
  "success": false,
  "error": {
    "code": "string",
    "message": "string",
    "details": [
      {
        "field": "string",
        "message": "string",
        "code": "string"
      }
    ]
  },
  "meta": {
    "timestamp": "datetime",
    "request_id": "string",
    "version": "string"
  }
}
```

### 8.2 Error Codes

| Code      | Description           |
| --------- | --------------------- |
| `SYS_001` | Internal server error |
| `RES_001` | Resource not found    |
| `VAL_001` | Validation error      |

### 8.3 Enum Values

#### Sentiment

- `POSITIVE`
- `NEUTRAL`
- `NEGATIVE`

#### Risk Level

- `LOW`
- `MEDIUM`
- `HIGH`
- `CRITICAL`

#### Granularity

- `day`
- `week`
- `month`

#### Sort Order

- `asc`
- `desc`

---

## Response Headers

Tất cả response đều có header:

| Header         | Description                                    |
| -------------- | ---------------------------------------------- |
| `X-Request-ID` | UUID duy nhất cho mỗi request, dùng để tracing |

---

## Notes

1. Tất cả datetime đều ở format ISO 8601 (UTC)
2. `project_id` là required cho hầu hết các endpoint
3. Pagination mặc định: page=1, page_size=20
4. Content trong list view được truncate để tối ưu performance
