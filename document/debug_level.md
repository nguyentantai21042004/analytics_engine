| Muốn log (%) | DEBUG_RAW_DATA | DEBUG_SAMPLE_RATE   |
|-------------|---------------|---------------------|
| 100         | "true"        | (không cần)         |
| 50          | "sample"      | "2"                 |
| 10          | "sample"      | "10"                |
| 1           | "sample"      | "100"               |
| 0           | "false"       | (không cần)         |

**Công thức:**  
`DEBUG_SAMPLE_RATE = 100 / (phần trăm muốn log)`

**Ví dụ:** Muốn log 50%:

- `DEBUG_RAW_DATA`: `"sample"`
- `DEBUG_SAMPLE_RATE`: `"2"`  <!-- 100/50 = 2 -->