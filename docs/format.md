## Các trường

- `event`: `round_start`, `client_step`, `round_end`
- `server_round`: số vòng hiện tại, bắt đầu từ 1
- `phase`: `train`, `evaluate`, `round`
- `status`: trạng thái của event
- `node_id`: ID client (số nguyên); event của round dùng `null`
- `metrics`: các chỉ số kèm theo, có thể là `{}`
- `timestamp`: Unix timestamp, tính bằng giây
- `schema`: `fl-event/v1`

## Event và trạng thái

| Event | Phase | Status |
| --- | --- | --- |
| `round_start` | `train` | `started` |
| `client_step` | `train` | `training`, `uploading`, `failed` |
| `client_step` | `evaluate` | `evaluating`, `online`, `failed` |
| `round_end` | `round` | `completed` |

## Metrics

- Kết quả client: `loss`, `accuracy`, `num-examples`, `partition-id`. Nếu gặp lỗi có thể có `error`.
- Kết quả round: `train_loss`, `evaluate_loss`, `evaluate_accuracy` và các bộ đếm `train_ok`, `train_failed`, `evaluate_ok`, `evaluate_failed`.
- `accuracy` dùng giá trị 0–1. Metric nào không có thì để thiếu, không tự điền `0`.
- Ngoài các tên trên vẫn có thể xuất hiện metric khác, nên `metrics` để dạng object mở.