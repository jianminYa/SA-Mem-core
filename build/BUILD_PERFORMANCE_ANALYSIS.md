# 构建过程运行时间长的原因分析和优化方案

## 🔍 问题分析

### 主要瓶颈

运行 `build_stage.py` 时间长的主要原因：

1. **LLM API 调用次数多**
   - 每个 memory block 调用 1 次 LLM（提取 keywords, topic, events）
   - 每个 event 调用 1 次 LLM（分类标签）
   - 假设 1 session 有 20 条消息 → 4 个 blocks → 每个 block 4 个 events
   - 总计：4 + (4 × 4) = **20 次 LLM 调用**

2. **超时和重试配置过于保守**
   ```python
   TOPIC_CLASSIFY_TIMEOUT = 60.0  # 每次调用最多等 60 秒
   TOPIC_CLASSIFY_MAX_RETRIES = 4  # 失败后重试 4 次
   ```
   - 如果 API 慢或超时，每个调用最多需要：60s × 4 = **240 秒（4 分钟）**
   - 如果 5 个调用超时：5 × 240s = **1200 秒（20 分钟）**

3. **API 端点可能较慢**
   - BASE_URL: `https://tao.plus7.plus/v1`
   - 网络延迟或端点响应慢

4. **事件标签分类是最大瓶颈**
   - 每个 event 都要调用 LLM 获取分类标签
   - 如果有 16 个 events，就是 16 次额外的 LLM 调用

## 💡 优化方案

### 方案 1: 调整超时和重试参数（推荐）

修改 `memblock_extractor.py` 中的 Config 类：

```python
class Config:
    # 原配置
    TOPIC_CLASSIFY_TIMEOUT = 60.0
    TOPIC_CLASSIFY_MAX_RETRIES = 4

    # 优化后
    TOPIC_CLASSIFY_TIMEOUT = 15.0  # 60 → 15 秒
    TOPIC_CLASSIFY_MAX_RETRIES = 2  # 4 → 2 次
```

**效果：**
- 每次超时调用从 240 秒降至 45 秒（15s × 3 次尝试）
- 降低 80% 的等待时间

### 方案 2: 临时禁用事件标签分类（快速测试）

在 `build_impl.py` 中注释掉标签分类调用：

```python
# 原代码（第 439-443 行）
labels = self.worker.classify_event_labels(
    desc,
    top_k=getattr(mx.Config, "EVENT_LABEL_TOP_K", 3),
    with_meta=True,
)

# 临时禁用（用于快速测试）
labels = []  # 跳过 LLM 调用
```

**效果：**
- 减少约 80% 的 LLM 调用
- 构建速度提升 5-10 倍
- 注意：生成的 memory blocks 将没有事件标签

### 方案 3: 批量处理事件标签（需要修改代码）

将多个 events 合并为一次 LLM 调用：

```python
# 当前：每个 event 调用 1 次
for event in events:
    labels = classify_event_labels(event)

# 优化：批量调用
all_events = [event1, event2, event3, ...]
all_labels = classify_event_labels_batch(all_events)
```

**效果：**
- 减少 LLM 调用次数 80%
- 需要修改 prompt 和解析逻辑

### 方案 4: 添加详细进度日志

在 `build_impl.py` 中添加日志，了解卡在哪里：

```python
# 在 extract_box_from_dialog 开始时
mx.logger.info(f"📦 Processing box {box_id}...")

# 在 classify_event_labels 调用前后
mx.logger.info(f"🏷️  Classifying event {event_idx}/{total_events}: {desc[:50]}...")
start_time = time.time()
labels = self.worker.classify_event_labels(...)
elapsed = time.time() - start_time
mx.logger.info(f"✓ Event {event_idx} classified in {elapsed:.2f}s")
```

## 🚀 立即可用的优化

### 快速方案：修改配置参数

创建一个快速配置文件 `build_stage_fast.py`：

```python
from memblock_cli import main
from memblock_extractor import Config

# 覆盖配置
Config.TOPIC_CLASSIFY_TIMEOUT = 15.0  # 减少超时
Config.TOPIC_CLASSIFY_MAX_RETRIES = 2  # 减少重试
Config.EVENT_LABEL_TOP_K = 1  # 减少标签数量

if __name__ == "__main__":
    main(default_stage="build")
```

运行：
```bash
python build_stage_fast.py \
  --run-id preview \
  --raw-data-file /data/cwb/prototype/data/processed_halumem/1b846c59-456e-9550-dffd-4d23f1eca81f.json \
  --limit-conversations 1 \
  --limit-sessions 1
```

### 最快方案：跳过事件标签分类

如果只是想快速测试构建流程，可以临时禁用事件标签分类。

## 📊 预期效果

| 方案 | LLM 调用次数 | 预计时间（正常） | 预计时间（API 慢） |
|------|-------------|-----------------|-------------------|
| 原配置 | 20 次 | 1-2 分钟 | 10-20 分钟 |
| 方案 1（调整超时） | 20 次 | 1-2 分钟 | 3-5 分钟 |
| 方案 2（禁用标签） | 4 次 | 10-20 秒 | 1-2 分钟 |
| 方案 3（批量处理） | 8 次 | 30-40 秒 | 2-3 分钟 |

## 🔧 调试建议

1. **检查当前卡在哪里**
   - 查看日志输出，看是否有 "Building Sample 0..." 后的进度
   - 如果长时间没有新日志，说明某个 LLM 调用超时

2. **测试 API 响应速度**
   ```bash
   time curl -X POST https://tao.plus7.plus/v1/chat/completions \
     -H "Authorization: Bearer $OPENAI_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}'
   ```

3. **查看实时日志**
   ```bash
   tail -f out/preview/token_stream.jsonl
   tail -f out/preview/trace_build_process.jsonl
   ```

## 总结

**最可能的原因：** 事件标签分类调用超时并重试（60s × 4 次 = 240s per event）

**推荐方案：**
1. 立即：调整 TOPIC_CLASSIFY_TIMEOUT 和 TOPIC_CLASSIFY_MAX_RETRIES
2. 测试：临时禁用事件标签分类，验证是否是瓶颈
3. 长期：实现批量事件标签分类
