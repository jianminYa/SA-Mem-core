# 如何使用断点续传功能运行构建

## 问题

当前命令使用 `--raw-data-file` 模式，不支持断点续传：
```bash
python build_stage_fast.py \
  --run-id Halumem-1b846c59 \
  --raw-data-file /data/cwb/prototype/data/processed_halumem/1b846c59-456e-9550-dffd-4d23f1eca81f.json
```

## 解决方案

使用 `--raw-data-dir` 模式，支持断点续传和 `--resume` 参数。

### 方案 1: 使用目录模式（推荐）

```bash
# 首次运行
python build_stage_fast.py \
  --run-id Halumem-1b846c59 \
  --raw-data-dir /data/cwb/prototype/data/processed_halumem \
  --raw-data-glob "1b846c59-*.json"

# 如果中断，使用 --resume 继续
python build_stage_fast.py \
  --run-id Halumem-1b846c59 \
  --raw-data-dir /data/cwb/prototype/data/processed_halumem \
  --raw-data-glob "1b846c59-*.json" \
  --resume
```

### 方案 2: 创建临时目录（如果只有一个文件）

```bash
# 1. 创建临时目录并复制文件
mkdir -p /tmp/halumem_build
cp /data/cwb/prototype/data/processed_halumem/1b846c59-456e-9550-dffd-4d23f1eca81f.json /tmp/halumem_build/

# 2. 首次运行
python build_stage_fast.py \
  --run-id Halumem-1b846c59 \
  --raw-data-dir /tmp/halumem_build \
  --raw-data-glob "*.json"

# 3. 如果中断，使用 --resume 继续
python build_stage_fast.py \
  --run-id Halumem-1b846c59 \
  --raw-data-dir /tmp/halumem_build \
  --raw-data-glob "*.json" \
  --resume
```

## 断点续传功能说明

### 自动保存检查点

系统会在每个文件处理完成后自动保存检查点到：
```
out/Halumem-1b846c59/build_checkpoint.json
```

检查点包含：
- `next_file_index`: 下一个要处理的文件索引
- `processed_files`: 已处理的文件列表
- `sample_id_next`: 下一个 sample ID
- `updated_at`: 最后更新时间

### 查看检查点

```bash
python build_stage_fast.py \
  --run-id Halumem-1b846c59 \
  --show-checkpoint
```

### 手动指定起始位置

如果不想使用自动恢复，可以手动指定：

```bash
# 从第 5 个文件开始处理
python build_stage_fast.py \
  --run-id Halumem-1b846c59 \
  --raw-data-dir /data/cwb/prototype/data/processed_halumem \
  --uuid-start 5

# 只处理 10 个文件
python build_stage_fast.py \
  --run-id Halumem-1b846c59 \
  --raw-data-dir /data/cwb/prototype/data/processed_halumem \
  --uuid-start 5 \
  --uuid-count 10
```

### 禁用自动恢复

```bash
# 强制从头开始，忽略检查点
python build_stage_fast.py \
  --run-id Halumem-1b846c59 \
  --raw-data-dir /data/cwb/prototype/data/processed_halumem \
  --no-resume
```

## 处理中断的情况

### 情况 1: API Token 不够

系统会抛出异常并保存检查点。重新运行时使用 `--resume`：

```bash
python build_stage_fast.py \
  --run-id Halumem-1b846c59 \
  --raw-data-dir /data/cwb/prototype/data/processed_halumem \
  --resume
```

### 情况 2: 网络连接断开

同样会保存检查点。重新运行时使用 `--resume`。

### 情况 3: 手动中断 (Ctrl+C)

可能不会保存检查点。建议查看已处理的文件：

```bash
# 查看检查点
python build_stage_fast.py \
  --run-id Halumem-1b846c59 \
  --show-checkpoint

# 从检查点继续
python build_stage_fast.py \
  --run-id Halumem-1b846c59 \
  --raw-data-dir /data/cwb/prototype/data/processed_halumem \
  --resume
```

## 监控进度

### 查看输出文件

```bash
# 查看已生成的 memory blocks 数量
wc -l out/Halumem-1b846c59/final_boxes_content.jsonl

# 实时监控
tail -f out/Halumem-1b846c59/build_stats.jsonl
```

### 查看日志

```bash
# 查看 token 使用情况
tail -f out/Halumem-1b846c59/token_stream.jsonl

# 查看构建过程
tail -f out/Halumem-1b846c59/trace_build_process.jsonl
```

## 完整示例

```bash
# 1. 首次运行（处理所有 sessions）
python build_stage_fast.py \
  --run-id Halumem-1b846c59 \
  --raw-data-dir /data/cwb/prototype/data/processed_halumem \
  --raw-data-glob "1b846c59-*.json"

# 2. 如果中断，查看检查点
python build_stage_fast.py \
  --run-id Halumem-1b846c59 \
  --show-checkpoint

# 3. 从检查点继续
python build_stage_fast.py \
  --run-id Halumem-1b846c59 \
  --raw-data-dir /data/cwb/prototype/data/processed_halumem \
  --raw-data-glob "1b846c59-*.json" \
  --resume

# 4. 完成后查看结果
wc -l out/Halumem-1b846c59/final_boxes_content.jsonl
cat out/Halumem-1b846c59/build_stats.jsonl
```

## 注意事项

1. **必须使用 `--raw-data-dir` 模式** 才能使用断点续传
2. **`--resume` 参数** 会自动从上次中断的地方继续
3. **检查点文件** 保存在 `out/<run-id>/build_checkpoint.json`
4. **每个文件处理完成后** 都会保存检查点
5. **如果文件内有多个 conversations**，当前实现是按文件级别保存检查点，不是按 conversation 级别

## 如果只有一个文件怎么办？

如果你的数据集只有一个大文件，建议：

1. **拆分文件**（推荐）：
   ```python
   # 将大文件拆分为多个小文件
   import json

   with open('big_file.json', 'r') as f:
       data = json.load(f)

   # 每 10 个 conversations 一个文件
   chunk_size = 10
   for i in range(0, len(data), chunk_size):
       chunk = data[i:i+chunk_size]
       with open(f'chunk_{i//chunk_size}.json', 'w') as f:
           json.dump(chunk, f)
   ```

2. **使用单文件模式但接受无法断点续传**：
   - 设置更短的超时时间
   - 减少重试次数
   - 使用 `--limit-conversations` 分批处理
