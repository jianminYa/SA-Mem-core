# Event Type Classification Update Summary

## 修改日期
2026-02-11

## 修改目标
统一事件时间类型分类为四类：**OCCURRENCE**, **STATE**, **ATTRIBUTE**, **INTENTION**，并确保分类逻辑一致，时间标签可用于构建索引。

## 修改的文件

### 1. build_prompts.py
**修改内容：**
- 更新了 CATEGORY DEFINITIONS 部分，使其更加清晰和准确
- 添加了详细的分类优先级规则（CLASSIFICATION PRIORITY RULES）
- 明确了关键区分点：
  - 状态改变 → 总是 OCCURRENCE
  - 年龄 → 总是 STATE（不是 ATTRIBUTE）
  - 关系改变 vs. 关系存在
  - 出生事件 vs. 出生日期
  - 不确定时优先选择 OCCURRENCE

**关键改进：**
- 每个类别都有明确的"核心语义"定义
- 添加了大量实例说明
- 时间逻辑保持不变，确保可构建索引

### 2. build_impl.py
**修改内容：**
- Line 423: 注释从 `OCCURRENCE/STATE/INTENTION/MISC` 改为 `OCCURRENCE/STATE/ATTRIBUTE/INTENTION`
- Line 473: 验证集合从 `{"OCCURRENCE", "STATE", "INTENTION", "MISC"}` 改为 `{"OCCURRENCE", "STATE", "ATTRIBUTE", "INTENTION"}`
- Line 477-483: 默认回退值从 `MISC` 改为 `ATTRIBUTE`

### 3. memblock_extractor.py
**修改内容：**
- Line 1041: `_coerce_temporal_type` 函数的验证集合从 `{"OCCURRENCE", "STATE", "INTENTION", "MISC"}` 改为 `{"OCCURRENCE", "STATE", "ATTRIBUTE", "INTENTION"}`
- Line 201-234: 完全重写了 `infer_event_type` 函数，添加了：
  - ATTRIBUTE 检测（name, MBTI, nationality, gender, birth date, blood type, ethnicity）
  - 更精确的 OCCURRENCE 检测（state changes, completed actions）
  - 更精确的 STATE 检测（包括年龄标记）
  - 优先级顺序：INTENTION > ATTRIBUTE > OCCURRENCE > STATE

### 4. retrieval_enhanced.py
**修改内容：**
- Line 217: 默认事件类型从 `"MISC"` 改为 `"ATTRIBUTE"`

## 分类规则总结

### OCCURRENCE (发生/完成/转变)
- **语义**: 一次性动作，实际发生的事件
- **示例**: "was born in 1998", "got married in 2020", "became CEO", "stopped smoking"
- **时间**: start_time/end_time 表示事件发生的时间

### STATE (状态/持续)
- **语义**: 可变的状态，在一段时间内为真
- **示例**: "is married", "works as teacher", "likes jazz", "is 30 years old"
- **时间**: start_time 是状态开始时间，end_time 是状态结束时间或 session_end_time（如果仍在持续）
- **关键**: 年龄总是 STATE

### ATTRIBUTE (属性/身份)
- **语义**: 稳定的、无时间性的身份事实
- **示例**: "name is Daniel", "MBTI is INTJ", "nationality is Canadian", "birthday is May 12"
- **时间**: 使用 session window 作为占位符
- **关键**: 与 OCCURRENCE 的区别在于是否有时间点

### INTENTION (意图/计划)
- **语义**: 面向未来的计划、目标、意图
- **示例**: "plans to retire in 2050", "wants to visit Japan", "will apply next year"
- **时间**: start_time/end_time 表示意图被观察到的时间（通常是 session window）

## 优先级规则
1. 状态改变 → 总是 OCCURRENCE
2. 年龄 → 总是 STATE
3. 关系改变是 OCCURRENCE，关系存在是 STATE
4. 出生事件是 OCCURRENCE，出生日期是 ATTRIBUTE
5. 不确定时优先选择 OCCURRENCE

## 测试结果
创建了 test_event_classification.py，包含 30 个测试用例：
- OCCURRENCE: 10 个测试 ✓
- STATE: 9 个测试 ✓
- ATTRIBUTE: 6 个测试 ✓
- INTENTION: 5 个测试 ✓

**测试结果: 30/30 通过 (100%)**

## 时间索引兼容性
所有事件类型都有明确的 start_time 和 end_time 定义：
- OCCURRENCE: 事件发生的具体时间
- STATE: 状态的开始和结束时间
- ATTRIBUTE: Session window 作为占位符
- INTENTION: 意图被观察到的时间

这些时间标签都是 ISO 8601 格式（YYYY-MM-DD 或 YYYY-MM-DDTHH:MM:SS），可以用于构建 interval tree 索引。

## 向后兼容性
- 旧的 MISC 类型会被自动转换为 ATTRIBUTE（通过 _coerce_temporal_type 函数）
- 现有的记忆块不需要重新构建（除非需要更精确的分类）
- 检索系统会自动使用新的分类逻辑
