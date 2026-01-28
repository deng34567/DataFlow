# API Pipelines 使用说明文档

本文档介绍 DataFlow 项目中三个基于 API 的 QA 生成 Pipeline 的使用方法。

---

## 目录

- [新增Pipeline 概览](#新增pipeline-概览)
- [1. PDF Table QA Pipeline](#1-pdf-table-qa-pipeline)
- [2. Markdown Table QA Pipeline](#2-markdown-table-qa-pipeline)
- [3. Policy QA Pipeline](#3-policy-qa-pipeline)
- [通用配置说明](#通用配置说明)
- [输出格式说明](#输出格式说明)

---

## Pipeline 概览

| Pipeline | 输入类型 | 主要用途 | 输出位置 |
|---------|----------|---------|---------|
| `pdf_table_qa_pipeline.py` | PDF 文件（JSONL列表） | 从 PDF 中提取表格并生成 Q/A | `.cache/pdf_table_qa/` |
| `pdf_table_qa_pipeline_markdown.py` | Markdown 文件目录 | 从 Markdown 表格生成 Q/A | `.cache/markdown_table_qa/` |
| `policy_qa_pipeline.py` | Markdown 文件目录 | 从政策法规文本生成 Q/A | `.cache/policy_qa/` |

三个 Pipeline 均采用 **Map-Reduce** 架构处理长文档：
1. **Map 阶段**：将文档拆分为小块（chunks），为每个块生成摘要
2. **Reduce 阶段**：基于摘要规划并生成高质量 Q/A 对

---

## 1. PDF Table QA Pipeline

**文件路径**: `pdf_table_qa_pipeline.py`

### 功能描述
从 PDF 文件中提取表格数据，并基于表格内容生成问答对。

### 使用步骤

1. **准备输入文件**
   创建 JSONL 文件，每行包含 PDF 的路径或 URL：
   ```json
   {"source": "/path/to/your/document1.pdf"}
   {"source": "/path/to/your/document2.pdf"}
   ```

2. **修改配置**（在 `__init__` 方法中）
   ```python
   self.storage = FileStorage(
       first_entry_file_name="你的JSONL文件路径.jsonl",  # 修改为你的输入文件
       cache_path="./.cache/pdf_table_qa",
       ...
   )
   ```

3. **运行 Pipeline**
   ```bash
   python pdf_table_qa_pipeline.py
   ```

### 运行模式

```python
pipeline = PDFTableQA_APIPipeline()
pipeline.forward(use_map_reduce=True, merge_mode=True)
```

| 参数 | 说明 |
|-----|------|
| `use_map_reduce=True` | 使用 Map-Reduce 模式处理大文档（推荐） |
| `use_map_reduce=False` | 直接处理，适用于小文档 |
| `merge_mode=True` | 合并所有文档后统一生成 Q/A |
| `merge_mode=False` | 对每个文档单独生成 Q/A |

### 处理流程
```
PDF 文件 → PDF 转 Markdown → 按表格边界分块 → Map-Reduce 生成 Q/A
```
注意：除了需要在环境变量中设置export DF_API_KEY=sk-
      还需要设置export MINERU_API_KEY=""

---

## 2. Markdown Table QA Pipeline

**文件路径**: `pdf_table_qa_pipeline_markdown.py`

### 功能描述
直接从 Markdown 文件中的表格生成问答对，无需 PDF 转换步骤。

### 使用步骤

1. **准备 Markdown 文件**
   将包含表格的 Markdown 文件放入目录中。

2. **运行 Pipeline**
   ```python
   # 方式1：使用默认目录
   pipeline = MarkdownTableQA_APIPipeline()
   
   # 方式2：指定自定义目录（推荐）
   markdown_dir = "/path/to/your/markdown/files"
   pipeline = MarkdownTableQA_APIPipeline(markdown_dir=markdown_dir)
   
   pipeline.forward(use_map_reduce=True, merge_mode=True)
   ```

3. **直接运行脚本**
   ```bash
   # 修改文件中的 markdown_dir 变量后运行
   python pdf_table_qa_pipeline_markdown.py
   ```

### 关键配置

```python
# 在文件 main 部分修改
markdown_dir = "/home/wangdeng/Github/dataflow/DataFlow/dataflow/example/markdown/Cleaned/Cleaned/05_Nuclear_Radiation/Standards"
pipeline = MarkdownTableQA_APIPipeline(markdown_dir=markdown_dir)
```

### 输出特性
- 输出文件名自动包含目录名（如 `05_Nuclear_Radiation_Standards_qa_pairs.json`）
- 不同目录的结果不会相互覆盖

### 主要参数

| 类/参数 | 默认值 | 说明 |
|---------|--------|------|
| `max_chunk_size` | 20000 | 单个块最大字符数 |
| `max_qa` | 100 | 最大生成 Q/A 对数量 |
| `lang` | "zh" | 语言（"zh" 中文 / "en" 英文） |

注意：除了需要在环境变量中设置export DF_API_KEY=sk-

---

## 3. Policy QA Pipeline

**文件路径**: `policy_qa_pipeline.py`

### 功能描述
专门针对政策法规类文本生成问答对。与表格 QA 不同，此 Pipeline 关注：
- 法律法规条款
- 政策要点
- 标准规范
- 合规性问题

### 使用步骤

1. **准备政策法规 Markdown 文件**
   将 `.md` 文件放入指定目录。

2. **运行 Pipeline**
   ```python
   # 方式1：使用默认目录
   pipeline = PolicyQA_APIPipeline()
   
   # 方式2：指定自定义目录
   markdown_dir = "/path/to/your/policy/markdown/files"
   pipeline = PolicyQA_APIPipeline(markdown_dir=markdown_dir)
   
   # 运行
   pipeline.forward(merge_mode=True, max_files=None)
   ```

3. **直接运行脚本**
   ```bash
   python policy_qa_pipeline.py
   ```

### 运行参数

```python
pipeline.forward(merge_mode=True, max_files=None)
```

| 参数 | 说明 |
|-----|------|
| `merge_mode=True` | 合并所有文档后生成 Q/A（推荐） |
| `merge_mode=False` | 对每个文档单独生成 Q/A |
| `max_files=5` | 限制处理文件数量（用于测试） |
| `max_files=None` | 处理全部文件 |

### 配置参数

```python
# 块大小配置（比表格 QA 更小，适合政策文本）
self.chunk_splitter = TableChunkSplitter(
    max_chunk_size=8000,   # 政策文本使用更小块
    min_chunk_size=500,
    context_before=300,
    context_after=300,
)

# Q/A 生成配置
self.policy_qa_generator = PolicyQAGenerator(
    llm_serving=self.llm_serving,
    lang="zh",
    max_qa=100,            # 最大 Q/A 对数量
    max_relevant_chunks=5,
)
```

---

## 通用配置说明

### LLM API 配置

三个 Pipeline 都使用相同的 API 调用方式：

```python
self.llm_serving = APILLMServing_request(
    api_url="https://oneapi.hkgai.net/v1/chat/completions",
    model_name="kimi-k2",      # 或 "qwen3next"
    max_workers=4,             # 并发数
    temperature=0.3,           # 生成温度
    timeout=(10.0, 300.0),     # (连接超时, 读取超时)
)
```

### 环境准备

在运行 Pipeline 前，确保已设置 API Key：
```bash
export OPENAI_API_KEY="your-api-key-here"
conda activate your-env-name
```

### 目录结构

```
.cache/
├── pdf_table_qa/           # PDF Table QA 输出
│   ├── tables/             # 提取的表格 CSV
│   └── *.json              # 结果文件
├── markdown_table_qa/      # Markdown Table QA 输出
│   ├── {prefix}/
│   │   ├── tables/
│   │   └── {prefix}_qa_pairs.json
└── policy_qa/              # Policy QA 输出
    └── {prefix}/
        ├── {prefix}_policy_qa_result.json
        └── {prefix}_qa_pairs.json
```

注意：除了需要在环境变量中设置export DF_API_KEY=sk-

---

## 输出格式说明

### Q/A 对格式

所有 Pipeline 生成的 Q/A 对采用统一的 JSON 格式：

```json
[
  {
    "question": "问题内容",
    "answer": "答案内容"
  },
  ...
]
```

### 完整结果格式

```json
[
  {
    "source": "文件来源",
    "text_length": 12345,
    "num_chunks": 10,
    "chunks": ["块1内容", "块2内容", ...],
    "qa_pairs": [...],
    "chunk_summaries": ["摘要1", "摘要2", ...],
    "qa_plans": ["计划1", "计划2", ...]
  }
]
```

---

## 常见问题

### 1. 处理超时
增加 `timeout` 参数：
```python
timeout=(10.0, 600.0)  # 增加读取超时至 600 秒
```

### 2. 生成的 Q/A 数量不足
- 增加 `max_qa` 参数
- 检查文档内容是否包含足够的可问信息

### 3. 不同目录结果覆盖
Pipeline 会自动根据目录名生成唯一前缀，正常情况下不会覆盖。

---

## 快速开始示例

```python
# Markdown 表格 QA
from dataflow.statics.pipelines.api_pipelines.pdf_table_qa_pipeline_markdown import MarkdownTableQA_APIPipeline

pipeline = MarkdownTableQA_APIPipeline(
    markdown_dir="/your/markdown/path"
)
pipeline.forward(use_map_reduce=True, merge_mode=True)

# 政策法规 QA
from dataflow.statics.pipelines.api_pipelines.policy_qa_pipeline import PolicyQA_APIPipeline

pipeline = PolicyQA_APIPipeline(
    markdown_dir="/your/policy/markdown/path"
)
pipeline.forward(merge_mode=True)
```
