# CourseQA — 智能课程问答与多引擎信息检索系统

基于 **FastAPI + 多检索引擎（TF-IDF / BM25 / CountVectorizer）** 构建的轻量级课程知识库切分与问答检索系统，支持 Markdown 文档结构化语义分块、知识库无感热重载、批量并发检索、历史日志追踪导出与检索指标可视化看板。

## 核心架构与特性

- **结构化 Markdown 语义切分 (`core/qa_engine.py`)**：自动识别多级标题 (`#` / `##` / `###`) 与段落边界，保留文档章节上下文元信息。
- **多检索引擎动态切换**：
  - `TF-IDF + Cosine Similarity`：经典的词频-逆文档频率向量空间检索。
  - `BM25 概率检索模型`：引入词频饱和度 ($k_1=1.5$) 与文档长度归一化 ($b=0.75$)，有效抑制长文本偏置。
  - `CountVectorizer`：高召回词频向量基准对比。
- **工程化高可用接口 (`app.py`, `api/`)**：
  - 支持 `/api/reload` 在线热重载知识库，无需重启服务即可更新索引。
  - 支持 `/api/batch_search` 批量问题并发评测。
  - 提供 `/api/history/export` UTF-8-BOM CSV 报表导出与 `/api/stats` 零命中率、平均响应耗时监控。

## 快速启动

```bash
pip install -r requirements.txt
python app.py
```

浏览器访问：
- 检索主页：`http://127.0.0.1:8000/`
- 统计看板：`http://127.0.0.1:8000/dashboard`
- Swagger API 文档：`http://127.0.0.1:8000/docs`
