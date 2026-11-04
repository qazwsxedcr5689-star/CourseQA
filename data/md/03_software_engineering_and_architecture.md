# 软件工程与系统架构核心讲义

## 面向对象设计 SOLID 五大基本原则
高质量软件架构设计遵循 SOLID 五大原则：
1. 单一职责原则（SRP）：一个类或模块只应有一个引起它变化的原因，高内聚、低耦合。
2. 开放封闭原则（OCP）：软件实体应当对扩展开放，对修改关闭，通过抽象接口与多态插件化扩展新检索引擎。
3. 里氏替换原则（LSP）：子类对象必须能够透明地替换其基类对象而不破坏程序正确性。
4. 接口隔离原则（ISP）：客户端不应被迫依赖它不使用的方法，推崇细粒度的专用接口。
5. 依赖倒置原则（DIP）：高层业务模块不应直接依赖低层实现细节，二者都应依赖于抽象接口。

## 经典设计模式在工程中的应用 (Design Patterns)
GoF 23 种设计模式可分为创建型、结构型与行为型三大类：
- 策略模式（Strategy Pattern）：将不同算法（如 TF-IDF、BM25、CountVectorizer、Hybrid RRF）封装为可互换的策略分支，使检索算法可随请求参数动态切换。
- 工厂模式与单例模式（Factory & Singleton）：统一管理全局检索引擎实例与词袋矩阵缓存，避免每次请求重复加载分词词典。
- 观察者模式（Observer Pattern）：在状态变更（如热重载知识库、订单状态流转）时自动通知关联模块更新统计指标或清除缓存。
- 装饰器与上下文管理器（Decorator & Context Manager）：利用 Python `with Timer() as t:` 无侵入地切面统计接口毫秒级耗时并写入审计日志。

## RESTful API 规范与 FastAPI 异步架构
现代 Web 后端广泛采用 RESTful 架构风格：使用名词复数表示资源路径（如 `/api/history`、`/api/corpus`），通过 HTTP 动词表达语义操作（GET 查询、POST 创建/触发计算、PUT 全量更新、DELETE 删除），并配合标准状态码（200 OK、201 Created、400 Bad Request、404 Not Found、422 Validation Error）。

FastAPI 基于 Starlette 与 Pydantic 构建，利用 Python 类型注解（Type Hints）在运行时自动完成 JSON 请求体校验、序列化以及 OpenAPI（Swagger UI）交互式文档生成，兼具极高的开发效率与接近 Go/Node.js 的高并发吞吐性能。

## UML 统一建模语言核心视图
在软件需求分析与系统设计阶段，常用 UML 图包括：
- 用例图（Use Case Diagram）：从外部参与者（Actor）视角描述系统功能边界，包含关联、包含（include）、扩展（extend）与泛化关系。
- 类图（Class Diagram）：刻画系统静态结构，体现类之间的泛化（继承）、实现、关联、聚合（空心菱形，整体与部分可分离）与组合（实心菱形，强生命周期绑定）关系。
- 时序图（Sequence Diagram）与状态机图（State Diagram）：按时间轴展现对象间的消息交互生命线，或刻画业务实体（如电商订单从已确认、已发货到已完成/已取消）的状态迁移条件。

## 软件测试体系与知识库无感热重载
软件质量保障体系涵盖单元测试（Unit Test）、集成测试、系统测试与回归测试。白盒测试关注代码语句覆盖、分支覆盖与路径覆盖；黑盒测试采用等价类划分与边界值分析验证接口行为。在线服务高可用设计中，通过提供 `/api/reload` 热重载接口，可在新增或修改 Markdown 课程文档后实时重建内存倒排索引与 TF-IDF 矩阵，实现零停机（Zero-Downtime）知识更新。
