# 两个进程之间的现有兼容契约

资料目录端 `GET /packets/{revision}` 返回 `packet` 与规范 JSON 的 `sha256`；不存在的版本返回 404。分析端 `POST /analyses` 接收 `revision`、已登记的 `analysis` 和可选对象 `options`，返回 201、`id` 与完整 `envelope`。分析端先检查目录返回的版本与摘要，再进行领域校验和分析。

`GET /results/{id}` 返回创建时保存的 envelope，不重新读取目录或重算。envelope 包含 analysis、analysis_version、revision、source_sha256、options、result；所有 JSON 数值必须有限，无穷区间须用显式边界状态表示。id 是内容摘要，完整相同请求及结果幂等，改变情景产生不同对象。无效分析和输入为 422，来源不可取得为 503，未知结果为 404。

资料服务与分析服务分别运行在同一个应用容器的回环端口。现有 HTTP 数据契约、不可变结果和资料摘要语义是兼容要求，不限定新统计组件的内部模块结构。`app.demo` 启动真实两端、创建分析并从结果端回读，结束时清理临时状态和进程。
