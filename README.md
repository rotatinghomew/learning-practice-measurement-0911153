# 练习记录多源误差下学习投入与成效关系的校正

面向社区与老年学习项目的 Python 分析服务。支持 `repeatability` 的已观察资料核对与 `dose_response` 的投入—成绩关联分析；业务资料在 `docs/data-contract.md`，现有调用与保存语义在 `docs/http-contract.md`，分析目标的验证边界在 `docs/verification.md`，统计识别与假设见 `docs/dose-response.md`。

## 环境与运行

Python 3.11（独立产品镜像按 digest 固定为 3.11.14，执行器可使用 3.11.x），依赖完整固定在 requirements.txt。首次准备环境安装依赖；正式运行与测试无需联网。

```sh
python -m pip install -r requirements.txt
python -m compileall -q app tests
python -m unittest discover -s tests -v
python -m app.demo
```

单个 Linux 应用容器构建与当前产品展示：

```sh
docker build -t practice-measurement:initial .
docker run --rm --network none practice-measurement:initial python -m unittest discover -s tests -v
docker run --rm --network none practice-measurement:initial python -m app.demo
```

`app.demo` 运行真实 HTTP 资料目录与分析进程边界并回读结果，通常数秒结束。数据全部为本地合成记录，没有外部服务或凭据；运行时不会拉包。测试与演示默认读取 data/current.json，后续可用 `--packet` 指定同契约的业务资料。所有时间及单位含义见资料契约，不以零代替缺失。

现有业务视图：`app/measurements.py`，当前结果在 `source_profile` 中公开。

容器默认入口为真实 HTTP 分析服务（8322 端口），同时启动仅回环可见的资料目录；`GET /health` 可检查就绪。离线限时演示显式运行 `python -m app.demo`，不是让常驻服务自行退出。

## `dose_response` 分析

```sh
python -m app.demo --analysis dose_response
```

估计“以练习前成绩调整后，每多 60 分钟常规练习与后续测验成绩的关联”。它把自填、家属代填、设备计时都视为有误差的测量，不指定任何渠道为标准答案；核验资料按声明概率加权，只作为工具变量情景，不当作真值。默认输出保留“共同偏差 vs 真实差异”的未识别性：给出单侧识别界、共同偏差比例情景、家户内对照以及有资料支持的练习区间，不补出确定曲线。字段与选项见 `docs/data-contract.md`，识别假设与退化情形见 `docs/dose-response.md`；领域测试为 `python -m unittest tests.test_dose_response -v`。
