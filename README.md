# 练习记录多源误差下学习投入与成效关系的校正

面向社区与老年学习项目的 Python 分析服务。当前支持 `repeatability` 的已观察资料核对；业务资料在 `docs/data-contract.md`，现有调用与保存语义在 `docs/http-contract.md`，分析目标的验证边界在 `docs/verification.md`。

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
