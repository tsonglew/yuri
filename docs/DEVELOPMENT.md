# Yuri 0.2 开发设计与交付计划

更新：2026-09-28。运行环境：WSL Ubuntu，`/home/tsonglew/yuri`。

## 1. 目标与状态

将旧版星际争霸 II 机器人升级为宏观决策实验平台：规则建立基线，Laya 提供语义决策，后续接入行为克隆、PPO 等方案。官网目标为 `https://tsonglew.github.io/yuri/`，包含可交互沙盘和真实记录回放。

不宣称 Laya 已学会星际争霸，不沿用旧版 95% 胜率，不把规则输出冒充模型推理；动作置信度也不是获胜概率。

| 交付 | 实现位置 | 当前边界 |
| --- | --- | --- |
| 局面 / 决策协议 | contracts.py | 版本、数值、动作合法性校验 |
| 规则策略 | policies.py | 无模型依赖，可离线运行 |
| Laya 策略 | policies.py | 真实 Router API；权重需通过 preflight 输出提交号并固定 |
| 异步调度 | runner.py | 单在途任务，晚到或紧急状态回退 |
| 游戏适配 | game.py | 实验性神族运营与执行，未实战验证 |
| 官网与回放 | site/ | 规则实时计算，模型结果导入回放 |
| 依赖升级 | pyproject.toml / uv.lock | Python 3.12，新入口不依赖旧 TF |
| 发布 | .github/workflows/pages.yml | 仓库需开启 Pages / GitHub Actions |

## 2. 旧代码诊断与迁移原则

旧链路为 `main.py → GameLauncher → MainBot → AttackChoiceBot → AttackBot`。规则负责运营，CNN 对绘制的 176×200 图像分类，使用胜局动作作为标签。训练代码属于监督分类，不是完整 DQN。

问题包括：Python 3.6 / TF GPU 1.11 / Keras 2.2.4 过时；macOS 数据软链接失效；None 动作混入训练集；四分类与五项攻击函数不一致；Full 分支通道和返回值不一致；固定图像尺寸；缺乏自动测试。

迁移采用并行新入口：`src/yuri_next` 是受支持运行时，根目录旧 main、basebots、models、trainers、Pipfile* 保留作历史参考，不能视作已兼容 Python 3.12。新安装只使用 uv，不运行 pipenv。新运行时不读取旧模型和软链接。若后续复用旧 npy，单独编写受信任数据转换工具，不默认对任意文件开启 pickle。

## 3. 架构与职责

```mermaid
flowchart LR
  SC2[burnysc2 观测] --> O[Observation v1]
  O --> R[规则策略]
  O --> L[Laya Router]
  O -.扩展.-> F[行为克隆 / PPO]
  L --> G[阈值 / 延迟 / 动作检查]
  G -->|回退| R
  G --> D[Decision v1]
  R --> D
  F -.-> D
  D --> E[游戏执行器]
  D --> J[JSON / JSONL 记录]
  J --> W[官网回放]
  S[浏览器参数] --> B[JS 规则基线]
  B --> W
```

策略只选择宏观动作，执行器负责单位命令，运营脚本负责工人、人口、科技、生产、扩张。策略不得直接操作客户端，前端不包含模型密钥或隐式本机服务调用。

```text
src/yuri_next/
  contracts.py    Observation / Decision / Policy / Action
  policies.py     RulePolicy / LayaPolicy / 注册表
  runner.py       单在途后台推理
  game.py         SC2 观测、运营、执行与轨迹
  cli.py          decide / game 命令
examples/        离线局面
tests/           契约、回退、规则一致性测试
site/            官网、交互沙盘、回放、在线文档
scripts/         站点构建与校验
docs/            开发设计与验收计划
```

## 4. 数据协议

### Observation v1

| 字段 | 范围 | 语义 |
| --- | --- | --- |
| schema_version | 整数 1 | 不兼容修改升版本 |
| army_supply | 非负数 | 可调动己方战斗部队人口 |
| enemy_army_supply | 非负数 | 120 秒记忆中的最后可见敌战斗人口估计；不是实时兵力，也不保证存活 |
| base_threat | 0–1 | 基地附近可见敌战斗人口 / 12，截断为 1 |
| army_health | 0–1 | 部队生命与护盾总和 / 对应上限；空军队为 1 |
| intel_age_seconds | 非负数 | 距看到敌战斗部队的游戏秒数 |
| minerals / gas | 非负数 | 可用资源 |
| supply_left | 非负数 | 剩余人口 |
| game_time_seconds | 非负数 | 游戏时间，与墙钟推理延迟区分 |
| enemy_visible | 布尔 | 当前是否看见敌战斗部队 |

拒绝 NaN、无穷、负数、数字字符串。无视野不能视为敌方没有军队。游戏适配器按敌军 tag 保留最后可见兵力与时间戳，记忆超过 120 秒后丢弃；迷雾下无法确认单位死亡，所以保留值只是最后可见估计，可能过时或已阵亡。

### Action v1

| ID | 语义 | 初版执行 |
| --- | --- | --- |
| hold | 集结 | 基地朝地图中心方向附近 attack-move |
| defend | 防守 | 基地附近敌军，否则返回基地 |
| attack | 进攻 | 已知敌建筑，否则敌方起点 |
| retreat | 撤退 | move 回己方基地 |
| scout | 侦察 | 一名战斗单位前往敌方起点 |

无战斗部队时只允许 hold。初版管理追猎者、虚空辉光舰、狂热者，不实现复杂微操。hold 指集结，非游戏 hold-position 技能。

### Decision v1 与轨迹

字段：`action`、`requested_policy`、`executed_policy`、`reason`、`latency_ms`、可空 `confidence`、`fallback_reason`、`model`、`model_revision`、`proposed_action`、`schema_version`。`model_revision` 保存 Laya 实际加载的 Hub commit；动作迟滞覆盖策略提案时，`action` 表示执行动作，`proposed_action` 保留原提案，其余情况为空。

规则不生成置信度。Laya 原因只描述来源，不虚构推理解释。请求 laya 但实际 rules 时必须显示回退。墙钟延迟包括首次模型加载。

记录外层：`schema_version`、`source`、`observation`、`decision`。异步结果额外记录 `inference_observation`，避免把执行时局面误当成推理输入。页面展示执行时局面。JSON 用于单条 / 数组，JSONL 用于逐决策轨迹。每局另存 `.meta.json` 记录策略、地图、难度、种子、SC2 build 和结果；`yuri evaluate` 复用配对种子，汇总胜率 Wilson 区间、模型实际执行次数、回退率和 Laya 热推理 p50 / p95。

## 5. Laya 实现规范

版本固定 `laya==0.3.21`，已核对发布 wheel 中 Router.predict 和置信度代码。采用英文结构化状态与 `model="english"`，避免自动路由变化。

1. 排序序列化 Observation，候选仅包含合法动作。
2. 构造 `type="choice"` 的 action 问题，说明敌军人口只是最后可见估计，不能当作当前兵力。
3. 一次延迟加载 Router 并复用；不在每 tick 加载权重。
4. 调用 `Router.predict(state, questions, model=..., min_confidence=...)`。
5. 读取 `answers.action.choice` 和 `answer_confidence`。
6. 缺字段、非法动作、无效数值、低置信度、异常、晚到结果都回退规则，保留原因。

默认阈值 0.65，结果预算 1500 ms，均为待校准工程初值。模型置信度不是策略正确率或获胜概率。

实战使用单工作线程，至多一个在途任务。模型加载或推理期间，游戏继续执行规则。已有任务未完成时不追加无限队列。紧急防守 / 撤退覆盖旧结果。预算检查拒绝晚到结果，但不能中断原生 GPU 调用；线程关闭也不保证进程立即退出。生产化应使用独立进程、硬超时、预热、熔断与重启。

首次权重下载可能触发预算回退。离线验收可调大预算。包版本已锁定，但权重 revision 尚未锁定；真实验证时记录模型 revision / 校验值、设备、冷 / 热延迟，再固定权重。不能把替身测试当成真实模型验收。

## 6. 依赖升级矩阵

| 原组件 | 新方案 | 理由 |
| --- | --- | --- |
| Python 3.6 | 3.12.x | 当前 WSL 已有，先收窄支持矩阵 |
| Pipenv | uv + pyproject + uv.lock | extras 与可复现安装 |
| sc2 0.9.0 | burnysc2 7.x，核查 7.3.0 | 现代 API 与 WSL 支持 |
| TF GPU 1.11 / Keras 2.2.4 | 新入口移除 | Laya 按需使用 PyTorch |
| NumPy 1.15 | 按新游戏 / 模型依赖解析 | 核心不引入数组框架 |
| OpenCV 3.4 | 新入口移除 | 结构化状态与 SVG 展示 |
| PyYAML | 核心不需要 | JSON / CLI 配置 |
| 无前端 | 原生 HTML / CSS / ES Modules | 静态站点，无 npm 构建依赖 |

安装以 uv.lock 为准；核心无运行时第三方依赖，game / laya 分开启用。当前锁定的 Laya 依赖解析到 PyTorch 2.14.0 和 CUDA 13 运行库，模型环境下载较大；实机预检仍须确认 `torch.cuda.is_available()` 和设备名称，依赖安装成功本身不能证明 CUDA 可用。

## 7. WSL 操作手册

以下均在 Ubuntu 的项目目录执行：

```bash
cd /home/tsonglew/yuri
uv sync --locked
uv run --locked pytest -q
uv run --locked yuri decide --policy rules \
  --state examples/observation.json --output artifacts/rules.json

# 模型依赖与真实推理；首次下载权重
uv sync --locked --extra laya
# 首次预热真实权重；成功条件是 status=passed 且 executed_policy=laya
uv run --locked --extra laya yuri preflight \
  --state examples/observation.json --budget-ms 120000 \
  --output artifacts/laya-preflight.json

# 从预检记录 decision.model_revision 复制实际 Hub 提交号，固定后重复预检
read -r -p '预检记录中的模型提交号: ' YURI_LAYA_REVISION
export YURI_LAYA_REVISION
uv run --locked --extra laya yuri preflight \
  --state examples/observation.json --budget-ms 120000 \
  --output artifacts/laya-preflight-pinned.json

# 常规离线决策仍使用置信度阈值；检查是否透明地回退
uv run --locked --extra laya yuri decide --policy laya \
  --state examples/observation.json --budget-ms 120000 \
  --output artifacts/laya.json

# 游戏与 Laya 对局
uv run --locked --extra game yuri doctor --map AcropolisLE
uv run --locked --extra game yuri game --policy rules --map AcropolisLE --seed 1
uv run --locked --extra game --extra laya yuri game --policy laya --map AcropolisLE --seed 1

# 同一组随机种子配对运行；收集胜负、推理回退率和模型实际执行次数
uv run --locked --extra game --extra laya yuri evaluate \
  --policies rules laya --games-per-policy 10 --seed-start 1 --map AcropolisLE
```

检查 executed_policy。即使 CLI 正常退出，也可能执行了规则回退。使用 `uv run` 时保留需要的 extra，避免同步环境时移除可选依赖。

Windows 调用示例：

```powershell
wsl.exe -d Ubuntu -- bash -lc 'cd /home/tsonglew/yuri && uv run --locked yuri decide --policy rules --state examples/observation.json'
```

不要使用 Windows Python 安装 WSL 依赖。新包安装不再需要从父目录执行 `python -m yuri.main`。

### 游戏环境

优先 Windows SC2 + WSL Python。burnysc2 默认检测 WSL 并使用 Windows 游戏。当前环境探测到安装目录为 `/mnt/c/Program Files (x86)/Battle.net/StarCraft II`，它与库默认的安装目录不同，因此启动前设置 `SC2PATH`。WSL2 还需设置 `SC2CLIENTHOST` 为 Windows 可达 IPv4 地址，`SC2SERVERHOST=0.0.0.0`；用 `powershell.exe -NoProfile -Command Get-NetIPConfiguration` 查看地址，不要把会变化的 IP 写入仓库。示例：

```bash
export SC2PATH='/mnt/c/Program Files (x86)/Battle.net/StarCraft II'
read -r -p 'Windows 可达的 IPv4 地址: ' SC2CLIENTHOST
export SC2CLIENTHOST SC2SERVERHOST=0.0.0.0
```

游戏目录当前未找到 `.SC2Map` 地图。需要先按 Blizzard 官方 [地图包说明](https://github.com/Blizzard/s2client-proto#map-packs)下载并解压获准使用的地图包到 `$SC2PATH/Maps`；上述命令使用 2019 S3 包中的 `AcropolisLE`。库按文件名定位地图，即使地图位于赛季子目录中也只传文件名。地图包由 Blizzard 单独提供并受其 AI and Machine Learning License 约束。

也可安装 Linux headless SC2，设置 `SC2_WSL_DETECT=0`，适合批量评估。两种方式均需地图和兼容游戏版本。`SC2PATH`、连接地址、可执行文件和地图诊断可用以下命令逐项确认：

```bash
find "$SC2PATH/Versions" -maxdepth 2 -name SC2_x64.exe -type f -print
find "$SC2PATH/Maps" -maxdepth 3 -iname '*.SC2Map' -print | head
```

`yuri doctor --map AcropolisLE` 可在不启动游戏的情况下检查可执行文件、地图和 WSL 连接变量。WSL 网络和安装位置已探测；完整游戏仍需本地地图文件并通过实机连接验收。

## 8. 官网和交互验收

目标 `https://tsonglew.github.io/yuri/`，所有资源用相对 URL，适配子路径，不添加 CNAME。

- 五个场景：优势、基地威胁、低血撤退、陈旧情报、开局。
- 五个滑块：双方兵力、健康、威胁、情报年龄；实时规则计算。
- SVG 地图显示基地、单位示意与动作方向，明确非游戏画面。
- 展示来源、理由、命中优先级和实测耗时，不虚构置信度。
- 导出 JSON，导入 JSON / 数组 / JSONL；逐条时间线回放。
- 回放冻结输入，避免修改状态后继续显示旧模型结果。
- 最多 10 MB / 10000 条记录，校验协议 / 数值 / 动作。
- 文件仅在浏览器解析，不上传；格式合法不证明记录真实性。
- 官网不运行模型，不自动连接访客本机端口。

```bash
python3 scripts/build_site.py
python3 scripts/check_site.py
python3 -m http.server 8765 --directory _site --bind 127.0.0.1
```

后续在线推理需要独立 HTTPS 服务：鉴权、CORS、限流、队列、预算和明确的在线状态。静态 GitHub Pages 本身不能承载 Python 推理。

## 9. GitHub Pages 发布和回滚

1. 功能分支运行 CI 后合入默认分支 master；工作流同时兼容 main。
2. Settings → Pages → Source 选择 GitHub Actions。
3. 默认分支 push 或 workflow_dispatch 运行 Pages。
4. 构建校验静态资源，只上传 `_site`；deploy 使用 pages:write、id-token:write 与 github-pages 环境。
5. 最终检查部署 URL、首页、文档、模块 MIME、子路径与导入导出，再声明上线成功。

不改个人根站仓库。项目站点自动位于 /yuri/；若根站已有自定义域名，最终地址以 Pages 输出为准。产物不包含仓库源码、模型、日志、虚拟环境和软链接。

站点回滚：回退错误提交并重新发布；策略回滚：`--policy rules`；依赖回滚：恢复旧 pyproject / uv.lock 后 locked sync。无需强推历史。

## 10. 可扩展策略

实现 `Policy.decide(Observation) -> Decision`，使用 `register_policy(name, factory)` 注册。当前 CLI 显式列出 rules / laya，新增策略需同步更新命令选项、extra、失败行为和测试，不自动加载不受控插件。

状态：

- P2 代码已加入 120 秒敌军 tag 记忆、多基地加权防守、6 秒动作迟滞（紧急防守 / 撤退立即覆盖）、工人 / 气矿目标和双星门上限。
- P2 配对评估命令和报告已实现，实测尚未完成；地图包缺失，尚无可比较的完整对局和轨迹。

后续推荐顺序：

1. 规则 v2：继续补兵种克制和可评估的目标选择规则。
2. 行为克隆：专家 / 筛选规则轨迹训练小 MLP；按完整对局划分数据集防止相邻帧泄漏。
3. Laya 微调：领域标签、阈值校准、未见地图与对手评估。
4. PPO：封装 reset / step / episode、奖励、动作 mask、固定决策周期；游戏运行成本单独评估。
5. 混合策略：低频战略模型与高频规则 / 微操，共用记录协议。

新增策略需明确 ID、配置、依赖、合法动作、延迟预算、回退、模型许可证与评估结果。

## 11. 测试与评估

离线检查：观察值边界；规则优先级；Laya 正常 / 低置信度 / 缺字段 / 非法动作 / 加载失败 / 晚到；调度单在途与旧结果覆盖；Python / JS 共用规则场景；五场景与滑块、重置、导入导出、坏文件；桌面 / 手机 / 键盘 / 控制台；站点子路径与构建产物白名单。

实机验收（尚未完成）：

1. 规则机器人完成一局并导出轨迹。
2. 真实 Laya 权重输出 executed_policy=laya，记录 revision / 设备。
3. 人为制造延迟或低置信度，游戏继续且回退可追溯。
4. 固定游戏版本、地图、对手、种族、难度、随机种子，配对运行。
5. 每策略先 10 局排错，再至少 100 局初步评估；报告 Wilson 胜率区间。
6. 报告崩溃率、对局时长、热推理 p50/p95、回退率、非法动作率与内存 / 显存。

工程门槛：100 局无未处理崩溃，非法执行动作 0，所有回退可追溯。胜率门槛在基线测出后设定；若 Laya 无收益，保留实验策略而非默认策略。

## 12. 分阶段里程碑

| 阶段 | 工作 | 验收 |
| --- | --- | --- |
| P0 本次基础 | 新包、规则、Laya 适配、实验游戏适配、文档、官网、CI | 检查通过，未验证项清楚 |
| P1 实机 | SC2 / 地图 / WSL、模型预热、权重固定 | 两种策略完整对局 |
| P2 质量 | 敌军记忆、多基地防守、迟滞、运营优化 | 配对评估报告和完整轨迹 |
| P3 服务化 | 独立进程、硬超时、熔断，可选 API | 卡死恢复、队列上限与负载延迟达标 |
| P4 学习 | 数据集、行为克隆 / PPO、评估工具 | 新策略不改执行器，结果可复现 |

## 13. 已知限制与来源

宏观运营未经调优，多基地 / 地形处理有限；后台线程不是硬超时；权重版本尚未固定；静态站点只做规则计算和回放；旧代码依然只作历史参考。当前未宣称真实模型、整局对战通过。

本次验证记录：先前 26 项 Python 测试、12 项前端测试通过；规则 CLI 导出、burnysc2 导入和观测器 smoke test 通过。浏览器已验证五场景、滑块、重置、JSON 导入 / 导出、JSONL 时间线、坏文件拒绝和手机布局。本轮新增 `yuri preflight`，报告 PyTorch/CUDA/设备与真实模型 revision；首次 Laya 同步需要下载数 GB 的 PyTorch/CUDA 包，本轮因下载耗时停止，虚拟环境内尚未安装 `laya` / `torch`，重新执行 `uv sync --locked --extra laya` 可复用已完成包缓存，权重推理尚未验证。`yuri doctor` 已确认 WSL2、Windows 游戏可执行文件存在且连接变量已设置，但地图目录不存在，因此整局无法启动；变量存在性不代表网络握手已通过。未在本轮运行测试套件。

官网已于 2026-09-28 发布至 [tsonglew.github.io/yuri](https://tsonglew.github.io/yuri/)，首版 [Pages 发布成功](https://github.com/tsonglew/yuri/actions/runs/36417064034)，[CI 通过](https://github.com/tsonglew/yuri/actions/runs/36417064112)。GitHub Pages 已选择 Actions，后续默认分支推送自动发布。

参考：

- [Laya PyPI](https://pypi.org/project/laya/) 与 [官方源码](https://github.com/NandhaKishorM/laya)
- [burnysc2 / WSL](https://github.com/BurnySc2/python-sc2#readme)
- [GitHub Pages 项目站点](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)
- [GitHub Pages 工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)

外部信息以本次核查为准，性能数字必须来自本项目实际测量。
