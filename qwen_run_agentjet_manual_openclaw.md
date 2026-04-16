# Qwen3-1.7B 在 AgentJet/OpenClaw 中跑通人工对话训练操作手册

## 1. 这份文档和 `qwen_run_agentjet.md` 的区别

这份文档对应的是“人工对话训练”版本：

- 不使用 `mock_user_request.py`
- 不走自动化并发压测脚本
- 由你自己在终端里通过 OpenClaw 发消息
- 每次真实对话请求都会进入 `fake_vllm_endpoint.py`
- 然后被展开成多个候选回答，进入 swarm 训练

对应的自动化脚本版文档是：

- `/gfs/space/private/fengzl/AgentJet/summary/qwen_run_agentjet.md`

这份文档更适合：

- 你想一边和 OpenClaw 对话，一边看训练有没有推进
- 你想自己控制提问内容，而不是用现成题库自动灌流量
- 你想先验证“人工对话也能作为训练样本进入 swarm”

## 2. 这条人工训练链路实际是什么

人工训练版的真实调用链路是：

```text
你在终端里执行 openclaw agent --agent <名字> --message <消息>
-> OpenClaw 收到请求
-> OpenClaw 把模型请求转给 training-proxy
-> training-proxy 实际就是 http://127.0.0.1:8090/v1
-> fake_vllm_endpoint.py 接住这次请求
-> 同一个请求被扩成 4 个 candidate rollout
-> reward 脚本给 4 个候选打分
-> swarm server 收到这些 rollout 并训练
-> 分数最高的一条回答返回给你
```

所以这里不是“你先和一个普通聊天机器人对话，之后再离线训练”。

更准确地说是：

```text
你发出一次真实对话请求
-> 后台立即复制成多个候选回答
-> 这次对话本身就直接参与在线训练
```

## 3. 关键路径

```bash
/gfs/space/private/fengzl/Models/Qwen/Qwen3-1.7B  # 基础模型目录
/gfs/space/private/fengzl/AgentJet/configs/fengzl/qwen3_17b_openclaw_swarm.yaml  # swarm 配置文件
/gfs/space/private/fengzl/AgentJet/tutorial/opencode_build_openclaw_agent/fake_vllm_endpoint.py  # fake vLLM 训练代理
/gfs/space/private/fengzl/AgentJet/tutorial/opencode_build_openclaw_agent/on_compute_relative_reward.py  # 相对奖励脚本
/gfs/space/private/fengzl/OpenClaw/scripts/openclaw_env.sh  # OpenClaw 环境脚本
/gfs/space/private/fengzl/OpenClaw/scripts/onboard_training_proxy.sh  # 把 OpenClaw provider 接到本地 8090 的脚本
```

## 4. 建议开几个终端

建议开 `4` 个终端，最不容易混乱。

### 终端 A：启动 `ajet-swarm start`:
  <img width="909" height="525" alt="image" src="https://github.com/user-attachments/assets/cce0c473-4a9c-4d52-a3bd-8e35541f2919" />
  
- 会拉起：

- AgentJet swarm server
- verl 训练进程
- interchange server
- 真正的 vLLMHttpServer

### 终端 B：启动 `fake_vllm_endpoint.py`：
  <img width="917" height="530" alt="image" src="https://github.com/user-attachments/assets/23bb8e02-9e65-4d49-8da3-8137aa3e284b" />
  
-监听：http://127.0.0.1:8090/v1
- OpenClaw 实际先打到它，不是直接打到真 vLLM。

- 接住 OpenClaw 的 /v1/chat/completions
- begin_episode() 向 swarm server 申请多个 episode
- 拿到 server 分配的真实 base_url/api_key
- 再把同一个请求转发给真正的模型端
- 收集多个 candidate
- 打 reward
- end_episode() 回传训练
- 只把最好的一条返回给 OpenClaw
### 终端 C：启动 OpenClaw 和相关配置文件，open claw后台：
  <img width="899" height="598" alt="image" src="https://github.com/user-attachments/assets/adb251db-2719-4ce3-99f9-1435b4329097" />

### 终端 D：openclaw看训练状态（进入虚拟环境执行：ajet-swarm overwatch）、TensorBoard、导出曲线：
<img width="782" height="731" alt="image" src="https://github.com/user-attachments/assets/b4a13e2f-182f-4f96-8066-b80f848f0270" />

- AgentJet Swarm Overwatch 日志解读

| 字段 | 当前值 | 含义 | 这张图说明了什么 |
|---|---:|---|---|
| Server | `http://localhost:10086` | Overwatch 当前连接的 Swarm Server 地址 | 监控连的是本机 `10086` 端口的 AgentJet Swarm 服务 |
| Current Time | `2026-04-16 10:33:24` | 当前界面时间 | 这是这次截图对应的时间点 |
| Last Update | `10:33:24` | 面板最近一次刷新时间 | 说明数据是刚刷新的，不是旧缓存 |
| Refresh | `2.0s` | 面板刷新间隔 | 每 2 秒更新一次状态 |
| Requests | `41` | 当前累计请求数 | 说明系统已经处理了一批请求，不是刚启动 |
| Engine Status | `ENGINE.ROLLING` | 引擎当前状态 | 表示系统正在进行 rollout / 样本采集，不是空闲或报错 |
| Global Step (Model's Weight Version) | `4` | 当前模型权重版本号 | 说明已经发生过多次参数更新，不只是单纯采样 |

- 2. Completed Episode Pool Summary

| 指标 | Current | Target | Progress | 含义 | 这张图说明了什么 |
|---|---:|---:|---:|---|---|
| Completed Episodes | `8` | `16` | `50.0%` | 已完成 episode 数 / 下一次更新所需 episode 数 | 为下一次权重更新已经攒到一半样本 |
| Completed Tasks (chosen) | `2` | `4` | `50.0%` | 已完成的被选中 task 数 / 目标 task 数 | 当前已完成 2 个有效 task |
| Completed Non-Dummy Tasks | `2` | `4` | `50.0%` | 已完成的真实任务数，不含 dummy task | 目前完成的都是真实任务 |
| Average Episode Per Task | `4.00` | `4` | `-` | 每个 task 平均对应多少个 episode | 当前正好每个 task 平均 4 个 episode，结构很整齐 |

- 3. Running Episodes

| 字段 | 当前值 | 含义 | 这张图说明了什么 |
|---|---:|---|---|
| Running Episodes | `4` | 当前并发运行中的 episode 数量 | 当前有 4 条 episode 正在跑 |
| Episode UUID | 多个 UUID | 每条 episode 的唯一标识 | 用来区分不同 rollout |
| Status | `claimed` | 该 episode 已被 worker/client 领取并在执行 | 4 条 episode 都已经被分配出去在跑 |
| LLM Calls | `1` | 当前 episode 已发生的 LLM 调用次数 | 每条 episode 目前只调用了 1 次 LLM，说明都还比较早期或任务较简单 |
| Last Req / Patience | `6.6s / 240.0s` | 上次请求距现在多久 / 超时耐心阈值 | 这些 episode 都很活跃，没有卡死或超时 |

### 终端 E：和openclaw对话：
  <img width="727" height="290" alt="image" src="https://github.com/user-attachments/assets/edfd6b23-4f75-47de-8f94-b6cf703e8cc0" />


终端分工理解成这样最清楚：

- A 是 server
- B 是 client 代理
- C 是openclaw启动
- D 是监控面板
- E 是对话

## 5. 先准备环境

### 5.1 终端 A / B / D

这 3 个终端都需要激活 AgentJet 环境。

```bash
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活 AgentJet 虚拟环境
```

### 5.2 终端 C

这个终端要执行 `openclaw` 命令，所以需要加载 OpenClaw 环境。  
如果你还要在终端 C 里顺手执行 Python 脚本，也可以再额外激活 AgentJet 环境。

```bash
source /gfs/space/private/fengzl/OpenClaw/scripts/openclaw_env.sh  # 加载 OpenClaw 环境变量和 CLI wrapper
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 可选，只有当你想在这个终端顺手跑 Python/检查脚本时才需要
```

### 5.3 确认模型和配置存在

```bash
ls /gfs/space/private/fengzl/Models/Qwen/Qwen3-1.7B  # 确认基础模型目录存在
ls /gfs/space/private/fengzl/AgentJet/configs/fengzl/qwen3_17b_openclaw_swarm.yaml  # 确认训练配置存在
```

## 6. 第一步：启动 swarm server

这一步放在终端 A。

```bash
cd /gfs/space/private/fengzl/AgentJet  # 进入 AgentJet 根目录
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活 AgentJet 环境
ajet-swarm start --swarm-port=10086  # 启动 swarm server，监听 10086 端口
```

如果你想后台运行：

```bash
cd /gfs/space/private/fengzl/AgentJet  # 进入 AgentJet 根目录
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活 AgentJet 环境
nohup ajet-swarm start --swarm-port=10086 > /gfs/space/private/fengzl/AgentJet/swarm_server.log 2>&1 &  # 后台启动 swarm server 并保存日志
```

## 7. 第二步：启动 fake vLLM 训练代理

这一步放在终端 B。  
它本质上是“OpenClaw 到 swarm server 中间的 client 代理层”。

它负责：

- 接收 OpenClaw 发来的 `/v1/chat/completions`
- 把一次请求扩成 `4` 个 candidate rollout
- 调用 `begin_episode()` / `end_episode()`
- 计算相对奖励
- 把最好的一条结果返回给 OpenClaw

```bash
cd /gfs/space/private/fengzl/AgentJet/tutorial/opencode_build_openclaw_agent  # 进入教程目录
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活 AgentJet 环境
export AJET_SWARM_URL=http://127.0.0.1:10086  # 指向 swarm server
export MODEL_PATH=/gfs/space/private/fengzl/Models/Qwen/Qwen3-1.7B  # 指定基础模型路径
export NUM_REPEAT=4  # 一个用户请求扩成 4 个候选回答
export REWARD_MODE=heuristic  # 本地 smoke/manual 测试默认使用启发式奖励，不要求 DASHSCOPE_API_KEY
python -u fake_vllm_endpoint.py  # 启动 fake vLLM 代理，默认监听 8090
```

如果你想后台运行：

```bash
cd /gfs/space/private/fengzl/AgentJet/tutorial/opencode_build_openclaw_agent  # 进入 fake vLLM 目录
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活 AgentJet 环境
export AJET_SWARM_URL=http://127.0.0.1:10086  # 指向 swarm server
export MODEL_PATH=/gfs/space/private/fengzl/Models/Qwen/Qwen3-1.7B  # 指向基础模型目录
export NUM_REPEAT=4  # 设置一请求多候选
export REWARD_MODE=heuristic  # 本地 smoke/manual 测试默认使用启发式奖励，不要求 DASHSCOPE_API_KEY
nohup python -u fake_vllm_endpoint.py > /gfs/space/private/fengzl/AgentJet/fake_vllm_endpoint.log 2>&1 &  # 后台启动 fake vLLM 并保存日志
```

说明：

- `on_compute_relative_reward.py` 默认 `REWARD_MODE=pointwise`
- `pointwise` / `listwise` 模式会要求 `DASHSCOPE_API_KEY`
- 当前这份人工对话训练手册默认是本地 smoke/manual 跑通，因此推荐显式设置 `REWARD_MODE=heuristic`
- 如果你后面想切回真正的 LLM judge，再去设置真实 `DASHSCOPE_API_KEY`

## 8. 第三步：把 OpenClaw 的模型地址指向本地拟态接口

这一步是人工对话训练版里最容易忽略的一步。  
如果 OpenClaw 没有指向 `http://127.0.0.1:8090/v1`，你后面的人工对话就不会进入 swarm 训练链路。

这里有两种做法：UI 配置法和脚本法。  
但要先说结论：

- 在当前这台机器上，默认推荐你用“脚本法”
- 因为 OpenClaw 的网页配置页不是天然就能直接打开的
- 如果想走 UI 配置法，你得先把 OpenClaw gateway / dashboard 跑起来
- 如果 gateway 没启动，当然就进不去网页
- 所以对当前环境来说，`onboard_training_proxy.sh` 才是主路径，UI 配置法只是可选补充

### 8.1 做法 A：在 OpenClaw 配置页面手工设置

这条路径只有在 OpenClaw 的 gateway / dashboard 已经启动时才成立。  
如果你现在还没有启动 OpenClaw 网页端，不要先走这条，直接跳到 `8.2`。

如果你确实想通过 OpenClaw 界面配置，先确保 gateway/UI 已经能打开，然后再按下面这条路径：

```text
设置 > 配置 > Models > Model Providers > vllm:http://localhost:8090/v1
```

要点是：

- provider/base URL 指向 `http://127.0.0.1:8090/v1` 或 `http://localhost:8090/v1`
- model id 使用 `training-proxy`
- provider id 使用 `training-proxy`
- compatibility 选择 OpenAI / openai-compatible

如果你后面真要补跑 UI，一般至少要先把 OpenClaw gateway 跑起来。  
在当前容器环境里，更稳妥的理解是：

```text
先有 gateway / dashboard
-> 才谈得上进入网页配置页
```

### 8.2 做法 B：用现成脚本一键接入

这个方式更适合当前这台机器，能避免手工点页面。  
这也是本文档默认推荐的方法。

```bash
source /gfs/space/private/fengzl/OpenClaw/scripts/openclaw_env.sh  # 加载 OpenClaw 环境
bash /gfs/space/private/fengzl/OpenClaw/scripts/onboard_training_proxy.sh  # 把 OpenClaw 的 provider 一键接到 127.0.0.1:8090/v1
```

这个脚本本质上会做这些事情：

- custom base url 设成 `http://127.0.0.1:8090/v1`
- custom model id 设成 `training-proxy`
- custom provider id 设成 `training-proxy`
- compatibility 设成 `openai`

### 8.3 配完以后做一次本地检查

```bash
grep -n "training-proxy" /gfs/space/private/fengzl/OpenClaw/state/openclaw.json  # 检查 OpenClaw 是否已经把 provider/model 指向 training-proxy
grep -n "127.0.0.1:8090" /gfs/space/private/fengzl/OpenClaw/state/openclaw.json  # 检查 baseUrl 是否已经指向本地 fake vLLM
```

### 8.4 启动 OpenClaw gateway

这一步在原版文档里容易漏掉，但在当前容器环境里是必须单独做的。  
原因是：

- `onboard_training_proxy.sh` 只负责改配置
- 它不会自动把 OpenClaw gateway 服务拉起来
- 如果 `18789` 没有监听，后面的 `openclaw agent --message ...` 就容易先报 gateway 连接失败

在当前环境里，最稳妥的做法是手动前台启动 gateway。

```bash
source /gfs/space/private/fengzl/OpenClaw/scripts/openclaw_env.sh  # 加载 OpenClaw 环境
"${OPENCLAW_REAL_BIN}" gateway run --bind loopback --port 18789 --token openclaw-local-training-token  # 前台启动 OpenClaw gateway，监听本地 18789
```

如果你想后台运行，也可以这样：

```bash
source /gfs/space/private/fengzl/OpenClaw/scripts/openclaw_env.sh  # 加载 OpenClaw 环境
nohup "${OPENCLAW_REAL_BIN}" gateway run --bind loopback --port 18789 --token openclaw-local-training-token > /gfs/space/private/fengzl/OpenClaw/logs/gateway.manual.log 2>&1 &  # 后台启动 gateway 并保存日志
```

启动后建议立刻检查一下端口：

```bash
python - <<'PY'  # 检查 18789 端口是否已经监听
import socket
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("127.0.0.1", 18789))
    print("18789 open")
except Exception as e:
    print("18789 closed", e)
finally:
    s.close()
PY
```

只有当这一步完成后，后面的终端对话才是完整链路：

```text
OpenClaw CLI
-> OpenClaw gateway
-> training-proxy (8090)
-> fake_vllm_endpoint.py
-> swarm server
```

## 9. 第四步：用终端手动和 OpenClaw 对话

这一步开始，才是真正的“人工对话训练”。  
前提是你前面已经完成：

- 第 8.2 步：把 provider 指到 `127.0.0.1:8090/v1`
- 第 8.4 步：把 OpenClaw gateway 真正启动起来

终端使用：

- 这一步放在终端 C
- 这个终端不再执行 `mock_user_request.py`
- 你直接手动发消息给 OpenClaw

### 9.1 最简单的一次性发消息方式

```bash
source /gfs/space/private/fengzl/OpenClaw/scripts/openclaw_env.sh  # 加载 OpenClaw 环境
/gfs/space/private/fengzl/OpenClaw/cli/bin/openclaw agent --agent smoke_train_agent --message "你好，先介绍一下你自己"  # 发一条人工消息给 OpenClaw
```

你可以连续发多条：

```bash
source /gfs/space/private/fengzl/OpenClaw/scripts/openclaw_env.sh  # 加载 OpenClaw 环境
/gfs/space/private/fengzl/OpenClaw/cli/bin/openclaw agent --agent smoke_train_agent --message "请更外向、更有感染力地介绍巴黎"  # 第 1 条人工消息
/gfs/space/private/fengzl/OpenClaw/cli/bin/openclaw agent --agent smoke_train_agent --message "再用更热情一点的风格改写一版"  # 第 2 条人工消息
/gfs/space/private/fengzl/OpenClaw/cli/bin/openclaw agent --agent smoke_train_agent --message "把回答缩短到三句话"  # 第 3 条人工消息
```

理解方式：

- 你每执行一次 `openclaw agent --message ...`
- 后台就会生成一次真正的模型请求
- 这次请求会被 fake vLLM 扩成多个 candidate rollout
- 所以每一条人工消息都可以进入训练

### 9.2 如果你想显式创建自己的 agent

当前环境里有 wrapper，可以把随机 agent 名映射到稳定的真实 agent。  
所以你也可以先创建一个你自己的 agent 名，再反复使用它。

```bash
source /gfs/space/private/fengzl/OpenClaw/scripts/openclaw_env.sh  # 加载 OpenClaw 环境
openclaw agents add fengzl_train_manual  # 新建一个人工训练用 agent
openclaw agent --agent fengzl_train_manual --message "你好，介绍一下你自己"  # 第 1 条消息
openclaw agent --agent fengzl_train_manual --message "请更像一个外向、健谈的人"  # 第 2 条消息
openclaw agent --agent fengzl_train_manual --message "给我一个更有情绪张力的版本"  # 第 3 条消息
```

训练结束后如果你想清理掉这个 agent：

```bash
source /gfs/space/private/fengzl/OpenClaw/scripts/openclaw_env.sh  # 加载 OpenClaw 环境
openclaw agents delete fengzl_train_manual  # 删除这个人工训练用 agent，并清理映射
```

### 9.3 关于 `pairing required`

如果你看到类似：

```text
gateway connect failed: pairing required
Gateway agent failed; falling back to embedded
```

这不一定是致命报错。  
在当前环境里，OpenClaw 往往会 fallback 到 embedded 模式，只要最后还能正常返回回答，说明这条人工对话还是打通了。

## 10. 第五步：边对话边看训练有没有推进

这一步放在终端 D。

### 10.1 看 swarm engine 状态

```bash
curl -s http://127.0.0.1:10086/get_engine_status  # 查看当前 engine 状态和 global_step
```

常见状态：

- `ENGINE.ROLLING`：正在收样本
- `ENGINE.ROLLING_POST`：rollout 后处理
- `ENGINE.WEIGHT_SYNCING`：同步训练后的权重
- `ENGINE.OFFLINE`：当前没有活跃训练

### 10.2 看关键进程是否都活着

```bash
ps -eo pid,etimes,cmd | grep -E "ajet-swarm start|fake_vllm_endpoint.py|openclaw|tensorboard" | grep -v grep  # 检查 server、fake vLLM、OpenClaw、TensorBoard 进程
```

### 10.3 看总步数配置

```bash
grep -n "total_training_steps" /gfs/space/private/fengzl/AgentJet/configs/fengzl/qwen3_17b_openclaw_swarm.yaml  # 查看这次总共要训练多少 step
```

当前这套配置是：

```text
total_training_steps: 25
```

### 10.4 看当前 step 对应多少 rollout

当前这套配置里：

- `train_batch_size: 4`
- `num_repeat: 4`

所以名义上：

```text
1 个 step
= 1 个 batch
= 4 个 task
= 16 个 rollout episodes
```

但因为采样模式是 `rollout_until_finish_enough_tasks`，实际运行时经常会把已经在飞的第 `5`、第 `6` 个 task 一起收尾，所以实际每 step 可能略多于 `16` 条 rollout。

## 11. 第六步：启动 TensorBoard 看训练曲线

这一步放在终端 D，或者后台运行。

```bash
cd /gfs/space/private/fengzl/AgentJet  # 进入 AgentJet 根目录
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活 AgentJet 环境
tensorboard --logdir /gfs/space/private/fengzl/AgentJet/tensorboard_log/fengzl_agentjet/qwen3_17b_openclaw_swarm --host 0.0.0.0 --port 6006  # 启动 TensorBoard，查看训练曲线
```

浏览器打开：

```text
http://127.0.0.1:6006/
```

如果你不想占住终端：

```bash
cd /gfs/space/private/fengzl/AgentJet  # 进入 AgentJet 根目录
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活 AgentJet 环境
nohup tensorboard --logdir /gfs/space/private/fengzl/AgentJet/tensorboard_log/fengzl_agentjet/qwen3_17b_openclaw_swarm --host 0.0.0.0 --port 6006 > /gfs/space/private/fengzl/AgentJet/tensorboard_qwen3_17b_openclaw_swarm.log 2>&1 &  # 后台启动 TensorBoard
```

## 12. 第七步：导出训练曲线图片和 CSV

```bash
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活 AgentJet 环境
python - <<'PY'  # 从 TensorBoard event 文件导出关键指标
from tensorboard.backend.event_processing import event_accumulator  # 读取 TensorBoard event 文件
import matplotlib.pyplot as plt  # 画图
from pathlib import Path  # 处理路径
import csv  # 写 csv

log_path = Path("/gfs/space/private/fengzl/AgentJet/tensorboard_log/fengzl_agentjet/qwen3_17b_openclaw_swarm/events.out.tfevents.1776239151.dd25b60a6c67f61b05484f5e173e3adf-taskrole1-0.42749.0")  # 当前实验 event 文件
out_dir = Path("/gfs/space/private/fengzl/AgentJet/summary")  # 输出目录
out_png = out_dir / "qwen3_17b_openclaw_swarm_training_curves.png"  # 输出图片路径
out_csv = out_dir / "qwen3_17b_openclaw_swarm_training_curves.csv"  # 输出 csv 路径

ea = event_accumulator.EventAccumulator(str(log_path))  # 创建 event 读取器
ea.Reload()  # 加载 event 文件

selected = [  # 选择要导出的关键指标
    "critic/rewards/mean",
    "critic/success_rate",
    "critic/real_reward",
    "actor/entropy",
    "actor/kl_loss",
    "timing_s/step",
    "perf/throughput",
    "response_length/mean",
]

fig, axes = plt.subplots(2, 4, figsize=(20, 8))  # 创建 2x4 子图布局
axes = axes.ravel()  # 拉平坐标轴数组
for ax, tag in zip(axes, selected):  # 逐个指标画图
    vals = ea.Scalars(tag)  # 读取某个指标的所有点
    x = [v.step for v in vals]  # 横轴用 global_step
    y = [v.value for v in vals]  # 纵轴用指标值
    ax.plot(x, y, marker="o", linewidth=2)  # 画折线
    ax.set_title(tag)  # 设置标题
    ax.set_xlabel("global_step")  # 设置横轴名称
    ax.grid(True, alpha=0.3)  # 打开网格

fig.suptitle("Qwen3-1.7B OpenClaw Swarm Training Curves", fontsize=15)  # 设置总标题
fig.tight_layout(rect=(0, 0, 1, 0.95))  # 调整布局
fig.savefig(out_png, dpi=180, bbox_inches="tight")  # 保存图片

steps = sorted({v.step for tag in selected for v in ea.Scalars(tag)})  # 收集所有出现过的 step
with out_csv.open("w", newline="", encoding="utf-8") as f:  # 打开 csv 文件
    writer = csv.writer(f)  # 创建 csv 写入器
    writer.writerow(["global_step"] + selected)  # 写表头
    for step in steps:  # 逐个 step 写数据
        row = [step]  # 先写当前 step
        for tag in selected:  # 再补每个指标在这个 step 的值
            mapping = {v.step: v.value for v in ea.Scalars(tag)}  # 转换成 step -> value 映射
            row.append(mapping.get(step, ""))  # 没值就写空字符串
        writer.writerow(row)  # 写入一行

print(out_png)  # 打印图片路径
print(out_csv)  # 打印 csv 路径
print("last_step", max(steps) if steps else None)  # 打印导出的最新 step
PY
```

## 13. 最短人工复现流程

如果你只是想最快复现一次“人工发消息也能训练”，直接按这个顺序做。

### 13.1 终端 A

```bash
cd /gfs/space/private/fengzl/AgentJet  # 进入 AgentJet 根目录
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活环境
ajet-swarm start --swarm-port=10086  # 启动 swarm server
```

### 13.2 终端 B

```bash
cd /gfs/space/private/fengzl/AgentJet/tutorial/opencode_build_openclaw_agent  # 进入 fake vLLM 目录
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活环境
export AJET_SWARM_URL=http://127.0.0.1:10086  # 指向 swarm server
export MODEL_PATH=/gfs/space/private/fengzl/Models/Qwen/Qwen3-1.7B  # 指向基础模型
export NUM_REPEAT=4  # 设置每请求 4 个候选回答
python -u fake_vllm_endpoint.py  # 启动 fake vLLM 训练代理
```

### 13.3 终端 C

```bash
source /gfs/space/private/fengzl/OpenClaw/scripts/openclaw_env.sh  # 加载 OpenClaw 环境
bash /gfs/space/private/fengzl/OpenClaw/scripts/onboard_training_proxy.sh  # 把 OpenClaw provider 指到本地 8090
openclaw agents add fengzl_train_manual  # 新建人工训练 agent
openclaw agent --agent fengzl_train_manual --message "你好，介绍一下你自己"  # 发第 1 条人工训练消息
openclaw agent --agent fengzl_train_manual --message "请用更外向、更热情的风格再说一遍"  # 发第 2 条人工训练消息
openclaw agent --agent fengzl_train_manual --message "把回答压缩到三句话以内"  # 发第 3 条人工训练消息
```

### 13.4 终端 D

```bash
curl -s http://127.0.0.1:10086/get_engine_status  # 查看训练是否已经进入 ROLLING
```

```bash
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活 AgentJet 环境
tensorboard --logdir /gfs/space/private/fengzl/AgentJet/tensorboard_log/fengzl_agentjet/qwen3_17b_openclaw_swarm --host 0.0.0.0 --port 6006  # 打开 TensorBoard 看曲线
```

## 14. 训练结束后怎么停掉

```bash
pkill -f fake_vllm_endpoint.py  # 停掉 fake vLLM 训练代理
pkill -f "ajet-swarm start --swarm-port=10086"  # 停掉 swarm server
pkill -f "ajet-swarm overwatch"  # 停掉 overwatch 监控
pkill -f "tensorboard --logdir /gfs/space/private/fengzl/AgentJet/tensorboard_log/fengzl_agentjet/qwen3_17b_openclaw_swarm"  # 停掉 TensorBoard
```

如果你还想把人工训练 agent 也清理掉：

```bash
source /gfs/space/private/fengzl/OpenClaw/scripts/openclaw_env.sh  # 加载 OpenClaw 环境
openclaw agents delete fengzl_train_manual  # 删除人工训练 agent
```

## 15. 一句话理解这份文档

这份文档跑通的是：

```text
你自己在终端里和 OpenClaw 发消息
-> 每条人工消息都进入 fake_vllm_endpoint.py
-> 每条消息都变成 one-to-many rollout
-> 然后直接参加 swarm 在线训练
```

它和 `mock_user_request.py` 版的区别只是“请求由谁发出”：

- `mock_user_request.py` 版：脚本自动发请求
- 本文档版：你自己手动发请求
