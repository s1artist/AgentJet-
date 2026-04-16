# Qwen3-1.7B 在 AgentJet/OpenClaw 中跑通 Swarm 训练操作手册

## 1. 这份文档是做什么的

这份文档总结的是当前已经实际跑通过的一条训练链路：

- 基础模型：`/gfs/space/private/fengzl/Models/Qwen/Qwen3-1.7B`
- AgentJet 配置：`/gfs/space/private/fengzl/AgentJet/configs/fengzl/qwen3_17b_openclaw_swarm.yaml`
- OpenClaw 根目录：`/gfs/space/private/fengzl/OpenClaw`
- AgentJet 根目录：`/gfs/space/private/fengzl/AgentJet`
- 训练任务名：`qwen3_17b_openclaw_swarm`

这条链路不是离线训练，而是在线 Agent RL：

```text
mock_user_request.py 发送用户请求
-> OpenClaw 接收请求
-> training-proxy(fake_vllm_endpoint.py) 把 1 个请求扩成多个候选回答
-> reward 脚本给多个候选打相对分
-> AgentJet swarm server 收集 rollout 并训练
-> TensorBoard 记录训练曲线
```

## 2. 先知道几个关键文件

下面这些文件是这次训练里最关键的入口：

```bash
# Swarm 训练配置文件，控制模型路径、步数、batch 大小等
/gfs/space/private/fengzl/AgentJet/configs/fengzl/qwen3_17b_openclaw_swarm.yaml

# OpenClaw 环境变量脚本，source 之后就能直接用 openclaw 命令
/gfs/space/private/fengzl/OpenClaw/scripts/openclaw_env.sh

# 训练代理，把 OpenClaw 的一次请求扩成多个候选 episode
/gfs/space/private/fengzl/AgentJet/tutorial/opencode_build_openclaw_agent/fake_vllm_endpoint.py

# 相对奖励脚本，给多个候选回答打分
/gfs/space/private/fengzl/AgentJet/tutorial/opencode_build_openclaw_agent/on_compute_relative_reward.py

# 并发模拟用户请求脚本
/gfs/space/private/fengzl/AgentJet/tutorial/opencode_build_openclaw_agent/mock_user_request.py
```

## 3. 当前这套训练的重要配置

本次已经跑通的配置核心参数如下：

```yaml
model.path: /gfs/space/private/fengzl/Models/Qwen/Qwen3-1.7B
rollout.num_repeat: 4
data.train_batch_size: 4
trainer_common.total_training_steps: 25
trainer_common.n_gpus_per_node: 1
```

可以理解为：

- 每个用户请求会扩成 `4` 个候选回答
- 攒够一批样本后做一次训练更新
- 一共训练 `25` 个 step
- 当前用 `1` 张卡

## 4. 正式启动前的准备

建议先开 4 个终端窗口，分别负责：

- 终端 A：启动 swarm server
- 终端 B：启动 fake vLLM 训练代理
- 终端 C：启动并发请求
- 终端 D：看训练状态和曲线

终端使用规则先说清楚：

- 如果你按文档里的“前台启动”方式执行，就需要新建终端
- 因为 `ajet-swarm start`、`python -u fake_vllm_endpoint.py`、`tensorboard` 都会持续占住当前终端
- `mock_user_request.py` 如果最后一条不加 `&`，也会占住当前终端
- 所以最稳妥的做法，就是直接新建 `4` 个终端分别跑
- 如果你改用 `nohup ... &` 这种后台方式，就不一定需要新建那么多终端
- 也就是说，“需不需要新建终端”取决于你是前台跑还是后台跑

一个简单判断方法：

- 看到命令执行后终端一直不返回提示符，说明这个命令占住前台了，需要换一个新终端继续下一步
- 看到命令末尾有 `&`，或者用了 `nohup`，通常说明它已经转到后台了，可以继续在当前终端执行后面的命令

### 4.1 进入 OpenClaw 环境

```bash
source /gfs/space/private/fengzl/OpenClaw/scripts/openclaw_env.sh  # 加载 OpenClaw 环境变量和 CLI 包装器
```

### 4.2 激活 AgentJet 虚拟环境

```bash
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活当前已经能跑通训练的 Python 环境
```

### 4.3 确认模型和配置文件存在

```bash
ls /gfs/space/private/fengzl/Models/Qwen/Qwen3-1.7B  # 查看 Qwen3-1.7B 本地模型目录是否存在
ls /gfs/space/private/fengzl/AgentJet/configs/fengzl/qwen3_17b_openclaw_swarm.yaml  # 查看训练配置是否存在
```

## 5. 第一步：启动 AgentJet swarm server

这一步是训练主控端，负责接收 episode、组织 rollout、执行训练。

终端说明：

- 如果你用前台方式启动，这一步建议放在“终端 A”
- 这个命令会持续运行，所以执行后当前终端不能再拿来跑后续步骤
- 如果你改成 `nohup ... &` 后台启动，就可以不额外占一个终端

```bash
cd /gfs/space/private/fengzl/AgentJet  # 进入 AgentJet 项目目录
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活 AgentJet 环境
ajet-swarm start --swarm-port=10086  # 启动 swarm server，监听 10086 端口
```

如果想后台运行并保留日志，可以这样：

```bash
cd /gfs/space/private/fengzl/AgentJet  # 进入 AgentJet 目录
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活虚拟环境
nohup ajet-swarm start --swarm-port=10086 > /gfs/space/private/fengzl/AgentJet/swarm_server.log 2>&1 &  # 后台启动 swarm server 并把日志写入文件
```

## 6. 第二步：启动 fake vLLM 训练代理

这一步是最关键的“训练代理层”。OpenClaw 不直接请求真实 vLLM，而是先请求这个代理。这个代理会：

- 接收一次 `/v1/chat/completions`
- 把一个用户请求扩成多个候选回答
- 调用 AgentJet 的 `begin_episode/end_episode`
- 算 reward
- 选最好的一条返回给 OpenClaw

终端说明：

- 如果你用前台方式启动，这一步建议放在“终端 B”
- 这个命令会持续监听 `8090` 端口，所以也会一直占住当前终端
- 不要在跑着 `fake_vllm_endpoint.py` 的同一个前台终端里继续执行后面的 mock 请求
- 如果你用 `nohup ... &` 后台启动，就可以继续复用当前终端

启动命令如下：

```bash
cd /gfs/space/private/fengzl/AgentJet/tutorial/opencode_build_openclaw_agent  # 进入教程目录
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活 AgentJet 环境
export AJET_SWARM_URL=http://127.0.0.1:10086  # 指向刚刚启动的 swarm server
export MODEL_PATH=/gfs/space/private/fengzl/Models/Qwen/Qwen3-1.7B  # 指定训练使用的基础模型
export NUM_REPEAT=4  # 每个请求扩成 4 个候选回答
python -u fake_vllm_endpoint.py  # 启动本地 OpenAI 兼容训练代理，默认监听 8090
```

如果想后台运行：

```bash
cd /gfs/space/private/fengzl/AgentJet/tutorial/opencode_build_openclaw_agent  # 进入训练代理目录
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活虚拟环境
export AJET_SWARM_URL=http://127.0.0.1:10086  # 让代理知道 swarm server 地址
export MODEL_PATH=/gfs/space/private/fengzl/Models/Qwen/Qwen3-1.7B  # 让代理知道基础模型路径
export NUM_REPEAT=4  # 一个请求复制成 4 个候选 episode
nohup python -u fake_vllm_endpoint.py > /gfs/space/private/fengzl/AgentJet/fake_vllm_endpoint.log 2>&1 &  # 后台启动训练代理并保存日志
```

## 7. 第三步：确认 OpenClaw 侧已经接到 training-proxy

OpenClaw 的模型请求要打到 `fake_vllm_endpoint.py`，所以要检查配置。

终端说明：

- 这一步只是检查和测试，不会长期占住终端
- 可以在任意一个空闲终端执行
- 如果前两个服务都是后台跑的，这一步也可以直接在当前终端做

```bash
grep -n "training-proxy" /gfs/space/private/fengzl/OpenClaw/state/openclaw.json  # 查看 OpenClaw 是否把 primary 模型指向 training-proxy
grep -n "127.0.0.1:8090" /gfs/space/private/fengzl/OpenClaw/state/openclaw.json  # 查看 OpenClaw 是否把 provider 指向本地 8090 端口
```

如果想手工验证 OpenClaw 是否能直接请求到代理，可以执行：

```bash
source /gfs/space/private/fengzl/OpenClaw/scripts/openclaw_env.sh  # 加载 OpenClaw 环境
/gfs/space/private/fengzl/OpenClaw/cli/bin/openclaw agent --agent smoke_train_agent --message "你好，介绍一下你自己"  # 用一个已知 agent 发一条测试消息
```

说明：

- 出现 `gateway connect failed: pairing required` 不一定是致命错误
- 当前环境会自动 fallback 到 embedded 模式
- 只要后面能正常返回内容，就说明 OpenClaw 这层是通的

## 8. 第四步：启动并发请求，真正开始“边收样本边训练”

这里的关键点是：没有请求，就没有新的 rollout，训练就会停在某个 step 不动。

教程里的并发压测就是这一步。

终端说明：

- 如果你按文档里的前台方式执行，建议放在“终端 C”
- 前两条命令结尾有 `&`，会在后台跑
- 最后一条 `python -u mock_user_request.py` 没有 `&`，所以会占住当前终端
- 如果你想让“终端 C”也空出来，就把三条都改成 `nohup ... &`
- 这一步之后，训练是否继续推进，取决于这里的请求进程是否还在持续运行

```bash
cd /gfs/space/private/fengzl/AgentJet/tutorial/opencode_build_openclaw_agent  # 进入 mock 请求脚本目录
source /gfs/space/private/fengzl/OpenClaw/scripts/openclaw_env.sh  # 加载 OpenClaw 环境变量
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活 AgentJet 环境
python -u mock_user_request.py &  # 第 1 路并发用户，请求会持续送进 OpenClaw
python -u mock_user_request.py &  # 第 2 路并发用户，增加采样吞吐
python -u mock_user_request.py    # 第 3 路并发用户，前台运行便于直接观察输出
```

如果你不想把一个终端占住，也可以全放后台：

```bash
cd /gfs/space/private/fengzl/AgentJet/tutorial/opencode_build_openclaw_agent  # 进入请求脚本目录
source /gfs/space/private/fengzl/OpenClaw/scripts/openclaw_env.sh  # 加载 OpenClaw 环境
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活虚拟环境
nohup python -u mock_user_request.py > /gfs/space/private/fengzl/AgentJet/mock_user_request_1.log 2>&1 &  # 后台启动第 1 路并发流量
nohup python -u mock_user_request.py > /gfs/space/private/fengzl/AgentJet/mock_user_request_2.log 2>&1 &  # 后台启动第 2 路并发流量
nohup python -u mock_user_request.py > /gfs/space/private/fengzl/AgentJet/mock_user_request_3.log 2>&1 &  # 后台启动第 3 路并发流量
```

## 9. 第五步：实时查看训练有没有在推进

终端说明：

- 这一步建议放在“终端 D”
- 这些检查命令本身不会长期占住终端
- 所以你也可以把“状态检查”和“导出曲线”都放在同一个终端里做

### 9.1 看 swarm engine 状态

```bash
curl -s http://127.0.0.1:10086/get_engine_status  # 查询 swarm server 当前状态和 global_step
```

正常情况下会看到类似：

```json
{"engine_status":"ENGINE.ROLLING","engine_status_detail":null,"global_step":19}
```

几个常见状态可以这样理解：

- `ENGINE.ROLLING`：正在收集 rollout
- `ENGINE.ROLLING_POST`：rollout 收完，正在后处理
- `ENGINE.WEIGHT_SYNCING`：正在同步训练后的权重

### 9.2 看总共会跑多少步

```bash
grep -n "total_training_steps" /gfs/space/private/fengzl/AgentJet/configs/fengzl/qwen3_17b_openclaw_swarm.yaml  # 查看总训练步数配置
```

当前配置是：

```text
total_training_steps: 25
```

### 9.3 看关键进程是否都还活着

```bash
ps -eo pid,etimes,cmd | grep -E "ajet-swarm start|fake_vllm_endpoint.py|mock_user_request.py|tensorboard" | grep -v grep  # 检查 swarm、代理、并发请求和 TensorBoard 进程
```

### 9.4 看 GPU 有没有真的被占用

```bash
nvidia-smi  # 查看显卡利用率、显存占用和当前 GPU 进程
```

说明：

- 如果一瞬间看不到 GPU 进程，不代表完全没训练
- 在线 Agent RL 有很多 CPU/网络/等待请求阶段
- 真正的 GPU 前向和反向只占整个流程中的一部分时间

## 10. 第六步：启动 TensorBoard 看训练曲线

终端说明：

- 如果你用前台方式启动 TensorBoard，建议还是放在“终端 D”
- `tensorboard` 会持续占住当前终端，所以启动后如果还想继续查状态，需要再新开一个终端
- 如果你不想再开新终端，可以把 TensorBoard 改成后台方式启动

```bash
cd /gfs/space/private/fengzl/AgentJet  # 进入 AgentJet 根目录
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活虚拟环境
tensorboard --logdir /gfs/space/private/fengzl/AgentJet/tensorboard_log/fengzl_agentjet/qwen3_17b_openclaw_swarm --host 0.0.0.0 --port 6006  # 启动 TensorBoard 并监听 6006 端口
```

浏览器打开：

```text
http://127.0.0.1:6006/
```

如果你希望 TensorBoard 不占住终端，可以这样启动：

```bash
cd /gfs/space/private/fengzl/AgentJet  # 进入 AgentJet 根目录
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活虚拟环境
nohup tensorboard --logdir /gfs/space/private/fengzl/AgentJet/tensorboard_log/fengzl_agentjet/qwen3_17b_openclaw_swarm --host 0.0.0.0 --port 6006 > /gfs/space/private/fengzl/AgentJet/tensorboard_qwen3_17b_openclaw_swarm.log 2>&1 &  # 后台启动 TensorBoard 并写日志
```

## 11. 第七步：把训练曲线导出成图片和 CSV

下面这个脚本会把 TensorBoard 里的关键指标抽出来，保存为 PNG 和 CSV。

```bash
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活 AgentJet 环境，保证能导入 tensorboard/matplotlib
python - <<'PY'  # 直接在命令行里运行一段导出脚本
from tensorboard.backend.event_processing import event_accumulator  # 读取 TensorBoard event 文件
import matplotlib.pyplot as plt  # 画训练曲线
from pathlib import Path  # 处理路径
import csv  # 导出 csv

log_path = Path("/gfs/space/private/fengzl/AgentJet/tensorboard_log/fengzl_agentjet/qwen3_17b_openclaw_swarm/events.out.tfevents.1776239151.dd25b60a6c67f61b05484f5e173e3adf-taskrole1-0.42749.0")  # 当前训练对应的 event 文件
out_dir = Path("/gfs/space/private/fengzl/AgentJet/summary")  # 输出目录
out_png = out_dir / "qwen3_17b_openclaw_swarm_training_curves.png"  # 曲线图片输出路径
out_csv = out_dir / "qwen3_17b_openclaw_swarm_training_curves.csv"  # 曲线表格输出路径

ea = event_accumulator.EventAccumulator(str(log_path))  # 创建 event 读取器
ea.Reload()  # 重新加载 event 数据

selected = [  # 选择要画出来的关键指标
    "critic/rewards/mean",
    "critic/success_rate",
    "critic/real_reward",
    "actor/entropy",
    "actor/kl_loss",
    "timing_s/step",
    "perf/throughput",
    "response_length/mean",
]

fig, axes = plt.subplots(2, 4, figsize=(20, 8))  # 创建 2x4 的子图布局
axes = axes.ravel()  # 拉平成一维，方便循环绘图

for ax, tag in zip(axes, selected):  # 逐个指标绘图
    vals = ea.Scalars(tag)  # 读取某个指标的所有点
    x = [v.step for v in vals]  # 横轴用 global_step
    y = [v.value for v in vals]  # 纵轴用指标值
    ax.plot(x, y, marker="o", linewidth=2)  # 画折线
    ax.set_title(tag)  # 设置子图标题
    ax.set_xlabel("global_step")  # 设置横轴标签
    ax.grid(True, alpha=0.3)  # 打开浅色网格

fig.suptitle("Qwen3-1.7B OpenClaw Swarm Training Curves", fontsize=15)  # 设置整张图标题
fig.tight_layout(rect=(0, 0, 1, 0.95))  # 调整布局，避免标题遮挡
fig.savefig(out_png, dpi=180, bbox_inches="tight")  # 保存图片

steps = sorted({v.step for tag in selected for v in ea.Scalars(tag)})  # 收集所有出现过的 step
with out_csv.open("w", newline="", encoding="utf-8") as f:  # 以 utf-8 打开 csv 文件
    writer = csv.writer(f)  # 创建 csv 写入器
    writer.writerow(["global_step"] + selected)  # 写表头
    for step in steps:  # 逐个 step 写一行
        row = [step]  # 当前行先写 step
        for tag in selected:  # 再补上每个指标在这个 step 的值
            mapping = {v.step: v.value for v in ea.Scalars(tag)}  # 把该指标转成 step -> value 映射
            row.append(mapping.get(step, ""))  # 如果这个 step 没值，就写空字符串
        writer.writerow(row)  # 把一整行写入 csv

print(out_png)  # 打印生成的图片路径
print(out_csv)  # 打印生成的 csv 路径
print("last_step", max(steps) if steps else None)  # 打印当前导出的最新 step
PY
```

导出后的文件默认在：

```bash
/gfs/space/private/fengzl/AgentJet/summary/qwen3_17b_openclaw_swarm_training_curves.png  # 训练曲线图
/gfs/space/private/fengzl/AgentJet/summary/qwen3_17b_openclaw_swarm_training_curves.csv  # 曲线原始数据
```

## 12. 第八步：快速判断“卡住了”还是“只是暂时没流量”

如果你发现 `global_step` 很久不动，优先检查下面 3 件事。

### 12.1 先看是不是没有并发请求了

```bash
ps -eo pid,etimes,cmd | grep "mock_user_request.py" | grep -v grep  # 查看还有没有 mock 用户请求进程在跑
```

如果没有请求在持续发送，常见现象就是：

- `ajet-swarm` 还活着
- `fake_vllm_endpoint.py` 也还活着
- 但 `global_step` 长时间不变

这通常不是训练崩了，而是没有新的 rollout 流入。

### 12.2 再看训练代理有没有挂

```bash
ps -eo pid,etimes,cmd | grep "fake_vllm_endpoint.py" | grep -v grep  # 查看训练代理进程是否还活着
```

### 12.3 最后看 swarm engine 状态

```bash
curl -s http://127.0.0.1:10086/get_engine_status  # 再次确认 engine 是否还在正常切换状态
```

## 13. 第九步：训练结束后怎么停掉

先停并发请求，再停代理，再停 swarm server，顺序最稳妥。

```bash
pkill -f mock_user_request.py  # 停掉所有并发 mock 用户请求
pkill -f fake_vllm_endpoint.py  # 停掉 fake vLLM 训练代理
pkill -f "ajet-swarm start --swarm-port=10086"  # 停掉 swarm server
pkill -f "tensorboard --logdir /gfs/space/private/fengzl/AgentJet/tensorboard_log/fengzl_agentjet/qwen3_17b_openclaw_swarm"  # 停掉 TensorBoard
```

停完后可以复查：

```bash
ps -eo pid,etimes,cmd | grep -E "ajet-swarm start|fake_vllm_endpoint.py|mock_user_request.py|tensorboard" | grep -v grep  # 确认相关进程已经都退出
```

## 14. 一套最短复现流程

如果你只是想最快速地重新跑一次，可以直接按下面顺序执行。

这里默认你是“前台运行”模式，所以建议新建 `4` 个终端。

- 如果你全部改成后台命令，其实用 `1` 到 `2` 个终端也能跑
- 但对第一次复现的人来说，分成 `4` 个终端最清楚，也最不容易把服务误停

### 14.1 终端 A

```bash
cd /gfs/space/private/fengzl/AgentJet  # 进入 AgentJet 根目录
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活环境
ajet-swarm start --swarm-port=10086  # 启动 swarm server
```

### 14.2 终端 B

```bash
cd /gfs/space/private/fengzl/AgentJet/tutorial/opencode_build_openclaw_agent  # 进入训练代理目录
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活环境
export AJET_SWARM_URL=http://127.0.0.1:10086  # 指向 swarm server
export MODEL_PATH=/gfs/space/private/fengzl/Models/Qwen/Qwen3-1.7B  # 指定基础模型
export NUM_REPEAT=4  # 指定每个请求扩成 4 个候选
python -u fake_vllm_endpoint.py  # 启动 fake vLLM 训练代理
```

### 14.3 终端 C

```bash
cd /gfs/space/private/fengzl/AgentJet/tutorial/opencode_build_openclaw_agent  # 进入 mock 请求目录
source /gfs/space/private/fengzl/OpenClaw/scripts/openclaw_env.sh  # 加载 OpenClaw 环境
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活 AgentJet 环境
python -u mock_user_request.py &  # 启动第 1 路并发请求
python -u mock_user_request.py &  # 启动第 2 路并发请求
python -u mock_user_request.py    # 启动第 3 路并发请求
```

### 14.4 终端 D

```bash
curl -s http://127.0.0.1:10086/get_engine_status  # 看训练是否开始推进
```

```bash
source /gfs/space/private/fengzl/AgentJet/.venv/bin/activate  # 激活环境
tensorboard --logdir /gfs/space/private/fengzl/AgentJet/tensorboard_log/fengzl_agentjet/qwen3_17b_openclaw_swarm --host 0.0.0.0 --port 6006  # 打开 TensorBoard
```

## 15. 当前这条链路已经确认过的事实

这次已经验证过的点有：

- OpenClaw 可以正常接收请求并返回回答
- fake vLLM 训练代理可以把 1 个请求扩成多个候选
- reward 脚本可以给候选回答打分
- AgentJet swarm server 可以收到 episode 并推进 `global_step`
- TensorBoard 可以看到训练曲线
- 当前配置的总步数是 `25`

## 16. 一个最容易误解的点

这条链路不是：

```text
先让 OpenClaw 正常回答一次
-> 事后再把这次对话拿去训练
```

更准确的实际流程是：

```text
一次用户请求
-> 在 fake_vllm_endpoint.py 中被复制成多个候选回答
-> reward 脚本做相对打分
-> 所有候选一起提交给 swarm server
-> 只把分数最高的一条返回给 OpenClaw 用户
```

也就是说，它是在线 one-to-many 采样训练，不是单回答后处理训练。
