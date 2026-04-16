系统可以拆成 4 层：

- 流量入口层：OpenClaw CLI / gateway，负责接住真实用户请求。
- 关键文件是 openclaw_env.sh、onboard_training_proxy.sh、openclaw_cli_wrapper.sh。
- 训练代理层：fake_vllm_endpoint.py 把一次 OpenClaw /chat/completions 扩成 NUM_REPEAT 个 episode，向 swarm server 申请算力，收集多条回答，再选最优答案返回给 OpenClaw。
- 奖励层：on_compute_relative_reward.py 不是 reward model，而是一个组合式 judge。
- 它把 外向性 + relevance + diversity 做加权，再乘上 quality gate，质量差的回答直接被压到接近 0。
- 训练执行层：AgentJet swarm + verl。server 端用 dummy task 占位，真正的用户 query 来自 proxy/client。训练主控还是 AgentJet
