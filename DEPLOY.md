# autoCrypto 部署文档

> 自适应混合策略实盘/模拟盘交易机器人

---

## 1. 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux (Ubuntu 20.04+ / Debian 11+) 或 macOS |
| Python | 3.11+ |
| 内存 | 最小 512MB，推荐 1GB |
| 磁盘 | 1GB（历史K线数据） |
| 网络 | 稳定互联网（延迟 <100ms 到 OKX API） |

---

## 2. 服务器初始化

### 2.1 安装依赖

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git curl

# 安装 uv (Python 包管理器)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env
```

### 2.2 克隆项目

```bash
# 从你的电脑传到服务器（选一种）
# 方式A: scp 整个目录
scp -r /Users/jing.guo/claudeCode/autoCrypto user@your-server:/home/user/

# 方式B: git（如果已推送到仓库）
git clone <your-repo-url> /home/user/autoCrypto

cd /home/user/autoCrypto
```

### 2.3 安装 Python 依赖

```bash
uv sync
```

---

## 3. 配置

### 3.1 API 密钥

```bash
cat > .env << 'EOF'
OKX_API_KEY=你的API_Key
OKX_SECRET=你的Secret_Key
OKX_PASSPHRASE=你的Passphrase
EOF

# 限制权限（防止其他用户读取）
chmod 600 .env
```

> ⚠️ `.env` 已在 `.gitignore` 中，不会被提交到 git。

### 3.2 调整策略参数（可选）

编辑 `config/live.yaml`：

```yaml
strategy:
  symbol: "BTC/USDT"     # 交易币种

risk:
  initial_capital: 500    # 初始资金 USDT
  risk_per_trade: 0.02    # 单笔风险 2%
  leverage: 3             # 杠杆
  max_daily_loss_pct: 0.05  # 日熔断 5%
```

---

## 4. 运行模式

### 4.1 纸上交易（推荐起步 — 只看不下单）

```bash
uv run python scripts/live.py --dry-run
```

- 使用实盘行情数据
- 模拟信号和盈亏，**不下真实订单**
- 交易记录自动保存到 `outputs/live_trades.json`

### 4.2 模拟盘（需要 OKX Demo API Key）

```bash
uv run python scripts/live.py
```

- 连接 OKX 模拟盘环境
- 使用虚拟资金真实下单

### 4.3 实盘（⚠️ 真金白银）

```bash
uv run python scripts/live.py --live
```

---

## 5. 后台运行（生产环境）

### 5.1 使用 systemd（推荐）

创建服务文件：

```bash
sudo cat > /etc/systemd/system/autocrypto.service << 'EOF'
[Unit]
Description=autoCrypto Trading Bot
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/your-username/autoCrypto
Environment="PATH=/home/your-username/.cargo/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/your-username/.cargo/bin/uv run python scripts/live.py --dry-run
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

# 启动
sudo systemctl daemon-reload
sudo systemctl enable autocrypto
sudo systemctl start autocrypto

# 查看状态
sudo systemctl status autocrypto

# 查看日志
sudo journalctl -u autocrypto -f
```

### 5.2 使用 nohup（简单）

```bash
# 启动
nohup uv run python scripts/live.py --dry-run > outputs/bot.log 2>&1 &

# 记录进程ID
echo $! > bot.pid

# 查看日志
tail -f outputs/bot.log

# 停止
kill $(cat bot.pid)
```

---

## 6. 监控与维护

### 6.1 查看实时状态

```bash
# 最新日志（最后 10 行）
tail -10 outputs/bot.log

# 实时滚动
tail -f outputs/bot.log
```

### 6.2 查看收益

```bash
uv run python scripts/report.py
open outputs/performance.html    # 本地打开图表
```

### 6.3 健康检查

```bash
# 检查进程是否运行
ps aux | grep live.py

# 检查最新K线是否正常更新
grep "新K线" outputs/bot.log | tail -3
```

---

## 7. 升级流程

```bash
cd /home/your-username/autoCrypto

# 停止 bot
sudo systemctl stop autocrypto
# 或: kill $(cat bot.pid)

# 拉取最新代码
git pull origin main
# 或: scp 新代码到服务器

# 更新依赖
uv sync

# 重启
sudo systemctl start autocrypto
```

---

## 8. 安全注意事项

1. **API Key 权限最小化**: OKX API 只需「读取 + 交易」权限，不要开「提现」
2. **IP 白名单**: 在 OKX API 设置中绑定服务器 IP
3. **不要共享 .env**: `.env` 文件包含密钥，永远不要提交到 git
4. **先跑纸上交易**: 至少观察 2 周信号质量再切实盘
5. **小资金起步**: 实盘从 500U 开始，稳定盈利后再加
6. **设置日熔断**: `max_daily_loss_pct: 0.05` 每天最多亏 5%

---

## 9. 目录结构

```
autoCrypto/
├── .env                     # ⚠️ API密钥 (不要提交)
├── config/
│   └── live.yaml            # 实盘配置
├── src/
│   ├── live/bot.py          # 实盘机器人
│   ├── strategies/          # 策略库
│   └── data/                # 数据下载/处理
├── scripts/
│   ├── live.py              # 启动脚本
│   └── report.py            # 收益分析
└── outputs/
    ├── bot.log               # 运行日志
    ├── live_trades.json      # 交易记录
    └── performance.html      # 收益图表
```

---

## 10. 常见问题

### Q: 启动报 "APIKey does not match current environment"?
A: 你的 API Key 是实盘环境的，使用 `--dry-run` 模式即可（只读行情，不下单）。

### Q: 如何确认 bot 在正常工作？
A: `tail outputs/bot.log` 应该能看到每 4 小时一条 "新K线" 日志。

### Q: 服务器时区不对怎么办？
A: `sudo timedatectl set-timezone UTC`（OKX 使用 UTC 时间）。

### Q: 如何换币种？
A: 编辑 `config/live.yaml`，把 `symbol` 改成 `ETH/USDT`、`SOL/USDT` 等。
