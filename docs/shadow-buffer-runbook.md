# 4小时＋1%买回缓冲影子并跑

2026-10-14：用户确认发布，独立影子容器已部署至 jp。本机原实验数据库迁移并续跑，起始账户与历史样本未重置；原机器人、配置、看板及源账户未改。

## 单一差异

基线和候选从同一paper现金、BTC数量、合约持仓、累计盈亏及冷却状态起步。候选仅在发生过卖出后，对现货核心仓加仓要求 `收盘价 > 4h EMA200 × 1.01`；此前已有卖出记录可用于识别卖出状态，但历史成交不计入实验次数和费用。减仓、合约信号与风控、原加仓步长50个百分点均保留。不添加确认期或分批。

## 启动与续跑

```bash
.venv/bin/python scripts/run_shadow.py --once
.venv/bin/python scripts/run_shadow.py --interval 60
```

默认仅通过SSH从 `jp:/root/work/autoCrypto` 读取paper状态、配置和心跳，并获取带 `x-simulated-trading: 1` 的OKX公共4H行情。不会读取API密钥、不调用账户或下单接口、不在交易容器内做计算，不写服务器文件。两份账户与预热历史保存在本机独立SQLite。

实验默认目录 `output/shadow-buffer-1pct`。存在数据库时续跑，不重新播种；同一数据库有排他进程锁。每根自然收盘仅处理一次，两份账户、样本和事件在同一事务保存，失败回滚本根，成功处理的前序K线保留。缺口、源配置变化、源心跳过期或暂停时停止推进并保留最近指标，不静默跳过。最多由公共接口补取300根K线，较长离线缺口需人工核对。

初始化按捕获时的模拟盘最新价格为两份账户共同估值；其后指标按4小时自然收盘更新。初始采样的 `candle_ts` 是已收盘预热锚点，实际启动时间与初始估值价格分别保存在 `started_at` 和 `initial_mark_price`，不能把锚点当启动时间。

报告包括启动后收益、最大收盘回撤、现货调仓笔数和手续费，卖出后24小时买回及高于卖价买回次数、被拦截加仓、候选减基线的账户权益差。`underexposed_rise` 记录上涨区间期初少持BTC带来的毛收益差。此值不抵减下跌期间防御收益，不是候选净损失；整体权益差才包含账户净效果。

费用保持当前paper口径：现货按配置扣费；合约手续费/资金费仍不完整，不把结果称为已扣全部真实费用的净收益。当前不重构线上成本模型。

## 回读结果

- `shadow_status.json`：指标、最近读取错误、与线上基线核对结果。
- `影子并跑比较.md`：中文比较表。
- `equity_samples.csv`：同价估值序列。
- `decision_events.json`：被拦截买回、快速买回和少持上涨事件。
- `shadow.sqlite3`：完整两份状态、源快照与行情历史。

同一收盘下核对基线现金、BTC、累计合约盈亏及持仓关键字段。只有 `comparison_validated=true` 才能把本次指标标作已核对的线上规则对比。源机器人重启可能改变指标预热窗口；不一致会明确标记，绝不覆盖源账户或强行同步候选。

服务器运行目录 `/root/work/autocrypto-shadow`，容器 `autocrypto-shadow`，镜像 `autocrypto-shadow:20261014`，restart策略为unless-stopped。无env_file或API密钥、无公开端口，源outputs与config只读挂载；实验只写独立data目录。只新增影子容器，不依赖电脑唤醒，不重启原bot/dashboard。服务器本地读取使用 `--source-directory /source`，不再通过SSH读取自身。原本机实验及文件保留，原托管进程已结束。

停止：核对 `runner.pid` 对应命令为 `scripts/run_shadow.py`，只停止该进程；保留数据库，下次启动自动续跑。不会停止现有bot或dashboard。

## 验证

104项测试通过，含缓冲严格边界、相同减仓、拦截与恢复、重复K线、重启续跑、两个账户事务回滚、无生产文件写入和禁止订单访问。启动后已回读同源行情，基线与线上账户一致。首根新自然收盘和实际策略分歧仍待观察。

## 服务器发布验收（2026-10-14）

独立离线镜像105项测试通过。数据库原起始时间2026-10-13 21:43北京时间和原样本保留，服务器已补处理到累计5根自然收盘。影子容器健康、源基线核对通过，原生产源码/config/compose、账户与交易JSON哈希未变，bot/dashboard容器ID及启动时间未变，原24515条权益采样零缺失或修改。尚无新调仓或缓冲拦截分歧。

服务器报告在 `/root/work/autocrypto-shadow/data/影子并跑比较.md` 与 `shadow_status.json`，每分钟更新；本机 `output/shadow-server-20261014` 为本次服务器回读快照，不承诺自动同步。发布备份与验证位于 `/root/work/autocrypto-shadow/releases/20261014-shadow`。

仅管理影子服务：

```bash
cd /root/work/autocrypto-shadow
docker compose -p autocrypto-shadow -f compose.yaml ps
docker compose -p autocrypto-shadow -f compose.yaml logs --tail 30 shadow
# 停止只影响影子；data保留，重新up将续跑
docker compose -p autocrypto-shadow -f compose.yaml stop shadow
```
