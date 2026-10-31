# jp 指标历史修复发布：2026-10-31

状态：已获用户确认发布；代码与迁移数据已部署，受控重启验收通过；下一次自然收盘尚待验收。

## 版本与环境

- 源码提交：`34570e8d7dd84b44b66a4eb22b504a5b5fc9da0f`，已推送到`PerrywLiu/autoCrypto`的`btc_danyizhongcang`分支。
- jp镜像：`autocrypto-bot:history-20261031`。基于部署前固定镜像增量复制已测试源码，未更换依赖。
- 主服务`autocrypto`、看板`autocrypto-dashboard`使用新镜像。
- 全程保持`--dry-run`、`portfolio.mode=paper`，现货与合约真实交易开关仍关闭。生产配置字节未变。

## 备份与迁移

- 发布目录：`jp:/root/work/autocrypto-releases/20261031T021900Z-history`。
- 先停止主服务、看板、旧影子，冻结同一收盘时点，再备份源码、全部outputs、原影子data、配置和原镜像标签。
- 冻结时主账户与影子末根K线一致：开盘UTC `2026-10-30 20:00`，北京时间`2026-10-31 08:00`收盘。
- 使用新采集的状态、health和原实验DB重新执行离线重验，账户和全部11笔实验后现货成交精确匹配，才生成迁移检查点。
- 安装了377根连续历史；保留现金、BTC数量、合约累计盈亏、连续亏损数与冷却时间；4笔合约平仓、13笔现货调仓及3笔开仓事件均完整保留。
- 权益数据库完整性通过：冻结备份48545条，重启验证后48547条，历史未清空。
- 新迁移校验位于服务器`migration-verification.json`；本地副本位于`output/releases/20261031T021900Z-history/refreshed-migration/`。

## 原实验与新实验

- 原`autocrypto-shadow`容器已停止，原`/root/work/autocrypto-shadow/data`保持原样，数据库哈希与冻结备份一致。停止时超时退出137是本次显式停止操作，不视为此前运行故障或OOM证据。
- 新实验目录：`/root/work/autocrypto-shadow-history-20261031/data`。
- 新容器：`autocrypto-shadow-history-20261031`。从迁移后的同一paper账户、同一完整历史建立基线与1%缓冲候选，独立记账，不继承旧候选收益。
- 源账户与配置只读挂载；无网络、无公开端口、无订单权限；启动后`comparison_validated=true`、`indicator_history_matched=true`。

## 已完成验证

- 本地131项测试、18个subtests通过；jp独立镜像在无网络环境运行131项unittest通过，没有在主bot容器内运行测试。
- 裸pytest仍有既有`scripts/test_1h.py`缺少`src.data.downloader`的收集问题，未混入本次修复。
- 主服务与新影子部署后分别进行了受控容器重启。
- `controlled-restart-verification.json`的28项检查全部通过：主检查点哈希、账户与旧台账不变，历史前缀完整，重启后新心跳与新影子采样存在，账户核对通过，原实验哈希不变，SQLite完整性通过。
- 公网看板接口回读成功，主服务健康、无暂停、剩余冷却2根。人工重启通过不代表自然收盘通过。

## 尚待自然收盘验收

预计北京时间2026-10-31 12:00处理下一根4小时自然收盘。已建立临时heartbeat`autocrypto-2`，每小时03分复核，12:03之前不提前验收、不重复通知；成功后暂停自身。

只读验收命令：

```bash
ssh jp 'python3 /root/work/autocrypto-releases/20261031T021900Z-history/verify_release.py --natural'
```

通过标准：实际自然新K线到达、历史检查点增加并持久化、新影子至少处理1根且与主账户同收盘核对通过、paper/配置/原实验保持不变、数据库完整性与主服务健康通过。额外保存04:00 UTC后的真实主循环日志及公网状态回读。

成功后新增`docs/releases/2026-10-31-indicator-history-natural.json`，更新本文并同步GitHub；不得以手动回放或模拟触发替代自然验收。

## 回退条件

保留部署前镜像`autocrypto-bot:pre-history-20261031`及完整备份。回退前先保存当前状态和实验，核对是否已有新自然收盘或成交；不得把冻结时的旧账户直接覆盖到已继续运行的账户。所有回退仍必须保持paper并保全原始与新实验。
