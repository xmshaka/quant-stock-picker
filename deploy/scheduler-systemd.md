# 每日扫描调度器 systemd 部署说明

调度器入口：

```bash
/root/.openclaw/workspace/quant-stock-picker/venv/bin/python -m data.scheduler
```

默认每日 16:00 执行，配置来自 `config/settings.py`：

- `daily_scan_enabled`
- `daily_scan_hour`
- `daily_scan_minute`
- `daily_scan_lookback_days`
- `daily_scan_alert_file`
- `daily_scan_failure_threshold`

## 手动验证

```bash
cd /root/.openclaw/workspace/quant-stock-picker
venv/bin/python -m data.scheduler --once --symbols 600519,000001 --lookback-days 10 --max-workers 2
```

## 安装为 systemd 服务

> 需要 root 权限。确认无误后再执行。

```bash
sudo cp /root/.openclaw/workspace/quant-stock-picker/deploy/quant-scheduler.service /etc/systemd/system/quant-scheduler.service
sudo systemctl daemon-reload
sudo systemctl enable quant-scheduler
sudo systemctl start quant-scheduler
sudo systemctl status quant-scheduler --no-pager
```

## 查看日志

```bash
journalctl -u quant-scheduler -f
```

## 停止/禁用

```bash
sudo systemctl stop quant-scheduler
sudo systemctl disable quant-scheduler
```

## 看板状态页

Streamlit 页面：`dashboard/pages/5_数据状态.py`

读取报告文件：`logs/daily_scan_alerts.log`
