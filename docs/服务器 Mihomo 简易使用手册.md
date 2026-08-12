# 服务器 Mihomo 简易使用手册

Mihomo 位置：

```bash
/home/linglong/apps/mihomo
```

配置文件：

```bash
/home/linglong/apps/mihomo/config/config.yaml
```

本地代理地址：

```text
http://127.0.0.1:7890
```

## 一、启动 Mihomo

执行：

```bash
cd /home/linglong/apps/mihomo

tmux new -s mihomo
```

进入 tmux 后运行：

```bash
./mihomo \
  -d /home/linglong/apps/mihomo/config \
  -f /home/linglong/apps/mihomo/config/config.yaml
```

看到代理启动日志后，让它在后台运行：

```text
先按 Ctrl+B
松开后按 D
```

## 二、检查是否启动成功

```bash
ss -lntp | grep 7890
```

正常应看到：

```text
127.0.0.1:7890
```

也可以查看 tmux 会话：

```bash
tmux ls
```

应看到：

```text
mihomo
```

## 三、重新进入 Mihomo 界面

```bash
tmux attach -t mihomo
```

再次退出但不停止服务：

```text
Ctrl+B，然后按 D
```

## 四、停止 Mihomo

直接结束 tmux 会话：

```bash
tmux kill-session -t mihomo
```

检查是否停止：

```bash
ss -lntp | grep 7890 || echo "Mihomo 已停止"
```

## 五、常用操作汇总

```bash
# 启动
cd /home/linglong/apps/mihomo
tmux new -s mihomo
./mihomo -d /home/linglong/apps/mihomo/config \
  -f /home/linglong/apps/mihomo/config/config.yaml

# 查看运行界面
tmux attach -t mihomo

# 检查代理端口
ss -lntp | grep 7890

# 停止
tmux kill-session -t mihomo
```

服务器重启后，Mihomo 不会自动启动，需要重新执行启动命令。
