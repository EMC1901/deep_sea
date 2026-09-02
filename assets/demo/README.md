# 演示媒体

此目录存放本机演示与虚拟摄像头测试素材：

- `deep-sea-demo.mp4`：原始演示视频；
- `deep-sea-demo-camera.y4m`：由原始视频生成的 Chrome 虚拟摄像头文件。

视频文件体积较大，已通过 `.gitignore` 排除，不应提交到普通 Git 仓库。

重新生成 Y4M：

```powershell
python .\scripts\convert_mp4_to_y4m.py `
  ".\assets\demo\deep-sea-demo.mp4" `
  ".\assets\demo\deep-sea-demo-camera.y4m" `
  10 640 480
```

启动虚拟摄像头测试：

```powershell
.\scripts\dev.ps1 video-test
```
