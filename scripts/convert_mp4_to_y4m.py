import sys
import subprocess
import imageio_ffmpeg


def main():
    if len(sys.argv) < 3:
        print(
            '用法: python scripts/convert_mp4_to_y4m.py '
            '"输入.mp4" "输出.y4m" [fps] [width] [height]'
        )
        sys.exit(1)

    in_mp4 = sys.argv[1]
    out_y4m = sys.argv[2]
    fps = sys.argv[3] if len(sys.argv) > 3 else "10"
    w = sys.argv[4] if len(sys.argv) > 4 else "640"
    h = sys.argv[5] if len(sys.argv) > 5 else "480"

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    # 注意：y4m 是“几乎无压缩”的，会比较大；建议先用 10fps、640x480
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        in_mp4,
        "-vf",
        f"scale={w}:{h},fps={fps}",
        "-pix_fmt",
        "yuv420p",
        out_y4m,
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("OK:", out_y4m)


if __name__ == "__main__":
    main()
