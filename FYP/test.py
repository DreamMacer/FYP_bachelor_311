import wandb
import cv2
from pettingzoo.mpe import simple_tag_v3

# 初始化 wandb
wandb.init(project="your_project_name", name="video_run")

# 创建环境
env = simple_tag_v3.env(render_mode="rgb_array")

# 重置环境
env.reset()

# 用于存储图像帧的列表
img_list = []

# 遍历环境中的每个智能体
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()

    env.step(action)

    # 获取当前帧图像
    img = env.render()

    # 将当前帧添加到图像列表
    img_list.append(img)

# 将图像帧保存为视频（使用 .mp4 格式）
frame_height, frame_width, _ = img_list[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码器
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (frame_width, frame_height))

for img in img_list:
    out.write(img)

out.release()

# 使用 ffmpeg 转换格式为 .webm
import subprocess
subprocess.run(['ffmpeg', '-i', 'output_video.mp4', '-vcodec', 'libvpx-vp9', '-acodec', 'libopus', 'output_video.webm'])

# 上传视频到 wandb（确保使用 .webm 格式）
wandb.log({"example_video": wandb.Video("output_video.webm")})

# 结束 wandb 运行
wandb.finish()

# 关闭环境
env.close()
