# Orbbec RGB-D Collector

面向 Apple Silicon Mac 的独立 Orbbec Gemini 335/336L 数据采集工具。

它直接独占 USB 相机，不依赖机器人、IK_replay、teleimager、ZMQ 或 YOLO。
网页同时显示 RGB 与深度伪彩色，并将每次采集完整保存为：

- RGB 图像；
- 相机原始分辨率的 uint16 深度；
- 对齐到 RGB 像素坐标的 uint16 深度；
- 当前相机的内参、畸变、深度到彩色外参和深度尺度；
- 彩色/深度硬件时间戳、帧序号和主机时间。

## 1. Mac 环境

要求：

- Apple Silicon Mac（M1/M2/M3/M4/M5）；
- macOS 13 或更高版本；
- Python 3.10～3.13；
- USB 3.0 数据线和接口；
- Orbbec Gemini 335、336、335L 或 336L。

官方 SDK v2 的 PyPI 包名是 `pyorbbecsdk2`，Python 导入名仍是
`pyorbbecsdk`。Gemini 335/336(L) 官方推荐固件版本为 1.8.10；较旧固件若能
稳定输出所选 RGB-D 档位也可先使用，遇到同步或 UVC 问题时再按官方流程升级。

推荐使用 Conda 独立环境：

```bash
cd <orbbec_rgbd_collector 项目目录>

conda create -n rgbd-collector python=3.11
conda activate rgbd-collector
python -m pip install --upgrade pip
python -m pip install -e .
```

也可以直接使用已有的 Conda 环境：

```bash
conda activate fastapi
cd <orbbec_rgbd_collector 项目目录>
python -m pip install -e .
```

`run.sh` 会优先使用当前已激活的 Conda 环境；没有激活 Conda 时才尝试项目内的
`.venv` 或系统 `python3`。如果希望使用 venv：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

首次访问相机时，请允许 Terminal（或使用的终端应用）访问相机/USB 设备。
通常 macOS 不需要额外驱动或环境配置。

如果 SDK 报 `Permission denied` 或 `Library not loaded`，官方建议保留虚拟环境
变量后提权运行：

```bash
sudo -E ./run.sh
```

不要同时打开 Orbbec Viewer、其他相机采集程序或本项目的第二个实例。项目同时
使用进程文件锁和 SDK 设备独占；相机被占用时会明确报错，不会静默打开错误设备。

## 2. 启动

只有一台 Orbbec 时：

```bash
./run.sh
```

浏览器打开：

```text
http://127.0.0.1:7003
```

在另一个终端启动已拍摄数据的点云预览：

```bash
conda activate rgbd-collector   # 或你实际安装本项目的 Conda 环境
cd <orbbec_rgbd_collector 项目目录>
./run-pointcloud.sh
```

浏览器打开：

```text
http://127.0.0.1:17002
```

点云服务不连接、不占用相机。它按需读取 `datasets` 中的 `color.jpg`、
`depth_aligned.png`、内参和深度尺度重建点云，因此不额外保存重复的 PLY 文件。
若会话中记录了彩色畸变系数，重建时会先做像素去畸变再生成三维射线。
`17002` 页面还可拟合柜面、点选三维目标并把结果写入会话目录下的
`annotations.jsonl`。这不会修改原始 RGB-D 数据。

YOLO 是可选的离线分析能力。安装并传入模型后，实例分割模型会按 Mask
给三维点语义着色；普通检测模型则兼容回退为检测框：

```bash
python -m pip install -e '.[yolo]'
./run-pointcloud.sh --model /path/to/Xuanniu.pt --conf 0.25 --device mps
```

如果当前 Mac 的 PyTorch/MPS 组合不支持模型中的算子，可去掉 `--device mps`
或改为 `--device cpu`。不配置模型时，柜面拟合、三维点选和标注保存仍然可用；
以后传入模型即可对历史 RGB-D 帧补做 YOLO。

若采集时使用了自定义 `--data-dir`，点云服务也要传入相同目录：

```bash
./run-pointcloud.sh --data-dir /Volumes/RGBD_DATA/datasets
```

有多台相机时必须指定序列号：

```bash
./run.sh --serial CP0BB53000FS
```

允许局域网其他电脑打开前端：

```bash
./run.sh --host 0.0.0.0
```

修改数据目录或流档位：

```bash
./run.sh \
  --data-dir /Volumes/RGBD_DATA/datasets \
  --color-width 1920 --color-height 1080 \
  --depth-width 1280 --depth-height 800 \
  --fps 30
```

程序严格要求指定的彩色/深度档位存在。如果相机固件不提供默认档位，错误信息会
列出全部可用档位，据此修改启动参数即可。

## 3. 前端使用

1. 确认右上角显示“相机独占中”，RGB 和深度画面都在更新。
2. 输入会话名称。
3. 点击“拍摄一组”保存一组同步 RGB-D 数据；或者设置间隔、最大帧数后点击
   “开始连续采集”。
4. 停止连续采集后，等待“队列”归零。
5. 点击“结束当前会话”，确保所有文件写盘并关闭 manifest。

连续采集间隔可设为 0.1～3600 秒，默认 1 秒。写盘速度跟不上时使用有界队列，
前端会显示失败/丢弃数量，不会用不完整文件冒充成功数据。

## 4. 数据目录

```text
datasets/
└── 20260819_150000_switch_samples/
    ├── session.json
    ├── manifest.jsonl
    └── frames/
        └── 000001_1787123456789000000/
            ├── color.jpg
            ├── depth_raw.png
            ├── depth_aligned.png
            └── frame.json
```

### `session.json`

保存本次实际连接相机的：

- 型号、序列号、固件版本和 SDK 版本；
- 彩色和深度流的分辨率、FPS、像素格式；
- 彩色/深度内参和畸变；
- 深度坐标到彩色坐标的旋转和平移；
- SDK 对齐方式和深度尺度。

不能拿机器人上另一台相机的标定替代这里的数据。即使型号相同，不同物理相机的
内参和外参也会不同。

### 深度文件

`depth_raw.png` 与 `depth_aligned.png` 都是无损 uint16 PNG：

```text
实际深度（毫米） = PNG 像素值 × frame.json 中的 depth_scale.value
```

其中：

- `depth_raw.png` 保留深度传感器原生几何，后续使用深度相机内参与
  `depth_to_color` 外参；
- `depth_aligned.png` 已由 Orbbec `AlignFilter` 对齐到 `color.jpg`，可以直接
  用同一个 `(u, v)` 查 RGB 与深度；
- 网页中的伪彩色深度仅用于观察，永远不会写入上述深度文件。

### 点云与目标标注

对齐深度像素 `(u, v)` 的深度为 `z_mm` 时，可使用彩色内参：

```text
X = (u - cx) * z / fx
Y = (v - cy) * z / fy
Z = z
```

点云由 `17002` 服务按需重建，不另存重复的 PLY。页面执行以下流程：

1. 可选：用 YOLO 对历史 `color.jpg` 推理，并按实例 Mask（检测模型按框）
   给三维点着色；
2. 用 RANSAC 拟合场景中的主平面，并保存相机系法向；
3. 人工在三维点云中选择目标 TCP，或在 RGB 图中点击像素并用对齐深度
   反投影到同一个三维目标；
4. 以相机光心到拟合墙面的垂足为原点建立右手墙面坐标系：`+Y` 沿法向
   进入墙内，`+Z` 为相机向上方向在墙面上的投影，`+X = +Y × +Z`；
5. 保存目标的相机坐标和墙面 `(X, Y, Z)` 坐标。键盘 `A/D` 沿墙面
   `X−/X+`，`W/S` 沿 `Z+/Z−`，`Q/E` 沿 `Y−/Y+` 微调。

页面会在切换帧后自动拟合墙面。需要固定真实竖直方向时，点击“校准并保存
坐标系”，先选择墙面坐标原点，再选择位于其右侧的点。两点连线投影到
墙面后定义 `+X`，并由 `+Z = +X × +Y` 得到墙面向上方向。标定结果写入数据根目录的
`wall_coordinate_calibrations/{session_id}/{frame_id}.json`。每帧标定相互
独立，未标定帧继续使用自己的自动拟合坐标系。两点建议相距至少 30–50 cm。

“多平面分割着色 + 长宽轴（调试）”会对当前显示点云重复执行 RANSAC，
提取共面候选。中位深度最远的候选整体保留为 `P0`，即使它呈十字形、T 形、
“艹”形或包含多条细长区域也不拆开。其余候选再按图像邻接关系拆分不连通的
物理平面片。将各平面片与 P0 的点投影到 P0 平面并忽略墙面 Y 方向，仅保留
墙面 XZ 方向最近点距离不超过 10 mm 的平面片；
切缝或空隙分开的区域即使共面也使用不同编号和颜色。
“最少平面点数”可在前端调整，并在边界计算前过滤较小平面片。每个非 P0
平面会提取最靠近 P0 点集的边界点，用二维 RANSAC 拟合并按长度排序最多三条边；
严格拟合失败时使用这些边界点的最长主轴兜底。每个彩色平面片中心显示
`P编号`，并用同色加粗线绘制检测到的边界并标注长度；不再绘制平面的长宽轴。
未归属平面的点显示为深灰色。
该模式不会写入或修改目标标注，并与 YOLO 语义着色互斥。

未手动标定的帧会先剔除与相机原始上方向直线夹角小于 45° 的候选，再按无方向
夹角对上述 P0 最近边界分组，默认组内夹角不超过 1°，且至少来自两个不同平面的
线才构成可靠同向组；组内任意两条线都必须满足角度
阈值，每个平面最多投票一条，并去除短于组内最长线 25% 的成员。孤立的超长
误检线会被剔除。
系统优先选择成员最多、总长度最大的组，并以组内最长线作为 `+X`。调试视图将
同向组高亮，并把代表线显示为黄色虚线，标注“最终 X · P编号/边编号/长度”。
没有可靠同向组时，
再依次尝试次级窄平面长轴、两个大平面的交线和相机方向。按 `X` 完成的逐帧
手动标定始终具有最高优先级，分析面板会显示实际采用的来源。如果认可当前自动
结果，可点击“确认并保存当前坐标系”或按 `V`，直接逐帧保存当前原点和 X/Y/Z 轴；以后
打开该帧时优先加载已确认结果，无需重新选择标定点，并跳过 RANSAC、多平面分割
和边界拟合。仅在主动启用“多平面分割着色 + 边界线（调试）”时重新执行柜面分析。
显示点云和 YOLO 点云仍可保留最多 1,000,000 点，但找 X/柜面分析默认均匀降采样
到最多 200,000 点；前端可通过“找 X / 柜面分析最大点数”单独调整，不影响显示
密度和最终保存的 YOLO 点云密度。

点云页面会按 `IK_replay` 的“垂直度调试”定义显示当前帧的相机到平面垂距、
Yaw 偏差、Pitch 偏差和总倾角（`0°` 表示相机光轴正对平面）。点击“一键计算并
保存全部帧朝向与距离”后，会用每帧最多 60,000 点执行轻量主平面拟合，并写入
会话目录的 `camera_plane_pose_measurements.json`。距离定义为相机光心到拟合
无限平面的垂直距离 `|n·center|`，不是中心像素深度或最近点距离。

采集和标注不必同时完成。只要原始 RGB、对齐深度和标定元数据已保存，就可以
先连续采集，之后再离线补做 YOLO、平面拟合和目标点标注。

保存目标标注时，系统只选择当前帧置信度最高的 YOLO 实例，将其完整语义点云
写入 `yolo_semantic_pointclouds/<session>/<frame>.npz`。文件包含相机坐标
`xyz_camera_m`、墙面坐标 `xyz_wall_m` 和 `rgb`；同名 JSON 保存检测类别、
置信度、Mask/框、点云质心、坐标轴以及目标点相对语义质心的墙面坐标差。
每帧支持数字键 `1`–`9` 对应的九个独立目标点位，按 `0` 显示或隐藏当前帧的
全部点位，按 `C` 保存当前点位；旧版单点标注自动作为点位 `1`。同名 YOLO
JSON 的 `targets` 会分别保存各点位与
最高置信度语义点云质心的关系。

页面的“自动找点算法版本”可选择版本化的自动找点模型，生成初值后仍可用
`A/D/S/W/Q/E` 微调并按 `C` 保存。`0.1.0` 与 `0.2.0` 使用最高置信度
YOLO 点云的三维中位中心作为参考；其中 `0.2.0` 基于34帧人工确认数据，固定
墙面偏移为 `(+43.30, +9.00, -15.04) mm`。

`0.1.0-s` 改用 YOLO Mask 内拟合矩形的粉色几何中心，只允许用于已经保存
墙面坐标系的帧。点位1的固定墙面偏移为
`(+47.94, +5.86, -19.53) mm`。

默认模型 `0.2.0-s` 在59帧有效点位1数据上复核了同一偏移，留一验证 RMSE
为 `2.29 mm`。该模型根据最高置信度检测类别选择输出点位：

- `远方`：生成点位1，偏移 `(+47.94, +5.86, -19.53) mm`
- `就地`：生成点位3，偏移 `(-47.94, +5.86, -19.53) mm`

点位3规则是点位1沿墙面 X 轴的几何镜像，目前尚无独立点位3标注验证。在已有
对应点位的帧中运行时，页面会同时显示原始点和预测误差。

## 5. API

```text
GET  /api/status
POST /api/capture
POST /api/record/start
POST /api/record/stop
POST /api/session/close
POST /api/camera/restart
GET  /api/sessions
GET  /stream/color.mjpg
GET  /stream/depth.mjpg
```

手动拍摄示例：

```bash
curl -X POST http://127.0.0.1:7003/api/capture \
  -H 'Content-Type: application/json' \
  -d '{"session_name":"switch_samples"}'
```

## 6. 测试

不接相机即可运行存储测试：

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

Mac 真机验收：

1. 启动后设备名称、序列号和两路 profile 正确；
2. RGB 与深度预览同步更新；
3. 手动保存后生成四个文件，两个深度 PNG 读回仍是 uint16；
4. 连续采集 20 组，manifest 恰好 20 行且每行引用的文件都存在；
5. 停止服务后 Orbbec Viewer 能重新打开相机；
6. Orbbec Viewer 占用相机时，本项目启动应明确失败；
7. 拔掉相机后状态页显示错误，重新插入后点击“重新连接相机”恢复。
