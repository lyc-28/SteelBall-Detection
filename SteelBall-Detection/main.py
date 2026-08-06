import gc, time
from media.sensor import Sensor
from libs.PipeLine import PipeLine
from libs.YOLO import YOLO11
from machine import UART, FPIOA

# 1. 初始化 FPIOA 和 串口2 (UART2)
fpioa = FPIOA()
fpioa.set_function(11, FPIOA.UART2_TXD)  # GPIO11 -> TX2
fpioa.set_function(12, FPIOA.UART2_RXD)  # GPIO12 -> RX2

uart2 = UART(UART.UART2, baudrate=115200, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)

# 2. 模型与屏幕配置
kmodel_path = "/sdcard/models/steelball2.kmodel"
labels = {0: '0'}
model_input_size = [320, 320]
rgb888p_size = [640, 360]
display_size = [800, 480]  # lcd3_5 屏幕分辨率

# 初始化 PipeLine 和 摄像头
pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode="st7701")
pl.create(sensor=Sensor(id=2, width=1280, height=720))

display_size = pl.get_display_size()
center_x, center_y = display_size[0] // 2, display_size[1] // 2  # 屏幕中心点原点 (400, 240)

# 计算上下 1/3 区域的 Y 轴边界 (画面纵向三等分，两条横线)
roi_top = display_size[1] // 3          # 上边界 (160 px)
roi_bottom = display_size[1] * 2 // 3   # 下边界 (320 px)

# 初始化 YOLO11 实例
# 将置信度阈值由 0.8 适当降至 0.65，提升边缘和动态捕捉能力
yolo = YOLO11(
    task_type="detect",
    mode="video",
    kmodel_path=kmodel_path,
    labels=labels,
    rgb888p_size=rgb888p_size,
    model_input_size=model_input_size,
    display_size=display_size,
    conf_thresh=0.65,     # 适度调低阈值防丢帧
    nms_thresh=0.45,
    max_boxes_num=1,      # 限制最多输出 1 个框
    debug_mode=0,
)
yolo.config_preprocess()

clock = time.clock()

# 【抗闪烁逻辑变量】
MAX_MISS_FRAMES = 3   # 允许连续丢失的最大帧数（可抗 3 帧以内的偶发掉帧）
miss_counter = 0      # 连续丢失计数器
last_diff_x = 0       # 保持上一帧的 X 轴偏差

# 定义 ARGB8888 格式下的 4 通道不透明颜色
COLOR_GREEN  = (255, 0, 255, 0)     # 不透明绿色
COLOR_RED    = (255, 255, 0, 0)     # 不透明红色
COLOR_YELLOW = (255, 255, 255, 0)   # 不透明黄色
COLOR_WHITE  = (255, 255, 255, 255) # 不透明白色

try:
    while True:
        clock.tick()

        # 逐帧推理
        img = pl.get_frame()
        res = yolo.run(img)

        # 当前帧是否直接识别到钢球 (且在中间 1/3 区域内)
        raw_detected = False
        if res and len(res) == 3 and len(res[0]) > 0:
            box = res[0][0]
            ball_center_y = box[1] + box[3] / 2  # 计算钢球中心 Y 坐标
            
            if roi_top <= ball_center_y <= roi_bottom:
                raw_detected = True

        # 【丢帧缓冲与迟滞处理】
        if raw_detected:
            miss_counter = 0  # 重新捕获到目标，重置丢失计数器
            
            # 仅保留置信度最高的第 1 个框
            res = [res[0][:1], res[1][:1], res[2][:1]]
            box = res[0][0]
            ball_center_x = box[0] + box[2] / 2
            
            # 计算 X 轴偏差 (钢球中心X - 屏幕中心原点400)
            current_diff_x = int(ball_center_x - center_x)
            
            # 位置滤波：70% 当前帧 + 30% 上一帧，防止数值剧烈跳动
            diff_x = int(0.7 * current_diff_x + 0.3 * last_diff_x)
            last_diff_x = diff_x
            
            detected = True
            draw_res = res
        else:
            miss_counter += 1
            # 在许可的丢失缓冲帧数内 (<=3 帧)，保持识别有效，维持上一次坐标
            if miss_counter <= MAX_MISS_FRAMES:
                detected = True
                diff_x = last_diff_x
                draw_res = [[], [], []]  # 画面框可隐去，但逻辑和串口保持在线
            else:
                detected = False
                diff_x = 0
                draw_res = [[], [], []]

        # 根据抗闪烁后的判定状态决定串口与显示
        if detected:
            status_text = "BALL: YES"
            diff_text = "X_DIFF: %+d px" % diff_x
            uart_frame = f"X{diff_x}Y0Z1E\r\n"
        else:
            status_text = "BALL: NO"
            diff_text = "X_DIFF: N/A"
            uart_frame = "X0Y0Z0E\r\n"

        # 1. 串口2 发送协议数据包
        uart2.write(uart_frame.encode('utf-8'))

        # 2. 绘制 YOLO 检测框
        yolo.draw_result(draw_res, pl.osd_img)
        
        # 3. 绘制屏幕中心红色十字线 (原点)
        pl.osd_img.draw_cross(center_x, center_y, color=COLOR_RED, size=25, thickness=2)

        # 4. 绘制中间 1/3 区域的上下两条边界横线 (白色细线)
        pl.osd_img.draw_line(0, roi_top, display_size[0], roi_top, color=COLOR_WHITE, thickness=1)
        pl.osd_img.draw_line(0, roi_bottom, display_size[0], roi_bottom, color=COLOR_WHITE, thickness=1)

        # 5. 在 OSD 左上角绘制屏幕文字
        if detected:
            pl.osd_img.draw_string(10, 10, status_text, color=COLOR_GREEN, scale=2)   # 绿色状态
            pl.osd_img.draw_string(10, 40, diff_text, color=COLOR_YELLOW, scale=2)   # 黄色偏差值
        else:
            pl.osd_img.draw_string(10, 10, status_text, color=COLOR_RED, scale=2)     # 红色状态
            pl.osd_img.draw_string(10, 40, diff_text, color=COLOR_WHITE, scale=2)   # 白色未检测

        pl.show_image()
        gc.collect()

        print(status_text, diff_text, "UART2 Sent:", uart_frame.strip(), "FPS:", clock.fps())

finally:
    # 退出时安全释放资源
    uart2.deinit()
    yolo.deinit()
    pl.destroy()
