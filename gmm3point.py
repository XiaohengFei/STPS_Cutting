import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from sklearn.mixture import GaussianMixture
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GMMAnalyzer:
    def __init__(self, n_components=6):
        """初始化GMM分析器"""
        logger.info("Initializing GMM analyzer with %d components.", n_components)
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full')

    def preprocess_image(self, segment):
        """图像预处理，使用CLAHE进行直方图均衡化"""
        # 转换为Lab颜色空间
        lab_segment = cv2.cvtColor(segment, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab_segment)
        
        # 应用CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # 合并通道
        lab_clahe = cv2.merge((l_clahe, a, b))
        return lab_clahe

    def segment_colors(self, segment):
        """使用GMM分割颜色区域"""
        lab_clahe = self.preprocess_image(segment)
        
        # 将图像转换为2D数据点
        pixels = lab_clahe.reshape(-1, 3)

        # 使用GMM模型进行拟合
        self.gmm.fit(pixels)

        # 预测每个像素的类别
        labels = self.gmm.predict(pixels)

        # 通过标签将图像重构为各个颜色区域
        segmented_image = labels.reshape(segment.shape[0], segment.shape[1])

        # 后处理步骤：形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        segmented_image = cv2.morphologyEx(segmented_image.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        segmented_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)

        return segmented_image

    def apply_canny_edge_detection(self, segment):
        """应用Canny边缘检测"""
        # 转换为灰度图像
        gray_segment = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)

        # 动态调整Canny边缘检测阈值
        v = np.median(gray_segment)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))

        # 高斯模糊以减少噪声
        blurred = cv2.GaussianBlur(gray_segment, (5, 5), 0)

        # Canny边缘检测
        edges = cv2.Canny(blurred, lower, upper)

        return edges

    def find_cut_points(self, segment):
        """找出多个切割点"""
        # 将图像转换为Lab颜色空间
        lab_segment = self.preprocess_image(segment)
        
        # 将图像转换为2D数据点
        pixels = lab_segment.reshape(-1, 3)

        # 使用GMM模型进行拟合
        self.gmm.fit(pixels)

        # 获取高斯分布的中心点和权重
        means = self.gmm.means_
        weights = self.gmm.weights_

        # 按权重排序，选择最重要的三个高斯中心点
        indices = np.argsort(weights)[-3:]  # 选择权重最大的三个点

        # 将均值转换为图像坐标
        height, width = segment.shape[:2]
        cut_points = [(int(means[i][1] * width / 255.0), int(means[i][0] * height / 255.0)) for i in indices]

        logger.info("Cut points positions: %s", cut_points)
        return cut_points

    def visualize_segmentation(self, segment, segmented_image, edges, cut_points):
        """可视化分割结果、边缘检测和切割点"""
        # 创建一个随机颜色表，用于可视化不同的类别
        unique_labels = np.unique(segmented_image)
        colors = np.random.randint(0, 255, size=(len(unique_labels), 3))

        # 创建一个空的图像用于可视化
        colored_segment = np.zeros_like(segment)

        for label, color in zip(unique_labels, colors):
            colored_segment[segmented_image == label] = color

        # 叠加边缘检测结果
        colored_segment[edges > 0] = [0, 0, 255]  # 将边缘显示为红色

        # 绘制切割点
        for idx, cut_point in enumerate(cut_points):
            cv2.circle(colored_segment, cut_point, 5, (0, 255, 0), -1)
            cv2.putText(colored_segment, f"Point {idx+1}: ({cut_point[0]}, {cut_point[1]})", 
                        (cut_point[0] + 10, cut_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return colored_segment

class RealSenseYOLO:
    def __init__(self, model_path):
        """初始化RealSense管道和YOLOv8模型"""
        logger.info("Setting up RealSense pipeline...")
        # 初始化RealSense管道
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # 启动管道
        try:
            self.pipeline.start(self.config)
            logger.info("RealSense pipeline started successfully.")
        except Exception as e:
            logger.error("Error starting RealSense pipeline: %s", e)

        # 加载YOLOv8模型
        try:
            model_path = os.path.abspath(model_path)
            logger.info("Loading YOLOv8 model from: %s", model_path)
            self.model = YOLO(model_path)
            logger.info("YOLOv8 model loaded successfully.")
        except Exception as e:
            logger.error("Error loading YOLOv8 model: %s", e)

    def run(self, process_callback):
        """实时检测并处理每一帧"""
        try:
            while True:
                logger.info("Waiting for frames...")
                # 等待一帧数据
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                # 确认帧已被正确捕获
                if not color_frame:
                    logger.warning("No frame captured, skipping...")
                    continue

                # 将图像转换为numpy数组
                frame = np.asanyarray(color_frame.get_data())
                logger.info("Frame captured.")

                if not hasattr(self, 'model'):
                    logger.warning("Model not loaded correctly, skipping frame processing...")
                    continue

                # 使用YOLOv8模型进行预测
                results = self.model(frame, conf=0.5)  # 设置置信度阈值为50%
                logger.info("YOLOv8 detection completed.")

                # 提取“string”的检测框信息
                string_boxes = self.extract_string_boxes(results)
                logger.info("Detected %d string boxes.", len(string_boxes))

                # 调用处理回调函数进行进一步处理
                process_callback(frame, string_boxes)

                # 按下 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Exiting on user command...")
                    break

        except Exception as e:
            logger.error("An error occurred during frame processing: %s", e)
        finally:
            # 停止管道流
            self.pipeline.stop()
            # 关闭窗口
            cv2.destroyAllWindows()

    def extract_string_boxes(self, results):
        """提取‘string’类别的检测框信息"""
        string_boxes = []
        for result in results[0].boxes:
            # 使用 .item() 方法将 tensor 转换为标量
            class_id = result.cls.item()
            confidence = result.conf.item()
            logger.info("Detected class: %f, confidence: %f", class_id, confidence)

            # 假设类别 'string' 的ID是0
            if class_id == 0:
                # 将边界框坐标转换为标量
                coords = result.xyxy.cpu().numpy().flatten()
                if len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    string_boxes.append((x1, y1, x2, y2))

        return string_boxes

def process_frame(frame, string_boxes):
    """处理每一帧图像，进行颜色分割并显示切割点"""
    logger.info("Processing frame...")
    # 初始化GMM分析器
    gmm_analyzer = GMMAnalyzer(n_components=6)  # 增加聚类数量以提高分割精度

    # 初始化合并图像（原始图像的副本）
    combined_frame = frame.copy()

    for box in string_boxes:
        x1, y1, x2, y2 = map(int, box)  # 将坐标转换为整数

        # 裁剪检测到的区域
        segment = frame[y1:y2, x1:x2]

        # 进行颜色分割
        segmented_image = gmm_analyzer.segment_colors(segment)

        # 应用Canny边缘检测
        edges = gmm_analyzer.apply_canny_edge_detection(segment)

        # 找到颜色区域的中心点作为切割点
        cut_points = gmm_analyzer.find_cut_points(segment)

        # 可视化分割结果、边缘检测和切割点
        colored_segment = gmm_analyzer.visualize_segmentation(segment, segmented_image, edges, cut_points)

        # 将分割后的区域放回合并图像
        combined_frame[y1:y2, x1:x2] = colored_segment

        # 在合并图像上添加切割点信息
        for idx, cut_point in enumerate(cut_points):
            cv2.putText(combined_frame, f"Point {idx+1}: ({cut_point[0]}, {cut_point[1]})", (10, 50 + idx*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # 将原始图像和合并图像拼接到一起显示
    result_image = cv2.hconcat([frame, combined_frame])

    # 显示合并后的图像
    cv2.imshow('Original and Segmented View with Cut Points', result_image)

def main():
    """主函数，初始化检测器并开始检测"""
    logger.info("Initializing YOLOv8 and RealSense...")
    model_path = os.path.abspath("E:\\Project\\STPS\\best.pt")
    detector = RealSenseYOLO(model_path=model_path)
    
    logger.info("Starting detection...")
    detector.run(process_frame)

if __name__ == "__main__":
    logger.info("Starting main program...")
    main()
