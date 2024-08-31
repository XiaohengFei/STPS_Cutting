import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from sklearn.mixture import GaussianMixture
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GMMAnalyzer:
    def __init__(self, n_components=6):
        """Initialising the GMM analyser"""
        logger.info("Initializing GMM analyzer with %d components.", n_components)
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full')

    def preprocess_image(self, segment):
        """Image preprocessing, histogram equalisation using CLAHE"""
        # Conversion to Lab colour space
        lab_segment = cv2.cvtColor(segment, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab_segment)
        
        # Applying CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # merge channel
        lab_clahe = cv2.merge((l_clahe, a, b))
        return lab_clahe

    def segment_colors(self, segment):
        """Splitting colour regions using GMM"""
        lab_clahe = self.preprocess_image(segment)
        
        # Convert images to 2D data points
        pixels = lab_clahe.reshape(-1, 3)

        # Fitting using a GMM model
        self.gmm.fit(pixels)

        # Predict the class of each pixel
        labels = self.gmm.predict(pixels)

        # Reconstruction of the image into individual colour regions by means of labels
        segmented_image = labels.reshape(segment.shape[0], segment.shape[1])

        # Post-processing step: morphological manipulation to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        segmented_image = cv2.morphologyEx(segmented_image.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        segmented_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)

        return segmented_image

    def apply_canny_edge_detection(self, segment):
        """Applying Canny Edge Detection"""
        # Convert to greyscale
        gray_segment = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)

        # Dynamically Adjust Canny Edge Detection Thresholds
        v = np.median(gray_segment)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))

        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_segment, (5, 5), 0)

        # Canny Edge Detection
        edges = cv2.Canny(blurred, lower, upper)

        return edges

    def find_cut_point(self, segment):
        """Finding individual cut points"""
        # Converting images to Lab colour space
        lab_segment = self.preprocess_image(segment)
        
        # Convert images to 2D data points
        pixels = lab_segment.reshape(-1, 3)

        # Fitting using a GMM model
        self.gmm.fit(pixels)

        # Choose the centre of a Gaussian distribution as the cut point
        means = self.gmm.means_
        weights = self.gmm.weights_

        # Use weights to select the centre point of maximum impact
        max_index = np.argmax(weights)
        cut_point_mean = means[max_index]

        # Convert mean values to image coordinates
        height, width = segment.shape[:2]
        cut_point_x = int(cut_point_mean[1] * width / 255.0)
        cut_point_y = int(cut_point_mean[0] * height / 255.0)

        logger.info("Cut point position: (%d, %d)", cut_point_x, cut_point_y)
        return (cut_point_x, cut_point_y)

    def visualize_segmentation(self, segment, segmented_image, edges, cut_point):
        """Visualisation of segmentation results, edge detection and cut points"""
        # Create a random colour table for visualising different categories
        unique_labels = np.unique(segmented_image)
        colors = np.random.randint(0, 255, size=(len(unique_labels), 3))

        # Create an empty image for visualisation
        colored_segment = np.zeros_like(segment)

        for label, color in zip(unique_labels, colors):
            colored_segment[segmented_image == label] = color

        # Overlay edge detection results
        colored_segment[edges > 0] = [0, 0, 255]  # 将边缘显示为红色

        # Plotting cut points
        cv2.circle(colored_segment, cut_point, 5, (0, 255, 0), -1)
        cv2.putText(colored_segment, f"Cut Point: ({cut_point[0]}, {cut_point[1]})", (cut_point[0] + 10, cut_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return colored_segment

class RealSenseYOLO:
    def __init__(self, model_path):
        """Initialising the RealSense pipeline and YOLOv8 model"""
        logger.info("Setting up RealSense pipeline...")
        # Initialising the RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # start pipeline
        try:
            self.pipeline.start(self.config)
            logger.info("RealSense pipeline started successfully.")
        except Exception as e:
            logger.error("Error starting RealSense pipeline: %s", e)

        # Load YOLOv8 model
        try:
            model_path = os.path.abspath(model_path)
            logger.info("Loading YOLOv8 model from: %s", model_path)
            self.model = YOLO(model_path)
            logger.info("YOLOv8 model loaded successfully.")
        except Exception as e:
            logger.error("Error loading YOLOv8 model: %s", e)

    def run(self, process_callback):
        """Detect and process each frame in real time"""
        try:
            while True:
                logger.info("Waiting for frames...")
                # Waiting for a frame of data
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                # Confirmation that the frame has been captured correctly
                if not color_frame:
                    logger.warning("No frame captured, skipping...")
                    continue

                # Converting images to numpy arrays
                frame = np.asanyarray(color_frame.get_data())
                logger.info("Frame captured.")

                if not hasattr(self, 'model'):
                    logger.warning("Model not loaded correctly, skipping frame processing...")
                    continue

                # Prediction using the YOLOv8 model
                results = self.model(frame, conf=0.5)  # Setting the confidence threshold to 50 per cent
                logger.info("YOLOv8 detection completed.")

                # Extract the checkbox information for ‘string’.
                string_boxes = self.extract_string_boxes(results)
                logger.info("Detected %d string boxes.", len(string_boxes))

                # Call the processing callback function for further processing
                process_callback(frame, string_boxes)

                # Press the ‘q’ key to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Exiting on user command...")
                    break

        except Exception as e:
            logger.error("An error occurred during frame processing: %s", e)
        finally:
            # Stop the pipe flow
            self.pipeline.stop()
            # Close window
            cv2.destroyAllWindows()

    def extract_string_boxes(self, results):
        """Extract the checkbox information for the ‘string’ category"""
        string_boxes = []
        for result in results[0].boxes:
            # Converting a tensor to a scalar using the .item() method
            class_id = result.cls.item()
            confidence = result.conf.item()
            logger.info("Detected class: %f, confidence: %f", class_id, confidence)

            if class_id == 0:
                # Convert bounding box coordinates to scalars
                coords = result.xyxy.cpu().numpy().flatten()
                if len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    string_boxes.append((x1, y1, x2, y2))

        return string_boxes

def process_frame(frame, string_boxes):
    """Processes each image frame, performs colour segmentation and displays cut points"""
    logger.info("Processing frame...")
    # Initialising the GMM analyser
    gmm_analyzer = GMMAnalyzer(n_components=6)  # Increasing the number of clusters to improve segmentation accuracy

    # Initialise the merged image (a copy of the original image)
    combined_frame = frame.copy()

    for box in string_boxes:
        x1, y1, x2, y2 = map(int, box)

        # Crop the detected area
        segment = frame[y1:y2, x1:x2]

        # Perform colour segmentation
        segmented_image = gmm_analyzer.segment_colors(segment)

        # Applying Canny Edge Detection
        edges = gmm_analyzer.apply_canny_edge_detection(segment)

        # Find the centre of the colour area as the cutting point
        cut_point = gmm_analyzer.find_cut_point(segment)

        # Visualisation of segmentation results, edge detection and cut points
        colored_segment = gmm_analyzer.visualize_segmentation(segment, segmented_image, edges, cut_point)

        # Putting the segmented region back into the merged image
        combined_frame[y1:y2, x1:x2] = colored_segment

        # Adding cut point information to the merged image
        cv2.putText(combined_frame, f"Cut Point: ({cut_point[0]}, {cut_point[1]})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Stitching the original and merged images together for display
    result_image = cv2.hconcat([frame, combined_frame])

    # Displaying the merged image
    cv2.imshow('Original and Segmented View with Cut Point', result_image)

def main():
    """Main function, initialises the detector and starts detection"""
    logger.info("Initializing YOLOv8 and RealSense...")
    model_path = os.path.abspath("E:\\Project\\STPS\\yolomodel\\best.pt")
    detector = RealSenseYOLO(model_path=model_path)
    
    logger.info("Starting detection...")
    detector.run(process_frame)

if __name__ == "__main__":
    logger.info("Starting main program...")
    main()
