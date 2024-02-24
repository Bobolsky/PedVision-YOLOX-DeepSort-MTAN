import argparse, cv2, imutils, json, os, torch
from PIL import Image
from torchvision import transforms
from tracker import Tracker

from utils.visualize import vis_track

from PAR.models.mtan_net import MTANet
from PAR.utils.utils import MaxQueue

script_dir = os.path.dirname(os.path.abspath(__file__))

class HandleVideo():
    """
    This class encapsulates the logic for processing a video using YOLOX object detection and DeepSort tracking.
    It tracks people, records their attributes, and monitors their movement in specified regions of interest (ROIs).
        
    Attributes:
        - video (str):              Path to the video file.
        - configuration (str):      Path to the JSON configuration file specifying ROIs.
        - results (str):            Path to store the results in JSON format.
        - device (torch.device):    Device for running the Vilt model on CPU or GPU.
        - model (MTANet):           MTANet model for attribute prediction.
        - people (dict):            Dictionary to store information about tracked individuals.
        - roi1 (dict):              Dictionary for tracking ROI1 information, including counters and timers.
                                    Used to update the People's "roi1_passages" and "roi1_persistence_time".
        - roi2 (dict):              Dictionary for tracking ROI2 information, including counters and timers.
                                    Used to update the People's "roi2_passages" and "roi2_persistence_time".
    """

    def __init__(self, video: str, configuration: str, results: str) -> None:
        """
        Initialize the HandleVideo class.

        Arguments:
            - video (str):          Path to the video file.
            - configuration (str):  Path to the JSON configuration file specifying ROIs.
            - results (str):        Path to store the results in JSON format.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video = video
        self.configuration = configuration
        self.results = results
        self.people = {}
        self.roi1 = {
            "conf_data" : dict(),
            "counter" : 0,
            "id_list": set(),
            "timer" : dict(),
            "total_passages": 0,
        }
        self.roi2 = {
            "conf_data" : dict(),
            "counter" : 0,
            "id_list": set(),
            "timer" : dict(),
            "total_passages": 0,
        }
        
        self.model = MTANet().to(self.device)
        model_path = os.path.join(script_dir, "par_model.pth")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
    def track_cap(self) -> dict:
        """
        Main method to process the video, perform object detection and tracking, and collect results.

        Returns:
            dict: Dictionary containing information about tracked individuals.
        """
        # Load configuration data from a file.
        with open(self.configuration) as f:
            conf_data = json.load(f)
        # Store configuration data for each ROI.
        self.roi1["conf_data"] = conf_data["roi1"]
        self.roi2["conf_data"] = conf_data["roi2"]

        # Initialize video capture with the specified video file.
        cap = cv2.VideoCapture(self.video)
        # Initialize the Tracker with a filter to track only persons.
        tracker = Tracker(filter_class=['person'])

        # Get frames per second from the video file to calculate the skip interval.
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame = 0
        skip_interval = max(1, int(fps / 10))
        
        # Flag to ensure ROIs are preprocessed only once.
        roi_preprocessed = False
        while True:
            # Read a frame from the video.
            _, im = cap.read()
            if im is None:
                break
            
            # Skip processing for certain frames based on the skip interval.
            frame+=1
            if frame % skip_interval != 0:
                continue

            # Copy the frame for processing.
            copiedIm = im.copy()
            # Update tracker with the current frame and get bounding boxes for detected objects.
            image, bboxes = tracker.update(im)

            # Preprocess ROIs once based on the first frame's dimensions.
            if not roi_preprocessed:
                height, width, _ = im.shape
                self.roi1["conf_data"] = self.__preprocess_roi(self.roi1["conf_data"], width, height)
                self.roi2["conf_data"] = self.__preprocess_roi(self.roi2["conf_data"], width, height)
                roi_preprocessed = True

            # Draw rectangles around the ROIs on the frame.
            cv2.rectangle(im, (self.roi1["conf_data"]['x'], self.roi1["conf_data"]['y'] + self.roi1["conf_data"]['height']), (self.roi1["conf_data"]['x'] + self.roi1["conf_data"]['width'], self.roi1["conf_data"]['y']), (0, 0, 0), 3)
            cv2.rectangle(im, (self.roi2["conf_data"]['x'], self.roi2["conf_data"]['y'] + self.roi2["conf_data"]['height']), (self.roi2["conf_data"]['x'] + self.roi2["conf_data"]['width'], self.roi2["conf_data"]['y']), (0, 0, 0), 3)
            # Label the ROIs for visual identification.
            cv2.putText(img = image, text = "1", org = (self.roi1["conf_data"]['x'] + 10, self.roi1["conf_data"]['y'] + 36), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1.0, color = (0, 0, 0), thickness = 3)
            cv2.putText(img = image, text = "2", org = (self.roi2["conf_data"]['x'] + 10, self.roi2["conf_data"]['y'] + 36), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1.0, color = (0, 0, 0), thickness = 3)

            # Reset counters for each ROI.
            self.roi1["counter"] = 0
            self.roi2["counter"] = 0
            # List to keep track of IDs seen in the current frame.
            id_list = []
            
            # Process each detected bounding box.
            for bbox in bboxes:
                # Bounding box coordinates.
                x,y,w,h = bbox[:4]
                # Extracted ID for the detected person.
                id = int(bbox[4])
                id_list.append(id)
                if id not in self.people:
                    self.__populate(id)

                # Calculate the center of the bounding box.
                bbox_center = (int(((abs(w - x)) / 2) + x), int(y + ((abs(h - y)) / 2)))
                # Draw a circle at the center of the bounding box to visualize the center point.
                cv2.circle(image, bbox_center, 3, (0, 255, 0))

                # Extract the image portion within the bounding box for further processing.
                image_to_process = copiedIm[y:h, x:w, :]
                # Update pedestrian attributes for the current detected ID using the extracted image portion.
                self.__update_par(image_to_process, id)

                # Retrieve and display pedestrian attributes on the GUI.
                pedestrian_attributes = self.__par_processing(id)
                # Draw a white rectangle to serve as a background for text display below each bounding box.
                cv2.rectangle(im, (x - 20, h), (w + 30, h + 30), (255, 255, 255), -1)
                for i, t in enumerate(pedestrian_attributes):
                    cv2.putText(img = image,text = t, org = (x - 10, (h + 10) + i * 10), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.3, color = (0, 0, 0), thickness = 1)
                
                # Handle entry and exit logic for regions of interest (ROIs) for the current bounding box.
                self.__handle_roi(roi=self.roi1, bbox_center=bbox_center, id=id, frame=frame, fps=fps, number_roi=1, out_roi=False, last_frame=False)
                self.__handle_roi(roi=self.roi2, bbox_center=bbox_center, id=id, frame=frame, fps=fps, number_roi=2, out_roi=False, last_frame=False)

                # Visualize tracking information on the image.
                image = vis_track(image, bbox, roi1=self.__is_in_roi(self.roi1["conf_data"], bbox_center), roi2=self.__is_in_roi(self.roi2["conf_data"], bbox_center))
            
            # Process IDs that were in an ROI but not detected in the current frame.
            for id in list(self.roi1["id_list"]):
                if id not in id_list:
                    self.__handle_roi(roi=self.roi1, bbox_center=bbox_center, id=id, frame=frame, fps=fps, number_roi=1, out_roi=True, last_frame=False)

            for id in list(self.roi2["id_list"]):
                if id not in id_list:
                    self.__handle_roi(roi=self.roi2, bbox_center=bbox_center, id=id, frame=frame, fps=fps, number_roi=2, out_roi=True, last_frame=False)
            
            # Update the GUI with ROI information.
            cv2.rectangle(im, (0, 0), (275, 120), (255, 255, 255), -1)
            roi_info = self.__roi_processing(bboxes=bboxes)
            for i, t in enumerate(roi_info):
                cv2.putText(img = image,text = t, org = (5, 24 + 30*i), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.7, color = (0, 0, 0), thickness = 2)

            # Display the processed image in a window.
            cv2.imshow('Group 08 - AV Contest', image)
            cv2.waitKey(1)
            # Break the loop if the window is closed.
            if cv2.getWindowProperty('Group 08 - AV Contest', cv2.WND_PROP_AUTOSIZE) < 1:
                break
        
        # Final processing for ROIs after the video ends.
        self.__handle_roi(roi=self.roi1, bbox_center=bbox_center, id=None, frame=frame, fps=fps, number_roi=1, out_roi=False, last_frame=True)
        self.__handle_roi(roi=self.roi2, bbox_center=bbox_center, id=None, frame=frame, fps=fps, number_roi=2, out_roi=False, last_frame=True)

        # Release the video capture and destroy all OpenCV windows.
        cap.release()
        cv2.destroyAllWindows()
        
        # Process and return the final attributes for all tracked individuals.
        self.final_people = {}
        for person_id, attributes in self.people.items():
            self.final_people[person_id] = {}
            for attribute, queue in attributes.items():
                if isinstance(queue, MaxQueue):
                    max_attribute = queue.get_max()
                    self.final_people[person_id][attribute] = max_attribute
                else:
                    self.final_people[person_id][attribute] = queue
        
        return self.final_people
        
    # Private Methods
    def __get_time(self, frame: int, fps: float) -> float:
        """
        Calculate the time in seconds based on the frame number and frames per second (fps).

        Arguments:
            - frame (int): Frame number.
            - fps (float): Frames per second.

        Returns:
            - float: Time in seconds.
        """
        return frame/fps
        
    def __handle_roi(self, roi: dict, bbox_center: tuple, id: int, frame: int, fps: float, number_roi: int, out_roi: bool, last_frame: bool) -> None:
        """
        Handle the logic for updating ROI counters, timers, and people attributes.

        Arguments:
            - roi (dict):          Dictionary containing ROI information.
            - bbox_center (tuple): Tuple containing the (x, y) coordinates of the bounding box center.
            - id (int):            Person's identifier (Bounding Box ID).
            - frame (int):         Current frame number.
            - fps (float):         Frames per second.
            - number_roi (int):    Number of the ROI (1 or 2).
            - out_roi (bool):      True if a person was in the roi but not detected anymore in the current frame. False, otherwise.
            - last_frame (bool):   True if processing the last frame. False, otherwise.
        """
        if out_roi is True and roi["timer"][id]["counting"] is True:
            roi["timer"][id]["counting"] = False
            roi["timer"][id]["start"] = 0
            roi["timer"][id]["stop"] = round(self.__get_time(frame, fps))
            persistence_time = int(roi["timer"][id]["stop"] - roi["timer"][id]["start"])
            self.people[id][f"roi{number_roi}_persistence_time"] += persistence_time if persistence_time > 0 else 1
            roi["id_list"].remove(id)
        elif last_frame is True:
            for k in roi['timer'].keys():
                if roi["timer"][k]["counting"] is True:
                    persistence_time = int(round(self.__get_time(frame,fps), 0) - roi["timer"][k]["start"])
                    self.people[k][f"roi{number_roi}_persistence_time"] += persistence_time if persistence_time > 0 else 1
        else:
            if self.__is_in_roi(roi["conf_data"], bbox_center) is True:
                roi["counter"] += 1
                if roi["timer"][id]["counting"] is False:
                    roi["id_list"].add(id)
                    roi["timer"][id]["start"] = round(self.__get_time(frame, fps), 0)
                    roi["timer"][id]["counting"] = True
                    self.people[id][f'roi{number_roi}_passages'] += 1
            elif roi["timer"][id]["counting"] is True:
                roi["id_list"].remove(id)
                roi["timer"][id]["counting"] = False
                roi["timer"][id]["start"] = 0
                roi["timer"][id]["stop"] = round(self.__get_time(frame, fps))
                persistence_time = int(roi["timer"][id]["stop"] - roi["timer"][id]["start"])
                self.people[id][f"roi{number_roi}_persistence_time"] += persistence_time if persistence_time > 0 else 1

    def __is_in_roi(self, roi: dict, bbox_center: dict) -> bool:
        """
        Check if the given bounding box center is within the specified ROI.

        Arguments:
            - roi (dict):         Dictionary containing ROI coordinates.
            - bbox_center (tuple): Tuple containing the (x, y) coordinates of the bounding box center.

        Returns:
            - bool: True if the bbox_center is within the ROI, False otherwise.
        """
        if(bbox_center[0] >= roi['x'] and bbox_center[0] <= (roi['width'] + roi['x']) and bbox_center[1] >= roi['y'] and bbox_center[1] <= (roi['height'] + roi['y'])):
            return True
        else:
            return False

    def __populate(self, id: int) -> None:
        """
        Initialize entries for a new person in the people dictionary and corresponding ROI timers.

        Arguments:
            - id (int): Person's identifier.
        """
        self.people[id] = {
                        "id":id,
                        "upper_color": MaxQueue(),
                        "lower_color": MaxQueue(),
                        "gender": MaxQueue(),
                        "bag": MaxQueue(),
                        "hat": MaxQueue(),
                        "roi1_passages": 0,
                        "roi1_persistence_time": 0,
                        "roi2_passages": 0,
                        "roi2_persistence_time": 0
                    }

        self.roi1["timer"][id] = {
                        "counting": False,
                        "start": 0.0,
                        "stop" : 0.0
                    }
        self.roi2["timer"][id] = {
                        "counting": False,
                        "start": 0.0,
                        "stop" : 0.0
                    }

    def __preprocess_image(self, image_np) -> torch.Tensor:
        """
        Preprocess an image for model inference. Converts the image from a NumPy array to a PyTorch tensor, applying resizing, normalization, and adding a batch dimension.

        Arguments:
            - image_np (numpy.ndarray): The image to preprocess, represented as a NumPy array.

        Returns:
            - torch.Tensor: The preprocessed image as a PyTorch tensor, ready for model input.
        """
        # Convert the numpy array to a PIL Image
        image = Image.fromarray(image_np.astype('uint8'), 'RGB')
        # Define the preprocessing pipeline
        preprocess = transforms.Compose([
            transforms.Resize((96, 288)),
            transforms.ToTensor()
        ])        
        # Apply the preprocessing pipeline
        image_tensor = preprocess(image)
        # Add a batch dimension (B x C x H x W)
        image_tensor = image_tensor.unsqueeze(0)        
        return image_tensor

    def __preprocess_roi(self, roi: dict, width: int, height: int) -> dict:
        """
        Convert ROI coordinates from relative to absolute based on the image width and height.

        Arguments:
            - roi (dict):  Dictionary containing ROI coordinates.
            - width (int): Width of the video.
            - height (int): Height of the video.

        Returns:
            - dict: Updated ROI coordinates.
        """
        roi['x'] = int(roi['x'] * width)
        roi['width'] = int(roi['width'] * width)
        roi['y'] = int(roi['y']*height)
        roi['height'] = int(roi['height'] * height)
        return roi

    def __par_processing(self, id: int) -> list:
        """
        Processes the PAR (Pedestrian Attribute Recognition) attributes for an identified individual, generating a list of strings describing the recognized attributes (gender, upper and lower garment colors, presence of bag or hat).

        Arguments:
            - id (int): The identifier of the individual whose attributes are being processed.

        Returns:
            - list: A list of strings that describe the individual's attributes.
        """
        if self.people[id]['bag'].get_max() == False and self.people[id]['hat'].get_max() == False:
            bagText = "No Bag "
            hatText = "No Hat"
        elif self.people[id]['bag'].get_max() == True and self.people[id]['hat'].get_max() == False:
            bagText = "Bag"
            hatText = ""
        elif self.people[id]['bag'].get_max() == False and self.people[id]['hat'].get_max() == True:
            bagText = ""
            hatText = "Hat"
        elif self.people[id]['bag'].get_max() == True and self.people[id]['hat'].get_max() == True:
            bagText = "Bag "
            hatText = "Hat"
        return [f"Gender: {self.people[id]['gender'].get_max()}", f"{bagText}{hatText}", f"U-L: {self.people[id]['upper_color'].get_max()} - {self.people[id]['lower_color'].get_max()}"]

    def __roi_processing(self, bboxes: list) -> list:
        """
        Processes information related to Regions of Interest (ROIs) based on current bounding boxes, calculating the total number of people in ROIs and the total number of passages through each ROI.

        Arguments:
            - bboxes (list): A list of detected bounding boxes in the current frame.

        Returns:
            - list: A list of strings that describe the count of people in ROIs and the total passages.
        """
        counter = f"People in ROI: {self.roi1['counter'] + self.roi2['counter']}"
        total_person = f"Total persons: {len(bboxes)}"

        self.roi1['total_passages'] = 0
        self.roi2['total_passages'] = 0
        for person in self.people.values():
            self.roi1['total_passages'] += person["roi1_passages"]
            self.roi2['total_passages'] += person["roi2_passages"]
        
        passages_roi1 = f"Passages in ROI 1: {self.roi1['total_passages']}"
        passages_roi2 = f"Passages in ROI 2: {self.roi2['total_passages']}"
        
        return [counter, total_person, passages_roi1, passages_roi2]
    
    def __update_par(self, bbox: tuple, id : int) -> None:
        """
        Updates the PAR (Pedestrian Attribute Recognition) attributes for an individual based on the image corresponding to their bounding box. Performs inference using the MTANet model and updates the individual's attribute data in the `people` dictionary.

        Arguments:
            - bbox (tuple): The bounding box of the individual to process, containing coordinates (x, y, w, h).
            - id (int): The identifier of the individual.
        """
        self.model.eval()
        predictions = {}
        image_tensor = self.__preprocess_image(bbox)
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            for task, output in outputs.items():
                output = output.cpu()
                if len(output.shape) > 1 and output.shape[1] > 1:  # Multi-class classification
                    predictions[task] = int(torch.argmax(output, dim=1).item()) + 1
                else:  # Binary classification
                    predictions[task] = int((torch.sigmoid(output) > 0.5).item())
                    
        self.people[id]["upper_color"].add(MaxQueue.int_to_color(predictions["upper_color"]), predictions["upper_color"])
        self.people[id]["lower_color"].add(MaxQueue.int_to_color(predictions["lower_color"]), predictions["lower_color"])
        self.people[id]["gender"].add(MaxQueue.int_to_gender(predictions["gender"]), predictions["gender"])
        self.people[id]["bag"].add(MaxQueue.int_to_bag(predictions["bag"]), predictions["bag"])
        self.people[id]["hat"].add(MaxQueue.int_to_hat(predictions["hat"]), predictions["hat"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser("YOLOX-Tracker Demo!")
    parser.add_argument('-p', "--video", type=str, default='test.mp4', help="Choose a video")
    parser.add_argument("--configuration", type=str, help = "Specify ROIs")
    parser.add_argument("--results", type=str, help = "Specify where to store results")
    args = parser.parse_args()
    handler = HandleVideo(args.video, args.configuration, args.results)
    results = handler.track_cap()
    results = {"people": list(results.values())}

    with open(args.results, 'w') as json_file:
        json.dump(results, json_file, indent=2)