from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
import gc, torch, tqdm, os

script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.join(script_dir, "data\\training_set") 
training_annotation_file = os.path.join(script_dir, "annotation\\old_training_set.txt") 

# Map
color_to_int_map = {
    "black": 1, 
    "blue": 2, 
    "brown": 3,
    "gray": 4, 
    "green": 5, 
    "orange": 6, 
    "pink": 7, 
    "purple": 8, 
    "red": 9, 
    "white": 10, 
    "yellow": 11,
    "N/A": -1,
}

gender_to_int_map = {
    'male': 0,
    'female': 1,
    'N/A': -1,
}

bag_to_int_map = {
    'no': 0,
    'yes': 1,
    'N/A': -1,
}

hat_to_int_map = {
    'no': 0,
    'yes': 1,
    'N/A': -1,
}

int_to_color_map = {value: key for key, value in color_to_int_map.items()}
int_to_gender_map = {value: key for key, value in gender_to_int_map.items()}
int_to_bag_map = {value: key for key, value in bag_to_int_map.items()}
int_to_hat_map = {value: key for key, value in hat_to_int_map.items()}

class VQA:
    
    def __init__(self) -> None:
        # Carica il modello e il processore
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(self.device)

    def vqa_inference(self, image, question):
        # Process the input image and question to prepare for the model
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        # Get the outputs from the model
        outputs = self.model(**inputs)
        # Extract the logits from the outputs
        logits = outputs.logits
        # Apply the softmax function to convert logits into probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        # Find the index of the highest probability answer
        idx = logits.argmax(-1).item()
        # Retrieve the probability associated with the predicted answer
        predicted_probability = probabilities[0, idx].item()
        # Map the index to the corresponding answer
        predicted_answer = self.model.config.id2label[idx]
        # Return the predicted answer
        return predicted_answer, predicted_probability

    def read_annotation(self, file):
        # Reading Training Annotations
        with open(file, "r") as file:
            annotations = file.readlines()
        return annotations

    def update_dataset(self, file_path):
        annotations = self.read_annotation(training_annotation_file)
        # Updating Training Annotations
        with open(os.path.join(script_dir, file_path), "w") as file:
            
            for annotation in tqdm.tqdm(annotations):
                parts = annotation.strip().split(',')
                try:
                    image_path = os.path.join(training_dir, parts[0])
                    raw_image = Image.open(image_path).convert('RGB')
                except FileNotFoundError:
                    print(f"{parts[0]} not found!")
                    continue
                
                if parts[1] == "-1":
                    answer, _ = self.vqa_inference(raw_image, "Are there any shirts?")
                    if answer.lower() == "yes":
                        predicted_color, _ = self.vqa_inference(raw_image, "What color is the person's shirt?")
                        # print(f"file: {parts[0]} -> upper color = {predicted_color}")
                        try:
                            predicted_color = "gray" if predicted_color == "grey" else predicted_color
                            if " and " in predicted_color:
                                color_splits = predicted_color.strip().split("and")
                                predicted_color = color_splits[0]
                            parts[1] = str(color_to_int_map[predicted_color])
                        except KeyError:                            
                            pass
                        
                if parts[2] == "-1":
                    answer, _ = self.vqa_inference(raw_image, "Are there any pants?")
                    if answer.lower() == "yes":
                        predicted_color, _ = self.vqa_inference(raw_image, "What color are the person's pants?")
                        # print(f"file: {parts[0]} -> lower color = {predicted_color}")
                        try:
                            predicted_color = "gray" if predicted_color == "grey" else predicted_color
                            if " and " in predicted_color:
                                color_splits = predicted_color.strip().split("and")
                                predicted_color = color_splits[0]
                            parts[2] = str(color_to_int_map[predicted_color])
                        except KeyError:
                            pass

                if parts[3] == "-1":
                    answer, _ = self.vqa_inference(raw_image, "Is the person male or female?")
                    try:
                        parts[3] = str(gender_to_int_map[answer.lower()])
                        # print(f"file: {parts[0]} -> gender = {parts[3]}")
                    except KeyError:
                        pass
                        
                if parts[4] == "-1":
                    answer1, predicted_probability1 = self.vqa_inference(raw_image, "Is the person carrying a bag?")
                    answer2, predicted_probability2 = self.vqa_inference(raw_image, "Is the person carrying a knapsack?")
                    answer3, predicted_probability3 = self.vqa_inference(raw_image, "Is the person carrying a suitcase?")
                    answer4, predicted_probability4 = self.vqa_inference(raw_image, "Is the person carrying a purse?")
                    try:
                        bag1 = str(bag_to_int_map[answer1.lower()])
                        bag2 = str(bag_to_int_map[answer2.lower()])
                        bag3 = str(bag_to_int_map[answer3.lower()])
                        bag4 = str(bag_to_int_map[answer4.lower()])
                        if (bag1 == "1" and predicted_probability1 > 0.80) or (bag2 == "1" and predicted_probability2 > 0.96) or (bag3 == "1" and predicted_probability3 > 0.80) or (bag4 == "1" and predicted_probability4 > 0.80):
                            parts[4] = "1"
                        else:
                            parts[4] = "0"                
                        # print(f"file: {parts[0]} -> bag = {parts[4]}")
                    except KeyError:
                        pass

                if parts[5] == "-1":
                    answer, _ = self.vqa_inference(raw_image, "Is the person wearing a hat?")
                    try:
                        parts[5] = str(hat_to_int_map[answer.lower()])
                        # print(f"file: {parts[0]} -> hat = {parts[5]}")
                    except KeyError:
                        pass

                data = ','.join(parts)
                file.write(data + "\n")
        
    def release_memory(self):
        # Free GPU memory by deleting the model and empty the CUDA cache.
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        
    def dataset_bar_chart(self, file_path):
        upper_values = {key: 0 for key in color_to_int_map.keys()}
        lower_values = {key: 0 for key in color_to_int_map.keys()}
        gender_values = {key: 0 for key in gender_to_int_map.keys()}
        bag_values = {key: 0 for key in bag_to_int_map.keys()}
        hat_values = {key: 0 for key in hat_to_int_map.keys()}

        annotations = vqa.read_annotation(os.path.join(script_dir, file_path))

        for annotation in annotations:
            parts = annotation.strip().split(',')

            upper_values[str(int_to_color_map[int(parts[1])])] += 1
            lower_values[str(int_to_color_map[int(parts[2])])] += 1
            gender_values[str(int_to_gender_map[int(parts[3])])] += 1
            bag_values[str(int_to_bag_map[int(parts[4])])] += 1
            hat_values[str(int_to_hat_map[int(parts[5])])] += 1

        # Creazione e visualizzazione di tutti e cinque i grafici
        self.__create_bar_chart('Upper Body Clothing Colors', upper_values, 'navy')
        self.__create_bar_chart('Lower Body Clothing Colors', lower_values, 'green')
        self.__create_bar_chart('Gender Distribution', gender_values, 'purple')
        self.__create_bar_chart('Bag Presence', bag_values, 'red')
        self.__create_bar_chart('Hat Presence', hat_values, 'blue')
    
    def __create_bar_chart(self, title, data, color):
        plt.figure(figsize=(10, 5))
        bars = plt.bar(data.keys(), data.values(), color=color)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, yval, ha='center', va='bottom')
        plt.xlabel('Categories')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.show()
    
import matplotlib.pyplot as plt

vqa = VQA()
# vqa.update_dataset()
vqa.dataset_bar_chart("annotation\\training_set.txt")
# vqa.dataset_bar_chart("training2.txt")
vqa.release_memory()