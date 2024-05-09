import torch

# Import function to make predictions on images
from predictions import pred_and_plot_image

class Cocobolo():
    
    def __init__(self) -> None:
        print("================")
        # Setup custom image path
        self.IMAGE = "/home/thechancher/ViT/images/image.jpg"
        self.CLASS_NAMES = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '3', '4', '5', '6', '7', '8', '9']

        self.MODEL = torch.load("/home/thechancher/ViT/models/model.pth")
        self.MODEL.eval()
        
    def getNames(self):
        return self.CLASS_NAMES
        
    def predict_test(self):
        # Setup custom image path
        test_dir = "/home/thechancher/ViT/images/0258.JPG"

        # Predict on custom image
        pred_and_plot_image(model=self.MODEL,
                            image_path=test_dir,
                            class_names=self.CLASS_NAMES)
    
    def predict(self):
        # Predict on custom image
        return pred_and_plot_image(model=self.MODEL,
                            image_path=self.IMAGE,
                            class_names=self.CLASS_NAMES)