import torch

# Import function to make predictions on images
from predictions import pred_and_plot_image

class Cocobolo():
    
    def __init__(self) -> None:
        print("================")
        # Setup custom image path
        self.IMAGE = "/home/thechancher/ViT/images/image.jpg"
        # self.CLASS_NAMES_raw = {"1":"1-Aspidosperma polyneuron", "2":"2-Araucaria angustifolia", "3":"3-Tabebuia sp.", "4":"4-Cordia goeldiana", "5":"5-Cordia sp.", "6":"6-Hura crepitans", "7":"7-Acrocarpus fraxinifolius", "8":"8-Hymenaea sp.", "9":"9-Peltogyne sp.", "10":"10-Hymenolobium petraeum", "11":"11-Myroxylon balsamum", "12":"12-Dipteryx sp.", "13":"13-Machaerium sp.", "14":"14-Bowdichia sp.", "15":"15-Mimosa scabrella", "16":"16-Cedrelinga catenaeformis", "17":"17-Goupia glabra", "18":"18-Ocotea porosa", "19":"19-Mezilaurus itauba", "20":"20-Laurus nobilis", "21":"21-Bertholethia excelsa", "22":"22-Cariniana estrellensis", "23":"23-Couratari sp.", "24":"24-Carapa guianensis", "25":"25-Cedrela fissilis", "26":"26-Melia azedarach", "27":"27-Swietenia macrophylla", "28":"28-Brosimum paraense", "29":"29-Bagassa guianensis", "30":"30-Virola surinamensis", "31":"31-Eucalyptus sp.", "32":"32-Pinus sp.", "33":"33-Podocarpus lambertii", "35":"35-Balfourodendron riedelianum", "36":"36-Euxylophora paraensis", "37":"37-Micropholis venulosa", "38":"38-Pouteria pachycarpa", "39":"39-Manilkara huberi", "40":"40-Erisma uncinatum", "41":"41-Vochysia sp."}
        self.CLASS_NAMES_raw = ["0", "1-Aspidosperma polyneuron", "2-Araucaria angustifolia", "3-Tabebuia sp.", "4-Cordia goeldiana", "5-Cordia sp.", "6-Hura crepitans", "7-Acrocarpus fraxinifolius", "8-Hymenaea sp.", "9-Peltogyne sp.", "10-Hymenolobium petraeum", "11-Myroxylon balsamum", "12-Dipteryx sp.", "13-Machaerium sp.", "14-Bowdichia sp.", "15-Mimosa scabrella", "16-Cedrelinga catenaeformis", "17-Goupia glabra", "18-Ocotea porosa", "19-Mezilaurus itauba", "20-Laurus nobilis", "21-Bertholethia excelsa", "22-Cariniana estrellensis", "23-Couratari sp.", "24-Carapa guianensis", "25-Cedrela fissilis", "26-Melia azedarach", "27-Swietenia macrophylla", "28-Brosimum paraense", "29-Bagassa guianensis", "30-Virola surinamensis", "31-Eucalyptus sp.", "32-Pinus sp.", "33-Podocarpus lambertii", "35-Balfourodendron riedelianum", "36-Euxylophora paraensis", "37-Micropholis venulosa", "38-Pouteria pachycarpa", "39-Manilkara huberi", "40-Erisma uncinatum", "41-Vochysia sp."]
        self.CLASS_NAMES = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '3', '4', '5', '6', '7', '8', '9']
        
        self.MODEL = torch.load("/home/thechancher/ViT/models/model.pth")
        self.MODEL.eval()
        
    def getClassNames(self):
        return self.CLASS_NAMES_raw
        
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
                            class_names=self.CLASS_NAMES,
                            class_labels=self.CLASS_NAMES_raw)