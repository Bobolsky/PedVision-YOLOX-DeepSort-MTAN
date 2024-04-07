import torch
from torchvision import transforms
from PIL import Image

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt

from models.mtan_net import *
from utils.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((96, 288)),
        transforms.ToTensor()
    ])
    image_path = os.path.join(script_dir, "test", image_path)
    image = Image.open(image_path).convert('RGB')
    return preprocess(image).unsqueeze(0)

def test_model_on_image(model, image_tensor):
    model.eval()
    predictions = {}
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        for task, output in outputs.items():
            output = output.cpu()
            print(output)
            if len(output.shape) > 1 and output.shape[1] > 1:  # Multi-class classification
                predictions[task] = int(torch.argmax(output, dim=1).item()) + 1
            else:  # Binary classification
                print(torch.sigmoid(output))
                predictions[task] = int((torch.sigmoid(output) > 0.5).item())
    return predictions

# def draw_activation_map(model, image_path, device):
#     input_tensor = preprocess_image(image_path).to(device)
#     model = model.to(device)
#     model.eval()

#     # Assumendo che l'output sia un dizionario di tensori per ogni task
#     with torch.no_grad():
#         output = model(input_tensor)

#     for task, output_tensor in output.items():
#         # Creare un'istanza del cam extractor per il task specifico
#         cam_extractor = SmoothGradCAMpp(model)
#         # Calcolare la mappa di attivazione per la classe più probabile
#         class_idx = output_tensor.squeeze(0).argmax().item()
#         activation_map = cam_extractor(class_idx, output_tensor)
        
#         # Sovrapporre la mappa di attivazione all'immagine originale
#         result = overlay_mask(Image.open(image_path), to_pil_image(activation_map[0].squeeze(0), mode="F"), alpha=0.5)
#         plt.imshow(result)
#         plt.axis("off")
#         plt.title(f"Activation Map for Task: {task}")
#         plt.savefig(f"{task}_activation_map.png")
        
def test(image_path):
    model = MTANet().to(device)

    # Carica lo stato del modello salvato
    model_path = os.path.join(script_dir, "par_model.pth")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    input_tensor = load_and_preprocess_image(image_path)
    predictions = test_model_on_image(model, input_tensor)

    verbose(predictions)

def verbose(predictions):
    print(
        f"Predictions: "
        f"upper_color: {int_to_color_map[predictions['upper_color']]}, "
        f"lower_color: {int_to_color_map[predictions['lower_color']]}, "
        f"gender: {int_to_gender_map[predictions['gender']]}, "
        f"bag: {int_to_bag_map[predictions['bag']]}, "
        f"hat: {int_to_hat_map[predictions['hat']]}"
    )  

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('--file_path', dest="path")
    (options, args) = parser.parse_args()
    
    test(options.path)
    
# Epoch:   0%|                                                           | 0/20 [00:00<?, ?it/s]
#         EPOCH (0) --> TRAINING LOSS: 0.2044,TRAINING ACCURACY: 0.7801,
#                 upper body loss: 0.1382,lower body loss: 0.0971,gender loss: 0.2660,bag loss: 0.3514,hat loss: 0.1694
#                 upper body acc: 0.5935,lower body acc: 0.7180,gender acc: 0.8520,bag acc: 0.8183,hat acc: 0.9187
#         EPOCH (0) --> VALIDATION LOSS: 0.3839,VALIDATION ACCURACY: 0.7545,
#                 upper body loss: 0.2672,lower body loss: 0.1480,gender loss: 0.7667,bag loss: 0.6459,hat loss: 0.0917
#                 upper body acc: 0.5227,lower Body acc: 0.7341,gender acc: 0.6976,bag acc: 0.8412,hat acc: 0.9771
#         ...best model saved with validation loss: 0.3839
# Epoch:   5%|██▍                                             | 1/20 [05:32<1:45:15, 332.38s/it]
#         EPOCH (1) --> TRAINING LOSS: 0.1277,TRAINING ACCURACY: 0.8612,
#                 upper body loss: 0.0950,lower body loss: 0.0640,gender loss: 0.1396,bag loss: 0.2475,hat loss: 0.0924
#         EPOCH (1) --> VALIDATION LOSS: 0.2160,VALIDATION ACCURACY: 0.7841,
#                 upper body loss: 0.1795,lower body loss: 0.1187,gender loss: 0.3473,bag loss: 0.3502,hat loss: 0.0842
#                 upper body acc: 0.5594,lower Body acc: 0.7527,gender acc: 0.7874,bag acc: 0.8431,hat acc: 0.9778
#         ...best model saved with validation loss: 0.2160
# Epoch:  10%|████▊                                           | 2/20 [10:43<1:35:52, 319.61s/it]
# Training:  18%|██████████████████████████▍                                                                                                                           | 225/1278 [00:55<03:21,  5.21it/s] 
#         EPOCH (2) --> TRAINING LOSS: 0.1143,TRAINING ACCURACY: 0.8733,
#                 upper body loss: 0.0881,lower body loss: 0.0599,gender loss: 0.1254,bag loss: 0.2147,hat loss: 0.0835
#                 upper body acc: 0.7536,lower body acc: 0.8266,gender acc: 0.9366,bag acc: 0.8906,hat acc: 0.9592
#         EPOCH (2) --> VALIDATION LOSS: 0.1924,VALIDATION ACCURACY: 0.7998,
#                 upper body loss: 0.1623,lower body loss: 0.1110,gender loss: 0.2926,bag loss: 0.3169,hat loss: 0.0795
#                 upper body acc: 0.5945,lower Body acc: 0.7574,gender acc: 0.8276,bag acc: 0.8421,hat acc: 0.9771
#         ...best model saved with validation loss: 0.1924
# Epoch:  15%|███████▏                                        | 3/20 [15:53<1:29:25, 315.63s/it]
# Training:   9%|█████████████▎                                                                                                                                        | 113/1278 [00:31<03:31,  5.50it/s] 
#         EPOCH (3) --> TRAINING LOSS: 0.1115,TRAINING ACCURACY: 0.8761,                                                                                                                                   
#                 upper body loss: 0.0873,lower body loss: 0.0588,gender loss: 0.1202,bag loss: 0.2088,hat loss: 0.0826
#                 upper body acc: 0.7565,lower body acc: 0.8293,gender acc: 0.9394,bag acc: 0.8953,hat acc: 0.9600
#         EPOCH (3) --> VALIDATION LOSS: 0.1737,VALIDATION ACCURACY: 0.7871,
#                 upper body loss: 0.1584,lower body loss: 0.1125,gender loss: 0.2133,bag loss: 0.3052,hat loss: 0.0789
#                 upper body acc: 0.5422,lower Body acc: 0.6783,gender acc: 0.8883,bag acc: 0.8504,hat acc: 0.9763
#         ...best model saved with validation loss: 0.1737
# Epoch:  20%|█████████▌                                      | 4/20 [20:54<1:22:36, 309.76s/it]
#         EPOCH (4) --> TRAINING LOSS: 0.1069,TRAINING ACCURACY: 0.8800,
#                 upper body loss: 0.0851,lower body loss: 0.0576,gender loss: 0.1138,bag loss: 0.2000,hat loss: 0.0782
#                 upper body acc: 0.7618,lower body acc: 0.8342,gender acc: 0.9439,bag acc: 0.8981,hat acc: 0.9618
#         EPOCH (4) --> VALIDATION LOSS: 0.1868,VALIDATION ACCURACY: 0.8254,
#                 upper body loss: 0.1409,lower body loss: 0.0863,gender loss: 0.2150,bag loss: 0.3865,hat loss: 0.1053
#                 upper body acc: 0.6477,lower Body acc: 0.7734,gender acc: 0.8785,bag acc: 0.8471,hat acc: 0.9800
# Epoch:  25%|████████████                                    | 5/20 [25:54<1:16:30, 306.01s/it]
#         EPOCH (5) --> TRAINING LOSS: 0.1008,TRAINING ACCURACY: 0.8831,                                                                                                                                   
#                 upper body loss: 0.0836,lower body loss: 0.0571,gender loss: 0.1047,bag loss: 0.1849,hat loss: 0.0737
#                 upper body acc: 0.7653,lower body acc: 0.8331,gender acc: 0.9480,bag acc: 0.9049,hat acc: 0.9641
#         EPOCH (5) --> VALIDATION LOSS: 0.1525,VALIDATION ACCURACY: 0.8498,
#                 upper body loss: 0.1227,lower body loss: 0.0798,gender loss: 0.2100,bag loss: 0.2825,hat loss: 0.0675
#                 upper body acc: 0.7073,lower Body acc: 0.7880,gender acc: 0.8961,bag acc: 0.8805,hat acc: 0.9770
#         ...best model saved with validation loss: 0.1525
# Epoch:  30%|██████████████▍                                 | 6/20 [30:15<1:07:51, 290.79s/it]
#         EPOCH (6) --> TRAINING LOSS: 0.1641,TRAINING ACCURACY: 0.8354,                                                                                                                                   
#                 upper body loss: 0.1071,lower body loss: 0.0712,gender loss: 0.2325,bag loss: 0.2621,hat loss: 0.1475
#                 upper body acc: 0.7008,lower body acc: 0.7972,gender acc: 0.8810,bag acc: 0.8636,hat acc: 0.9344
#         EPOCH (6) --> VALIDATION LOSS: 0.2190,VALIDATION ACCURACY: 0.7827,
#                 upper body loss: 0.1309,lower body loss: 0.0808,gender loss: 0.4397,bag loss: 0.3121,hat loss: 0.1315
#                 upper body acc: 0.6097,lower Body acc: 0.8000,gender acc: 0.7083,bag acc: 0.8373,hat acc: 0.9580
# Epoch:  35%|████████████████▊                               | 7/20 [34:49<1:01:50, 285.42s/it]
#         EPOCH (7) --> TRAINING LOSS: 0.2066,TRAINING ACCURACY: 0.7912,                                                                                                                                   
#                 upper body loss: 0.1230,lower body loss: 0.0856,gender loss: 0.3048,bag loss: 0.3178,hat loss: 0.2016
#                 upper body acc: 0.6516,lower body acc: 0.7573,gender acc: 0.8234,bag acc: 0.8253,hat acc: 0.8984
#         EPOCH (7) --> VALIDATION LOSS: 0.2055,VALIDATION ACCURACY: 0.7991,
#                 upper body loss: 0.1260,lower body loss: 0.0866,gender loss: 0.3914,bag loss: 0.3108,hat loss: 0.1127
#                 upper body acc: 0.6641,lower Body acc: 0.7855,gender acc: 0.7573,bag acc: 0.8208,hat acc: 0.9678
# Epoch:  40%|████████████████████                              | 8/20 [39:09<55:26, 277.23s/it]
#         EPOCH (8) --> TRAINING LOSS: 0.2223,TRAINING ACCURACY: 0.7775,                                                                                                                                   
#                 upper body loss: 0.1283,lower body loss: 0.0934,gender loss: 0.3155,bag loss: 0.3405,hat loss: 0.2337
#                 upper body acc: 0.6373,lower body acc: 0.7277,gender acc: 0.8164,bag acc: 0.8100,hat acc: 0.8964
#         EPOCH (8) --> VALIDATION LOSS: 0.2446,VALIDATION ACCURACY: 0.7803,
#                 upper body loss: 0.1255,lower body loss: 0.0899,gender loss: 0.4626,bag loss: 0.3257,hat loss: 0.2191
#                 upper body acc: 0.6634,lower Body acc: 0.7405,gender acc: 0.7186,bag acc: 0.8392,hat acc: 0.9400
# Epoch:  45%|██████████████████████▌                           | 9/20 [43:27<49:45, 271.38s/it]
#         EPOCH (9) --> TRAINING LOSS: 0.2030,TRAINING ACCURACY: 0.7962,                                                                                                                                   
#                 upper body loss: 0.1216,lower body loss: 0.0851,gender loss: 0.2823,bag loss: 0.3282,hat loss: 0.1976
#                 upper body acc: 0.6540,lower body acc: 0.7513,gender acc: 0.8491,bag acc: 0.8217,hat acc: 0.9049
#         EPOCH (9) --> VALIDATION LOSS: 0.2616,VALIDATION ACCURACY: 0.7757,
#                 upper body loss: 0.1344,lower body loss: 0.0845,gender loss: 0.5557,bag loss: 0.3660,hat loss: 0.1676
#                 upper body acc: 0.6269,lower Body acc: 0.7968,gender acc: 0.7092,bag acc: 0.8428,hat acc: 0.9029
# Epoch:  50%|████████████████████████▌                        | 10/20 [47:46<44:34, 267.47s/it]
#         EPOCH (10) --> TRAINING LOSS: 0.2152,TRAINING ACCURACY: 0.7835,                                                                                                                                  
#                 upper body loss: 0.1260,lower body loss: 0.0885,gender loss: 0.3221,bag loss: 0.3412,hat loss: 0.1984
#                 upper body acc: 0.6477,lower body acc: 0.7422,gender acc: 0.8135,bag acc: 0.8175,hat acc: 0.8967
#         EPOCH (10) --> VALIDATION LOSS: 0.2175,VALIDATION ACCURACY: 0.7842,
#                 upper body loss: 0.1345,lower body loss: 0.0866,gender loss: 0.3931,bag loss: 0.3171,hat loss: 0.1562
#                 upper body acc: 0.6366,lower Body acc: 0.7815,gender acc: 0.7538,bag acc: 0.8456,hat acc: 0.9033
# Epoch:  55%|██████████████████████████▉                      | 11/20 [52:05<39:42, 264.72s/it]
#         EPOCH (11) --> TRAINING LOSS: 0.1932,TRAINING ACCURACY: 0.8066,                                                                                                                                  
#                 upper body loss: 0.1182,lower body loss: 0.0822,gender loss: 0.2608,bag loss: 0.3201,hat loss: 0.1848
#                 upper body acc: 0.6679,lower body acc: 0.7609,gender acc: 0.8598,bag acc: 0.8303,hat acc: 0.9138
#         EPOCH (11) --> VALIDATION LOSS: 0.2113,VALIDATION ACCURACY: 0.7907,
#                 upper body loss: 0.1334,lower body loss: 0.0874,gender loss: 0.3866,bag loss: 0.3231,hat loss: 0.1262
#                 upper body acc: 0.6263,lower Body acc: 0.7945,gender acc: 0.7294,bag acc: 0.8555,hat acc: 0.9477
# Early stopping triggered



# Train Loader size: 1278 | Train Loader dataset size: 81737
# Validation Loader size: 191 | Validation Loader dataset size: 12161
# Epoch:   0%|                                                                                                                                                                     | 0/20 [00:00<?, ?it/s]
#         EPOCH (0) --> TRAINING LOSS: 0.1108, TRAINING ACCURACY: 0.7805,
#                 upper body loss: 0.1042, lower body loss: 0.0783, gender loss: 0.1359, bag loss: 0.1801, hat loss: 0.0557
#                 upper body acc: 0.6476, lower body acc: 0.7313, gender acc: 0.8356, bag acc: 0.8067, hat acc: 0.8812
#         EPOCH (0) --> VALIDATION LOSS: 0.4846, VALIDATION ACCURACY: 0.5568,
#                 upper body loss: 0.7775, lower body loss: 0.3647, gender loss: 0.4059, bag loss: 0.8066, hat loss: 0.0681
#                 upper body acc: 0.1916, lower Body acc: 0.5711, gender acc: 0.6964, bag acc: 0.3495, hat acc: 0.9756
#         ...best model saved with validation loss: 0.4846
# Epoch:   5%|███████▋                                                                                                                                                  | 1/20 [05:34<1:46:03, 334.94s/it]
#         EPOCH (1) --> TRAINING LOSS: 0.0762, TRAINING ACCURACY: 0.8556,
#                 upper body loss: 0.0747, lower body loss: 0.0524, gender loss: 0.0785, bag loss: 0.1374, hat loss: 0.0379
#                 upper body acc: 0.7511, lower body acc: 0.8192, gender acc: 0.9197, bag acc: 0.8595, hat acc: 0.9285
#         EPOCH (1) --> VALIDATION LOSS: 0.2091, VALIDATION ACCURACY: 0.7271,
#                 upper body loss: 0.2983, lower body loss: 0.2594, gender loss: 0.1739, bag loss: 0.2724, hat loss: 0.0413
#                 upper body acc: 0.5106, lower Body acc: 0.5981, gender acc: 0.8789, bag acc: 0.6722, hat acc: 0.9760
#         ...best model saved with validation loss: 0.2091
# Epoch:  10%|███████████████▍                                                                                                                                          | 2/20 [10:57<1:38:21, 327.88s/it] 
#         EPOCH (2) --> TRAINING LOSS: 0.0696, TRAINING ACCURACY: 0.8662,                                                                                                                                  
#                 upper body loss: 0.0708, lower body loss: 0.0490, gender loss: 0.0708, bag loss: 0.1224, hat loss: 0.0352
#                 upper body acc: 0.7653, lower body acc: 0.8311, gender acc: 0.9280, bag acc: 0.8750, hat acc: 0.9319
#         EPOCH (2) --> VALIDATION LOSS: 0.1217, VALIDATION ACCURACY: 0.7888,
#                 upper body loss: 0.1246, lower body loss: 0.1272, gender loss: 0.1844, bag loss: 0.1356, hat loss: 0.0369
#                 upper body acc: 0.5851, lower Body acc: 0.6893, gender acc: 0.9049, bag acc: 0.8419, hat acc: 0.9231
#         ...best model saved with validation loss: 0.1217
# Epoch:  15%|███████████████████████                                                                                                                                   | 3/20 [16:31<1:33:36, 330.36s/it] 
#         EPOCH (3) --> TRAINING LOSS: 0.0690, TRAINING ACCURACY: 0.8679,                                                                                                                                  
#                 upper body loss: 0.0712, lower body loss: 0.0485, gender loss: 0.0691, bag loss: 0.1204, hat loss: 0.0359
#                 upper body acc: 0.7669, lower body acc: 0.8328, gender acc: 0.9309, bag acc: 0.8798, hat acc: 0.9293
#         EPOCH (3) --> VALIDATION LOSS: 0.1058, VALIDATION ACCURACY: 0.8116,
#                 upper body loss: 0.1180, lower body loss: 0.0826, gender loss: 0.1335, bag loss: 0.1663, hat loss: 0.0285
#                 upper body acc: 0.6092, lower Body acc: 0.7595, gender acc: 0.9020, bag acc: 0.8084, hat acc: 0.9789
#         ...best model saved with validation loss: 0.1058
# Epoch:  20%|██████████████████████████████▊                                                                                                                           | 4/20 [22:04<1:28:26, 331.65s/it] 
#         EPOCH (4) --> TRAINING LOSS: 0.0686, TRAINING ACCURACY: 0.8679,                                                                                                                                  
#                 upper body loss: 0.0719, lower body loss: 0.0486, gender loss: 0.0681, bag loss: 0.1186, hat loss: 0.0356
#                 upper body acc: 0.7661, lower body acc: 0.8325, gender acc: 0.9315, bag acc: 0.8799, hat acc: 0.9295
#         EPOCH (4) --> VALIDATION LOSS: 0.1169, VALIDATION ACCURACY: 0.7733,
#                 upper body loss: 0.1071, lower body loss: 0.0786, gender loss: 0.1693, bag loss: 0.1836, hat loss: 0.0460
#                 upper body acc: 0.6726, lower Body acc: 0.7537, gender acc: 0.8566, bag acc: 0.7290, hat acc: 0.8545
# Epoch:  25%|██████████████████████████████████████▌                                                                                                                   | 5/20 [27:57<1:24:48, 339.21s/it] 
#         EPOCH (5) --> TRAINING LOSS: 0.0669, TRAINING ACCURACY: 0.8712,                                                                                                                                  
#                 upper body loss: 0.0713, lower body loss: 0.0485, gender loss: 0.0646, bag loss: 0.1148, hat loss: 0.0354
#                 upper body acc: 0.7684, lower body acc: 0.8336, gender acc: 0.9351, bag acc: 0.8875, hat acc: 0.9312
#         EPOCH (5) --> VALIDATION LOSS: 0.1092, VALIDATION ACCURACY: 0.8085,
#                 upper body loss: 0.1151, lower body loss: 0.0932, gender loss: 0.1332, bag loss: 0.1656, hat loss: 0.0389
#                 upper body acc: 0.6823, lower Body acc: 0.7152, gender acc: 0.8452, bag acc: 0.8204, hat acc: 0.9793
# Epoch:  30%|██████████████████████████████████████████████▏                                                                                                           | 6/20 [33:20<1:17:50, 333.61s/it] 
#         EPOCH (6) --> TRAINING LOSS: 0.0604, TRAINING ACCURACY: 0.8818,                                                                                                                                  
#                 upper body loss: 0.0672, lower body loss: 0.0457, gender loss: 0.0571, bag loss: 0.1004, hat loss: 0.0318
#                 upper body acc: 0.7821, lower body acc: 0.8423, gender acc: 0.9433, bag acc: 0.9012, hat acc: 0.9399
#         EPOCH (6) --> VALIDATION LOSS: 0.1234, VALIDATION ACCURACY: 0.8025,
#                 upper body loss: 0.1170, lower body loss: 0.0871, gender loss: 0.2316, bag loss: 0.1477, hat loss: 0.0334
#                 upper body acc: 0.6903, lower Body acc: 0.7233, gender acc: 0.7860, bag acc: 0.8759, hat acc: 0.9367
# Epoch:  35%|█████████████████████████████████████████████████████▉                                                                                                    | 7/20 [38:27<1:10:24, 324.99s/it] 
#         EPOCH (7) --> TRAINING LOSS: 0.0551, TRAINING ACCURACY: 0.8894,                                                                                                                                  
#                 upper body loss: 0.0640, lower body loss: 0.0439, gender loss: 0.0500, bag loss: 0.0876, hat loss: 0.0298
#                 upper body acc: 0.7923, lower body acc: 0.8455, gender acc: 0.9504, bag acc: 0.9145, hat acc: 0.9445
#         EPOCH (7) --> VALIDATION LOSS: 0.0900, VALIDATION ACCURACY: 0.8277,
#                 upper body loss: 0.0996, lower body loss: 0.0667, gender loss: 0.1023, bag loss: 0.1433, hat loss: 0.0380
#                 upper body acc: 0.7094, lower Body acc: 0.7754, gender acc: 0.8930, bag acc: 0.8611, hat acc: 0.8995
#         ...best model saved with validation loss: 0.0900
# Epoch:  40%|█████████████████████████████████████████████████████████████▌                                                                                            | 8/20 [43:33<1:03:48, 319.03s/it] 
#         EPOCH (8) --> TRAINING LOSS: 0.0472, TRAINING ACCURACY: 0.9027,                                                                                                                                  
#                 upper body loss: 0.0589, lower body loss: 0.0407, gender loss: 0.0405, bag loss: 0.0716, hat loss: 0.0242
#                 upper body acc: 0.8083, lower body acc: 0.8560, gender acc: 0.9613, bag acc: 0.9330, hat acc: 0.9548
#         EPOCH (8) --> VALIDATION LOSS: 0.0996, VALIDATION ACCURACY: 0.8392,
#                 upper body loss: 0.0916, lower body loss: 0.0617, gender loss: 0.1383, bag loss: 0.1528, hat loss: 0.0537
#                 upper body acc: 0.7033, lower Body acc: 0.7869, gender acc: 0.8624, bag acc: 0.8645, hat acc: 0.9790
# Epoch:  45%|██████████████████████████████████████████████████████████████████████▏                                                                                     | 9/20 [48:39<57:43, 314.89s/it] 
#         EPOCH (9) --> TRAINING LOSS: 0.0456, TRAINING ACCURACY: 0.9067,                                                                                                                                  
#                 upper body loss: 0.0567, lower body loss: 0.0395, gender loss: 0.0380, bag loss: 0.0663, hat loss: 0.0272
#                 upper body acc: 0.8148, lower body acc: 0.8610, gender acc: 0.9644, bag acc: 0.9375, hat acc: 0.9558
#         EPOCH (9) --> VALIDATION LOSS: 0.0894, VALIDATION ACCURACY: 0.8371,
#                 upper body loss: 0.0913, lower body loss: 0.0767, gender loss: 0.1137, bag loss: 0.1367, hat loss: 0.0284
#                 upper body acc: 0.7136, lower Body acc: 0.7817, gender acc: 0.9013, bag acc: 0.8765, hat acc: 0.9126
#         ...best model saved with validation loss: 0.0894
# Epoch:  50%|█████████████████████████████████████████████████████████████████████████████▌                                                                             | 10/20 [54:58<55:47, 334.71s/it] 
#         EPOCH (10) --> TRAINING LOSS: 0.0372, TRAINING ACCURACY: 0.9205,                                                                                                                                 
#                 upper body loss: 0.0498, lower body loss: 0.0359, gender loss: 0.0287, bag loss: 0.0515, hat loss: 0.0202
#                 upper body acc: 0.8360, lower body acc: 0.8729, gender acc: 0.9743, bag acc: 0.9539, hat acc: 0.9655
#         EPOCH (10) --> VALIDATION LOSS: 0.0933, VALIDATION ACCURACY: 0.8375,
#                 upper body loss: 0.0902, lower body loss: 0.0699, gender loss: 0.1322, bag loss: 0.1482, hat loss: 0.0260
#                 upper body acc: 0.7013, lower Body acc: 0.7874, gender acc: 0.8894, bag acc: 0.8369, hat acc: 0.9724
# Epoch:  55%|████████████████████████████████████████████████████████████████████████████████████▏                                                                    | 11/20 [1:00:11<49:12, 328.08s/it] 
#         EPOCH (11) --> TRAINING LOSS: 0.0310, TRAINING ACCURACY: 0.9317,                                                                                                                                 
#                 upper body loss: 0.0432, lower body loss: 0.0318, gender loss: 0.0239, bag loss: 0.0394, hat loss: 0.0167
#                 upper body acc: 0.8567, lower body acc: 0.8873, gender acc: 0.9788, bag acc: 0.9645, hat acc: 0.9715
#         EPOCH (11) --> VALIDATION LOSS: 0.0957, VALIDATION ACCURACY: 0.8282,
#                 upper body loss: 0.0988, lower body loss: 0.0945, gender loss: 0.1054, bag loss: 0.1525, hat loss: 0.0271
#                 upper body acc: 0.7176, lower Body acc: 0.7869, gender acc: 0.9050, bag acc: 0.7510, hat acc: 0.9804
# Epoch:  60%|███████████████████████████████████████████████████████████████████████████████████████████▊                                                             | 12/20 [1:06:09<44:56, 337.06s/it] 
#         EPOCH (12) --> TRAINING LOSS: 0.0245, TRAINING ACCURACY: 0.9454,                                                                                                                                 
#                 upper body loss: 0.0354, lower body loss: 0.0270, gender loss: 0.0178, bag loss: 0.0306, hat loss: 0.0116
#                 upper body acc: 0.8833, lower body acc: 0.9038, gender acc: 0.9849, bag acc: 0.9730, hat acc: 0.9819
#         EPOCH (12) --> VALIDATION LOSS: 0.0998, VALIDATION ACCURACY: 0.8349,
#                 upper body loss: 0.1103, lower body loss: 0.0827, gender loss: 0.1310, bag loss: 0.1473, hat loss: 0.0278
#                 upper body acc: 0.7127, lower Body acc: 0.7836, gender acc: 0.8813, bag acc: 0.8169, hat acc: 0.9800
# Epoch:  65%|███████████████████████████████████████████████████████████████████████████████████████████████████▍                                                     | 13/20 [1:11:15<38:15, 327.89s/it] 
#         EPOCH (13) --> TRAINING LOSS: 0.0182, TRAINING ACCURACY: 0.9599,                                                                                                                                 
#                 upper body loss: 0.0254, lower body loss: 0.0216, gender loss: 0.0133, bag loss: 0.0216, hat loss: 0.0093
#                 upper body acc: 0.9157, lower body acc: 0.9247, gender acc: 0.9893, bag acc: 0.9818, hat acc: 0.9883
#         EPOCH (13) --> VALIDATION LOSS: 0.1010, VALIDATION ACCURACY: 0.8297,
#                 upper body loss: 0.1183, lower body loss: 0.0888, gender loss: 0.1211, bag loss: 0.1468, hat loss: 0.0298
#                 upper body acc: 0.7074, lower Body acc: 0.7821, gender acc: 0.8940, bag acc: 0.8054, hat acc: 0.9598
# Epoch:  70%|███████████████████████████████████████████████████████████████████████████████████████████████████████████                                              | 14/20 [1:16:40<32:41, 326.88s/it] 
#         EPOCH (14) --> TRAINING LOSS: 0.0138, TRAINING ACCURACY: 0.9704,                                                                                                                                 
#                 upper body loss: 0.0181, lower body loss: 0.0162, gender loss: 0.0111, bag loss: 0.0165, hat loss: 0.0071
#                 upper body acc: 0.9415, lower body acc: 0.9437, gender acc: 0.9908, bag acc: 0.9863, hat acc: 0.9898
#         EPOCH (14) --> VALIDATION LOSS: 0.1092, VALIDATION ACCURACY: 0.8377,
#                 upper body loss: 0.1417, lower body loss: 0.1047, gender loss: 0.1342, bag loss: 0.1377, hat loss: 0.0275
#                 upper body acc: 0.7118, lower Body acc: 0.7750, gender acc: 0.8786, bag acc: 0.8501, hat acc: 0.9732
# Epoch:  75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                      | 15/20 [1:21:06<25:42, 308.42s/it] 
#         EPOCH (15) --> TRAINING LOSS: 0.0097, TRAINING ACCURACY: 0.9809,                                                                                                                                 
#                 upper body loss: 0.0118, lower body loss: 0.0110, gender loss: 0.0084, bag loss: 0.0136, hat loss: 0.0037
#                 upper body acc: 0.9626, lower body acc: 0.9639, gender acc: 0.9933, bag acc: 0.9892, hat acc: 0.9954
#         EPOCH (15) --> VALIDATION LOSS: 0.1105, VALIDATION ACCURACY: 0.8429,
#                 upper body loss: 0.1581, lower body loss: 0.1071, gender loss: 0.1300, bag loss: 0.1322, hat loss: 0.0252
#                 upper body acc: 0.7031, lower Body acc: 0.7739, gender acc: 0.8783, bag acc: 0.8797, hat acc: 0.9797
# Epoch:  80%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                              | 16/20 [1:25:29<19:39, 294.90s/it] 
#         EPOCH (16) --> TRAINING LOSS: 0.0070, TRAINING ACCURACY: 0.9877,                                                                                                                                 
#                 upper body loss: 0.0072, lower body loss: 0.0069, gender loss: 0.0073, bag loss: 0.0110, hat loss: 0.0026
#                 upper body acc: 0.9774, lower body acc: 0.9780, gender acc: 0.9945, bag acc: 0.9915, hat acc: 0.9972
#         EPOCH (16) --> VALIDATION LOSS: 0.1178, VALIDATION ACCURACY: 0.8472,
#                 upper body loss: 0.1892, lower body loss: 0.1344, gender loss: 0.1067, bag loss: 0.1308, hat loss: 0.0280
#                 upper body acc: 0.7021, lower Body acc: 0.7719, gender acc: 0.9057, bag acc: 0.8770, hat acc: 0.9794
# Epoch:  85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                       | 17/20 [1:30:41<15:00, 300.00s/it] 
#         EPOCH (17) --> TRAINING LOSS: 0.0056, TRAINING ACCURACY: 0.9911,                                                                                                                                 
#                 upper body loss: 0.0048, lower body loss: 0.0048, gender loss: 0.0066, bag loss: 0.0098, hat loss: 0.0022
#                 upper body acc: 0.9853, lower body acc: 0.9847, gender acc: 0.9950, bag acc: 0.9925, hat acc: 0.9978
#         EPOCH (17) --> VALIDATION LOSS: 0.1237, VALIDATION ACCURACY: 0.8479,
#                 upper body loss: 0.2033, lower body loss: 0.1410, gender loss: 0.1109, bag loss: 0.1330, hat loss: 0.0304
#                 upper body acc: 0.7006, lower Body acc: 0.7701, gender acc: 0.8989, bag acc: 0.8915, hat acc: 0.9785
# Early stopping triggered