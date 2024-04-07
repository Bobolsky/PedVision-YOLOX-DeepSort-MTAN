import torch, os
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.par_dataset import PARDataset
from models.mtan_net import MTANet
from models.train import *
from models.loss.ASLoss import *
from utils.utils import *

def main():

    # Initialize the Dataset
    train_dataset = PARDataset(
        data_folder=os.path.join(data_dir, "training_set"), 
        annotation_path=os.path.join(annotation_dir, "training_set.txt"), 
        augment=True
    )

    val_dataset = PARDataset(
        data_folder=os.path.join(data_dir, "validation_set"), 
        annotation_path=os.path.join(annotation_dir, "validation_set.txt"), 
        augment=False
    )

    # Initialize the DataLoader
    dataloader_params = {"batch_size": 64, "num_workers": 8, "pin_memory": True}
    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_params)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_params)

    # # Initialize the model and move it to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTANet().to(device)

    # Define loss functions for each task
    criterions = {
        'upper_color': AsymmetricLoss(
            gamma_neg=torch.tensor([1, 2, 4, 2, 3, 4, 4, 4, 2, 2, 4]), 
            gamma_pos=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            num_classes=11
        ),
        'lower_color': AsymmetricLoss(
            gamma_neg=torch.tensor([1, 2, 4, 2, 3, 5, 5, 5, 4, 4, 4]), 
            gamma_pos=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            num_classes=11
        ),
        'gender': AsymmetricLoss(gamma_neg=0, gamma_pos=2, num_classes=1),
        'bag': AsymmetricLoss(gamma_neg=0, gamma_pos=2, num_classes=1),
        'hat': AsymmetricLoss(gamma_neg=0, gamma_pos=3, num_classes=1),
    }

    # Define an optimizer and scheduler
    num_epochs = 20
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=num_epochs)

    print(f"Train Loader size: {len(train_loader)} | Train Loader dataset size: {len(train_loader.dataset)}")
    print(f"Validation Loader size: {len(val_loader)} | Validation Loader dataset size: {len(val_loader.dataset)}")

    # Initialize Trainer
    trainer = TrainNet(
        model=model,
        device=device,
        tasks=criterions.keys(),
        losses=criterions,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        max_patience=7
    )

    # Start training
    trainer.fit(train_loader=train_loader, val_loader=val_loader)

    # # Free GPU memory
    free_gpu(model=model)

if __name__ == '__main__':
    # from torchview import draw_graph
    # model_graph = draw_graph(MTANet(), input_size=(1, 3, 96, 288), expand_nested=True)
    # filepath = model_graph.visual_graph.render(directory=script_dir, format='svg')

    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
    

# Epoch:   0%|                                                                                                                                                                     | 0/20 [00:00<?, ?it/s]
#         EPOCH (0) --> TRAINING LOSS: 0.1084, TRAINING ACCURACY: 0.7755,
#                 upper body loss: 0.1161, lower body loss: 0.0840, gender loss: 0.1287, bag loss: 0.1577, hat loss: 0.0554
#                 upper body acc: 0.5983, lower body acc: 0.7053, gender acc: 0.8386, bag acc: 0.8178, hat acc: 0.9175
#         EPOCH (0) --> VALIDATION LOSS: 0.0733, VALIDATION ACCURACY: 0.8665,
#                 upper body loss: 0.0679, lower body loss: 0.0453, gender loss: 0.0872, bag loss: 0.1271, hat loss: 0.0389
#                 upper body acc: 0.7757, lower Body acc: 0.8490, gender acc: 0.9092, bag acc: 0.8600, hat acc: 0.9384
#         ...best model saved with validation loss: 0.0733
# Epoch:   5%|███████▋                                                                                                                                                  | 1/20 [04:38<1:28:17, 278.82s/it]
#         EPOCH (1) --> TRAINING LOSS: 0.0687, TRAINING ACCURACY: 0.8601,
#                 upper body loss: 0.0744, lower body loss: 0.0530, gender loss: 0.0672, bag loss: 0.1138, hat loss: 0.0349
#                 upper body acc: 0.7486, lower body acc: 0.8142, gender acc: 0.9220, bag acc: 0.8685, hat acc: 0.9475
#         EPOCH (1) --> VALIDATION LOSS: 0.0616, VALIDATION ACCURACY: 0.8833,
#                 upper body loss: 0.0611, lower body loss: 0.0391, gender loss: 0.0689, bag loss: 0.1121, hat loss: 0.0267
#                 upper body acc: 0.8024, lower Body acc: 0.8678, gender acc: 0.9176, bag acc: 0.8804, hat acc: 0.9482
#         ...best model saved with validation loss: 0.0616
# Epoch:  10%|███████████████▍                                                                                                                                          | 2/20 [09:21<1:24:22, 281.24s/it] 
#         EPOCH (2) --> TRAINING LOSS: 0.0614, TRAINING ACCURACY: 0.8737,
#                 upper body loss: 0.0685, lower body loss: 0.0481, gender loss: 0.0583, bag loss: 0.1006, hat loss: 0.0313
#                 upper body acc: 0.7678, lower body acc: 0.8305, gender acc: 0.9329, bag acc: 0.8836, hat acc: 0.9537
#         EPOCH (2) --> VALIDATION LOSS: 0.0592, VALIDATION ACCURACY: 0.8884,
#                 upper body loss: 0.0594, lower body loss: 0.0367, gender loss: 0.0743, bag loss: 0.1048, hat loss: 0.0210
#                 upper body acc: 0.8164, lower Body acc: 0.8734, gender acc: 0.9120, bag acc: 0.8879, hat acc: 0.9524
#         ...best model saved with validation loss: 0.0592
# Epoch:  15%|███████████████████████                                                                                                                                   | 3/20 [13:51<1:18:09, 275.85s/it] 
#         EPOCH (3) --> TRAINING LOSS: 0.0585, TRAINING ACCURACY: 0.8785,                                                                                                                                  
#                 upper body loss: 0.0677, lower body loss: 0.0468, gender loss: 0.0537, bag loss: 0.0950, hat loss: 0.0295
#                 upper body acc: 0.7718, lower body acc: 0.8357, gender acc: 0.9383, bag acc: 0.8924, hat acc: 0.9545
#         EPOCH (3) --> VALIDATION LOSS: 0.0655, VALIDATION ACCURACY: 0.8618,
#                 upper body loss: 0.0695, lower body loss: 0.0471, gender loss: 0.0728, bag loss: 0.1065, hat loss: 0.0316
#                 upper body acc: 0.7817, lower Body acc: 0.8109, gender acc: 0.9213, bag acc: 0.8848, hat acc: 0.9105
# Epoch:  20%|██████████████████████████████▊                                                                                                                           | 4/20 [18:21<1:13:00, 273.77s/it] 
#         EPOCH (4) --> TRAINING LOSS: 0.0568, TRAINING ACCURACY: 0.8805,                                                                                                                                  
#                 upper body loss: 0.0666, lower body loss: 0.0461, gender loss: 0.0520, bag loss: 0.0910, hat loss: 0.0285
#                 upper body acc: 0.7736, lower body acc: 0.8375, gender acc: 0.9403, bag acc: 0.8951, hat acc: 0.9560
#         EPOCH (4) --> VALIDATION LOSS: 0.0583, VALIDATION ACCURACY: 0.8798,
#                 upper body loss: 0.0604, lower body loss: 0.0402, gender loss: 0.0685, bag loss: 0.0998, hat loss: 0.0226
#                 upper body acc: 0.7889, lower Body acc: 0.8642, gender acc: 0.9211, bag acc: 0.8887, hat acc: 0.9360
#         ...best model saved with validation loss: 0.0583
# Epoch:  25%|██████████████████████████████████████▌                                                                                                                   | 5/20 [22:52<1:08:07, 272.50s/it] 
#         EPOCH (5) --> TRAINING LOSS: 0.0533, TRAINING ACCURACY: 0.8865,                                                                                                                                  
#                 upper body loss: 0.0645, lower body loss: 0.0444, gender loss: 0.0468, bag loss: 0.0848, hat loss: 0.0258
#                 upper body acc: 0.7818, lower body acc: 0.8433, gender acc: 0.9461, bag acc: 0.9021, hat acc: 0.9591
#         EPOCH (5) --> VALIDATION LOSS: 0.0562, VALIDATION ACCURACY: 0.8869,
#                 upper body loss: 0.0538, lower body loss: 0.0399, gender loss: 0.0726, bag loss: 0.0920, hat loss: 0.0229
#                 upper body acc: 0.8180, lower Body acc: 0.8615, gender acc: 0.9150, bag acc: 0.8957, hat acc: 0.9441
#         ...best model saved with validation loss: 0.0562
# Epoch:  30%|██████████████████████████████████████████████▏                                                                                                           | 6/20 [27:39<1:04:48, 277.76s/it] 
#         EPOCH (6) --> TRAINING LOSS: 0.0492, TRAINING ACCURACY: 0.8921,                                                                                                                                  
#                 upper body loss: 0.0614, lower body loss: 0.0426, gender loss: 0.0422, bag loss: 0.0762, hat loss: 0.0236
#                 upper body acc: 0.7883, lower body acc: 0.8471, gender acc: 0.9516, bag acc: 0.9117, hat acc: 0.9618
#         EPOCH (6) --> VALIDATION LOSS: 0.0570, VALIDATION ACCURACY: 0.8871,
#                 upper body loss: 0.0562, lower body loss: 0.0398, gender loss: 0.0605, bag loss: 0.1000, hat loss: 0.0287
#                 upper body acc: 0.8176, lower Body acc: 0.8673, gender acc: 0.9330, bag acc: 0.8905, hat acc: 0.9274
# Epoch:  35%|█████████████████████████████████████████████████████▉                                                                                                    | 7/20 [32:30<1:01:03, 281.81s/it] 
#         EPOCH (7) --> TRAINING LOSS: 0.0458, TRAINING ACCURACY: 0.8988,                                                                                                                                  
#                 upper body loss: 0.0586, lower body loss: 0.0408, gender loss: 0.0384, bag loss: 0.0696, hat loss: 0.0214
#                 upper body acc: 0.7979, lower body acc: 0.8531, gender acc: 0.9572, bag acc: 0.9197, hat acc: 0.9660
#         EPOCH (7) --> VALIDATION LOSS: 0.0559, VALIDATION ACCURACY: 0.8968,
#                 upper body loss: 0.0579, lower body loss: 0.0359, gender loss: 0.0630, bag loss: 0.1015, hat loss: 0.0210
#                 upper body acc: 0.8251, lower Body acc: 0.8775, gender acc: 0.9301, bag acc: 0.8940, hat acc: 0.9571
#         ...best model saved with validation loss: 0.0559
# Epoch:  40%|██████████████████████████████████████████████████████████████▍                                                                                             | 8/20 [37:21<56:57, 284.78s/it] 
#         EPOCH (8) --> TRAINING LOSS: 0.0406, TRAINING ACCURACY: 0.9080,                                                                                                                                  
#                 upper body loss: 0.0551, lower body loss: 0.0388, gender loss: 0.0310, bag loss: 0.0591, hat loss: 0.0188
#                 upper body acc: 0.8114, lower body acc: 0.8607, gender acc: 0.9653, bag acc: 0.9322, hat acc: 0.9706
#         EPOCH (8) --> VALIDATION LOSS: 0.0556, VALIDATION ACCURACY: 0.8910,
#                 upper body loss: 0.0581, lower body loss: 0.0366, gender loss: 0.0579, bag loss: 0.0945, hat loss: 0.0309
#                 upper body acc: 0.8045, lower Body acc: 0.8742, gender acc: 0.9331, bag acc: 0.8990, hat acc: 0.9444
#         ...best model saved with validation loss: 0.0556
# Epoch:  45%|██████████████████████████████████████████████████████████████████████▏                                                                                     | 9/20 [41:53<51:28, 280.73s/it] 
#         EPOCH (9) --> TRAINING LOSS: 0.0358, TRAINING ACCURACY: 0.9167,                                                                                                                                  
#                 upper body loss: 0.0507, lower body loss: 0.0362, gender loss: 0.0259, bag loss: 0.0500, hat loss: 0.0160
#                 upper body acc: 0.8250, lower body acc: 0.8681, gender acc: 0.9707, bag acc: 0.9440, hat acc: 0.9754
#         EPOCH (9) --> VALIDATION LOSS: 0.0551, VALIDATION ACCURACY: 0.8966,
#                 upper body loss: 0.0578, lower body loss: 0.0379, gender loss: 0.0571, bag loss: 0.0956, hat loss: 0.0270
#                 upper body acc: 0.8153, lower Body acc: 0.8735, gender acc: 0.9360, bag acc: 0.9013, hat acc: 0.9569
#         ...best model saved with validation loss: 0.0551
# Epoch:  50%|█████████████████████████████████████████████████████████████████████████████▌                                                                             | 10/20 [46:21<46:10, 277.07s/it] 
#         EPOCH (10) --> TRAINING LOSS: 0.0311, TRAINING ACCURACY: 0.9250,                                                                                                                                 
#                 upper body loss: 0.0465, lower body loss: 0.0334, gender loss: 0.0212, bag loss: 0.0412, hat loss: 0.0132
#                 upper body acc: 0.8372, lower body acc: 0.8771, gender acc: 0.9766, bag acc: 0.9538, hat acc: 0.9801
#         EPOCH (10) --> VALIDATION LOSS: 0.0585, VALIDATION ACCURACY: 0.8949,
#                 upper body loss: 0.0557, lower body loss: 0.0361, gender loss: 0.0692, bag loss: 0.1048, hat loss: 0.0265
#                 upper body acc: 0.8278, lower Body acc: 0.8748, gender acc: 0.9191, bag acc: 0.8967, hat acc: 0.9560
# Epoch:  55%|█████████████████████████████████████████████████████████████████████████████████████▎                                                                     | 11/20 [50:49<41:08, 274.27s/it] 
#         EPOCH (11) --> TRAINING LOSS: 0.0262, TRAINING ACCURACY: 0.9350,                                                                                                                                 
#                 upper body loss: 0.0415, lower body loss: 0.0301, gender loss: 0.0170, bag loss: 0.0319, hat loss: 0.0104
#                 upper body acc: 0.8552, lower body acc: 0.8889, gender acc: 0.9811, bag acc: 0.9648, hat acc: 0.9850
#         EPOCH (11) --> VALIDATION LOSS: 0.0600, VALIDATION ACCURACY: 0.8962,
#                 upper body loss: 0.0594, lower body loss: 0.0385, gender loss: 0.0619, bag loss: 0.1151, hat loss: 0.0251
#                 upper body acc: 0.8078, lower Body acc: 0.8795, gender acc: 0.9336, bag acc: 0.8988, hat acc: 0.9613
# Epoch:  60%|█████████████████████████████████████████████████████████████████████████████████████████████                                                              | 12/20 [55:17<36:16, 272.09s/it] 
#         EPOCH (12) --> TRAINING LOSS: 0.0215, TRAINING ACCURACY: 0.9452,                                                                                                                                 
#                 upper body loss: 0.0355, lower body loss: 0.0265, gender loss: 0.0133, bag loss: 0.0243, hat loss: 0.0080
#                 upper body acc: 0.8752, lower body acc: 0.9034, gender acc: 0.9853, bag acc: 0.9737, hat acc: 0.9886
#         EPOCH (12) --> VALIDATION LOSS: 0.0616, VALIDATION ACCURACY: 0.8966,
#                 upper body loss: 0.0593, lower body loss: 0.0404, gender loss: 0.0609, bag loss: 0.1142, hat loss: 0.0332
#                 upper body acc: 0.8153, lower Body acc: 0.8734, gender acc: 0.9354, bag acc: 0.9005, hat acc: 0.9584
# Epoch:  65%|████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                      | 13/20 [59:43<31:33, 270.44s/it] 
#         EPOCH (13) --> TRAINING LOSS: 0.0175, TRAINING ACCURACY: 0.9538,                                                                                                                                 
#                 upper body loss: 0.0298, lower body loss: 0.0229, gender loss: 0.0102, bag loss: 0.0183, hat loss: 0.0061
#                 upper body acc: 0.8929, lower body acc: 0.9157, gender acc: 0.9889, bag acc: 0.9803, hat acc: 0.9913
#         EPOCH (13) --> VALIDATION LOSS: 0.0645, VALIDATION ACCURACY: 0.8929,
#                 upper body loss: 0.0666, lower body loss: 0.0448, gender loss: 0.0612, bag loss: 0.1097, hat loss: 0.0403
#                 upper body acc: 0.8087, lower Body acc: 0.8634, gender acc: 0.9382, bag acc: 0.8996, hat acc: 0.9543
# Epoch:  70%|███████████████████████████████████████████████████████████████████████████████████████████████████████████                                              | 14/20 [1:04:10<26:55, 269.25s/it] 
#         EPOCH (14) --> TRAINING LOSS: 0.0133, TRAINING ACCURACY: 0.9643,                                                                                                                                 
#                 upper body loss: 0.0237, lower body loss: 0.0187, gender loss: 0.0076, bag loss: 0.0127, hat loss: 0.0040
#                 upper body acc: 0.9159, lower body acc: 0.9319, gender acc: 0.9921, bag acc: 0.9868, hat acc: 0.9946
#         EPOCH (14) --> VALIDATION LOSS: 0.0667, VALIDATION ACCURACY: 0.8965,
#                 upper body loss: 0.0735, lower body loss: 0.0495, gender loss: 0.0690, bag loss: 0.1057, hat loss: 0.0360
#                 upper body acc: 0.8101, lower Body acc: 0.8737, gender acc: 0.9387, bag acc: 0.9024, hat acc: 0.9575
# Epoch:  75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                      | 15/20 [1:08:36<22:21, 268.23s/it] 
#         EPOCH (15) --> TRAINING LOSS: 0.0101, TRAINING ACCURACY: 0.9728,                                                                                                                                 
#                 upper body loss: 0.0185, lower body loss: 0.0149, gender loss: 0.0059, bag loss: 0.0088, hat loss: 0.0026
#                 upper body acc: 0.9359, lower body acc: 0.9473, gender acc: 0.9937, bag acc: 0.9910, hat acc: 0.9963
#         EPOCH (15) --> VALIDATION LOSS: 0.0685, VALIDATION ACCURACY: 0.8964,
# Epoch:  80%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                              | 16/20 [1:13:02<17:51, 267.79s/it] 
#         EPOCH (16) --> TRAINING LOSS: 0.0078, TRAINING ACCURACY: 0.9792,
#                 upper body loss: 0.0139, lower body loss: 0.0117, gender loss: 0.0050, bag loss: 0.0065, hat loss: 0.0017
#                 upper body acc: 0.9512, lower body acc: 0.9592, gender acc: 0.9946, bag acc: 0.9935, hat acc: 0.9977
#         EPOCH (16) --> VALIDATION LOSS: 0.0733, VALIDATION ACCURACY: 0.8948,
#                 upper body loss: 0.0887, lower body loss: 0.0578, gender loss: 0.0638, bag loss: 0.1174, hat loss: 0.0387
#                 upper body acc: 0.8084, lower Body acc: 0.8689, gender acc: 0.9381, bag acc: 0.8989, hat acc: 0.9595
# Epoch:  85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                       | 17/20 [1:17:29<13:22, 267.36s/it] 
#         EPOCH (17) --> TRAINING LOSS: 0.0060, TRAINING ACCURACY: 0.9843,
#                 upper body loss: 0.0107, lower body loss: 0.0092, gender loss: 0.0042, bag loss: 0.0047, hat loss: 0.0012
#                 upper body acc: 0.9631, lower body acc: 0.9690, gender acc: 0.9955, bag acc: 0.9954, hat acc: 0.9985
#         EPOCH (17) --> VALIDATION LOSS: 0.0770, VALIDATION ACCURACY: 0.8965,
#                 upper body loss: 0.1005, lower body loss: 0.0661, gender loss: 0.0652, bag loss: 0.1160, hat loss: 0.0373
#                 upper body acc: 0.8135, lower Body acc: 0.8681, gender acc: 0.9395, bag acc: 0.9004, hat acc: 0.9612
# Early stopping triggered