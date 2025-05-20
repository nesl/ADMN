### Training Stage 1 Model

To train the base stage 1 model, utilize the train.py function to train a multimodal network with 12 layer image and depth encoders loaded with MAE pretrained weights
```
python3 train.py
``` 

To train the Naive Scratch models, utilize the train.py function but pass in the desired number of layers for vision and depth (e.g., 8 layers each)
This will bypass the loading of the pretrained network and will not freeze any layers
```
python3 train.py --vision_vit_layers 8 --depth_vit_layers 8
``` 


After we train this model, it will save the checkpoint (trained for 400 epochs)


### Training Stage 2 Controller Model

We call 
```
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=continuous --discretization_method admn
``` 
to perform training with a specified number of layers


### Baselines:

**We set the batch size to be equal to one since every sample for ADMN would undergo a different distribution of layers**

**Naive Alloc**: We simply take the trained Stage 1 Model, and then split the budget 50/50:
Example: 
``` 
python3 batch_test.py --folder ./logs/Noise_Tests/LD02/AWGN_LD_Progressive_02_Epoch400 --checkpoint last.pt --test_type continuous --drop_layers_img 2 4 5 6 7 9 --drop_layers_depth 2 4 5 6 7 9 --batch_size 1 
``` 
represents naive alloc. of 12 layers

**Naive Scratch**: We train Naive Scratch models from scratch by specifying vision_vit_layers and depth_vit_layers. We test them through batch_test_scratch.py 
Example: 
```
 python3 batch_test_scratch.py --folder ./logs/3_Layers_Scratch_200Epoch/ --checkpoint last.pt --vit_layers_img 3 --vit_layers_depth 3 --test_type continuous --batch_size 1 
 ``` 
 
 We need to specify the number of layers so we can initialize the GTDM_Early model properly



**ADMN**: We test the ADML model with batch_test_controller.py
Example: 
```
python3 batch_test_controller.py --folder ./logs/Controller_continuous_Layer_6_Seed_100/ --total_layers 6 --batch_size 1
```

This writes data to the test_loss.txt file where we can look at the average distance away from the ground truth