### Training Stage 1 Model

To train the base stage 1 model, utilize the train.py function to train a multimodal network with 12 layer image and depth encoders loaded with MAE pretrained weights
```python3 train.py``` 

To train the Naive Scratch models, utilize the train.py function but pass in the desired number of layers for vision and depth (e.g., 8 layers each)
This will bypass the loading of the pretrained network and will not freeze any layers
```python3 train.py --vision_vit_layers 8 --depth_vit_layers 8``` 


After we train this model, it will save the checkpoint (trained for 400 epochs)


### Training Stage 2 Controller Model

We call ```python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=continuous --discretization_method admn``` to perform training with a specified number of layerse