# Code Notes for Blur Dataset
**Checkpoints and Results** ./logs

**ADMN**
- stage1 model: train.py --> results: ./logs/stage1_model.pt
- stage2: train_controller.py --> results: ./logs/Controller_continuous...
- test: batch_test_controller.py

**ADMN-AE**
- AutoEncoder: train_AE.py --> results: ./logs/AE_Blur
- ADMN-AE controller training: train_AE_controller.py --> results: ./logs/AE_Controller...
- test: batch_test_controller.py

**AdaMML**
- subnet training:train_adamml_subnet.py --> results: ./logs/Depth... ./logs/Vision... ./logs/...Naive
- AdaMML controller training: train_adamml_controller.py --> results: ./logs/AdaMML_Selector...
- test: batch_test_adamml_controller.py

**utils-visualization**
batch_show_image_blur.py

