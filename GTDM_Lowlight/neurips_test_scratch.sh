# Naive Scratch
python3 batch_test_scratch.py --folder ./logs/6_Layer_Naive_Scratch --vit_layers_img 3 --vit_layers_depth 3 --batch_size 1
python3 batch_test_scratch.py --folder ./logs/8_Layer_Naive_Scratch --vit_layers_img 4 --vit_layers_depth 4 --batch_size 1
python3 batch_test_scratch.py --folder ./logs/12_Layer_Naive_Scratch --vit_layers_img 6 --vit_layers_depth 6 --batch_size 1
python3 batch_test_scratch.py --folder ./logs/16_Layer_Naive_Scratch --vit_layers_img 8 --vit_layers_depth 8 --batch_size 1

# Image Only
python3 batch_test_scratch.py --folder ./logs/6_Layer_Image --vit_layers_img 6 --valid_mods zed_camera_left --batch_size 1
python3 batch_test_scratch.py --folder ./logs/8_Layer_Image --vit_layers_img 8 --valid_mods zed_camera_left --batch_size 1
python3 batch_test_scratch.py --folder ./logs/12_Layer_Image --vit_layers_img 12 --valid_mods zed_camera_left --batch_size 1
python3 batch_test_scratch.py --folder ./logs/16_Layer_Image --vit_layers_img 16 --valid_mods zed_camera_left --batch_size 1

# Depth Only
python3 batch_test_scratch.py --folder ./logs/6_Layer_Depth --vit_layers_depth 6 --valid_mods realsense_camera_depth --batch_size 1
python3 batch_test_scratch.py --folder ./logs/8_Layer_Depth --vit_layers_depth 8 --valid_mods realsense_camera_depth --batch_size 1
python3 batch_test_scratch.py --folder ./logs/12_Layer_Depth --vit_layers_depth 12 --valid_mods realsense_camera_depth --batch_size 1
python3 batch_test_scratch.py --folder ./logs/16_Layer_Depth --vit_layers_depth 16 --valid_mods realsense_camera_depth --batch_size 1