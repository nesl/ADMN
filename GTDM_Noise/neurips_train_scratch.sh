# Train Naive Scratch Models
python3 train.py --dir_name 6_Layer_Naive_Scratch --vision_vit_layers 3 --depth_vit_layers 3 --learning_rate 1e-5 --max_layerdrop 0.0 --num_epochs 200 --from_scratch
python3 train.py --dir_name 8_Layer_Naive_Scratch --vision_vit_layers 4 --depth_vit_layers 4 --learning_rate 1e-5 --max_layerdrop 0.0 --num_epochs 200 --from_scratch
python3 train.py --dir_name 12_Layer_Naive_Scratch --vision_vit_layers 6 --depth_vit_layers 6 --learning_rate 1e-5 --max_layerdrop 0.0 --num_epochs 200 --from_scratch
python3 train.py --dir_name 16_Layer_Naive_Scratch --vision_vit_layers 8 --depth_vit_layers 8 --learning_rate 1e-5 --max_layerdrop 0.0 --num_epochs 200 --from_scratch

# Image Only
python3 train.py --dir_name 6_Layer_Image --vision_vit_layers 6 --valid_mods zed_camera_left --learning_rate 1e-5 --max_layerdrop 0.0 --num_epochs 200 --from_scratch
python3 train.py --dir_name 8_Layer_Image --vision_vit_layers 8 --valid_mods zed_camera_left --learning_rate 1e-5 --max_layerdrop 0.0 --num_epochs 200 --from_scratch
python3 train.py --dir_name 12_Layer_Image --vision_vit_layers 12 --valid_mods zed_camera_left --learning_rate 1e-5 --max_layerdrop 0.0 --num_epochs 200 --from_scratch
python3 train.py --dir_name 16_Layer_Image --vision_vit_layers 16 --valid_mods zed_camera_left --learning_rate 1e-5 --max_layerdrop 0.0 --num_epochs 200 --from_scratch

# Depth Only
python3 train.py --dir_name 6_Layer_Depth --valid_mods realsense_camera_depth --depth_vit_layers 6 --learning_rate 1e-5 --max_layerdrop 0.0 --num_epochs 200 --from_scratch
python3 train.py --dir_name 8_Layer_Depth --valid_mods realsense_camera_depth --depth_vit_layers 8 --learning_rate 1e-5 --max_layerdrop 0.0 --num_epochs 200 --from_scratch
python3 train.py --dir_name 12_Layer_Depth --valid_mods realsense_camera_depth --depth_vit_layers 12 --learning_rate 1e-5 --max_layerdrop 0.0 --num_epochs 200 --from_scratch
python3 train.py --dir_name 16_Layer_Depth --valid_mods realsense_camera_depth --depth_vit_layers 16 --learning_rate 1e-5 --max_layerdrop 0.0 --num_epochs 200 --from_scratch
