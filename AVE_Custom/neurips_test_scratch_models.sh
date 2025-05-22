# Train Naive Scratch Models
python3 batch_test_scratch.py --vit_layers_img 3 --vit_layers_audio 3  --batch_size 1 --folder ./logs/6_Layer_Naive_Scratch/
python3 batch_test_scratch.py --vit_layers_img 4 --vit_layers_audio 4  --batch_size 1 --folder ./logs/8_Layer_Naive_Scratch/
python3 batch_test_scratch.py --vit_layers_img 6 --vit_layers_audio 6  --batch_size 1 --folder ./logs/12_Layer_Naive_Scratch/
python3 batch_test_scratch.py --vit_layers_img 8 --vit_layers_audio 8  --batch_size 1 --folder ./logs/16_Layer_Naive_Scratch/

# Unimodal Image Models
python3 batch_test_scratch.py --valid_mods image --vit_layers_img 6 --batch_size 1 --folder ./logs/6_Layer_Image/
python3 batch_test_scratch.py --valid_mods image --vit_layers_img 8 --batch_size 1 --folder ./logs/8_Layer_Image/
python3 batch_test_scratch.py --valid_mods image --vit_layers_img 12 --batch_size 1 --folder ./logs/12_Layer_Image/
python3 batch_test_scratch.py --valid_mods image --vit_layers_img 16 --batch_size 1 --folder ./logs/16_Layer_Image/

# Unimodal Image Models
python3 batch_test_scratch.py --valid_mods audio --vit_layers_audio 6 --batch_size 1 --folder ./logs/6_Layer_Audio/
python3 batch_test_scratch.py --valid_mods audio --vit_layers_audio 8 --batch_size 1 --folder ./logs/8_Layer_Audio/
python3 batch_test_scratch.py --valid_mods audio --vit_layers_audio 12 --batch_size 1 --folder ./logs/12_Layer_Audio/
python3 batch_test_scratch.py --valid_mods audio --vit_layers_audio 16 --batch_size 1 --folder ./logs/16_Layer_Audio/