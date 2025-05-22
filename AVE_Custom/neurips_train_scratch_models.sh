# Train Naive Scratch Models
python3 train.py --vision_vit_layers 3 --audio_vit_layers 3  --max_layerdrop 0 --num_epochs 200 --dir_name 6_Layer_Naive_Scratch --from_scratch --batch_size 8
python3 train.py --vision_vit_layers 4 --audio_vit_layers 4  --max_layerdrop 0 --num_epochs 200 --dir_name 8_Layer_Naive_Scratch --from_scratch  --batch_size 8
python3 train.py --vision_vit_layers 6 --audio_vit_layers 6  --max_layerdrop 0 --num_epochs 200 --dir_name 12_Layer_Naive_Scratch --from_scratch  --batch_size 8
python3 train.py --vision_vit_layers 8 --audio_vit_layers 8  --max_layerdrop 0 --num_epochs 200 --dir_name 16_Layer_Naive_Scratch --from_scratch  --batch_size 8

# Unimodal Image Models
python3 train.py --valid_mods image --vision_vit_layers 6 --max_layerdrop 0 --num_epochs 200 --dir_name 6_Layer_Image --from_scratch  --batch_size 8
python3 train.py --valid_mods image --vision_vit_layers 8 --max_layerdrop 0 --num_epochs 200 --dir_name 8_Layer_Image --from_scratch  --batch_size 8
python3 train.py --valid_mods image --vision_vit_layers 12 --max_layerdrop 0 --num_epochs 200 --dir_name 12_Layer_Image --from_scratch  --batch_size 8
python3 train.py --valid_mods image --vision_vit_layers 16 --max_layerdrop 0 --num_epochs 200 --dir_name 16_Layer_Image --from_scratch  --batch_size 8

# Unimodal Image Models
python3 train.py --valid_mods audio --audio_vit_layers 6 --max_layerdrop 0 --num_epochs 200 --dir_name 6_Layer_Audio --from_scratch  --batch_size 8
python3 train.py --valid_mods audio --audio_vit_layers 8 --max_layerdrop 0 --num_epochs 200 --dir_name 8_Layer_Audio --from_scratch  --batch_size 8
python3 train.py --valid_mods audio --audio_vit_layers 12 --max_layerdrop 0 --num_epochs 200 --dir_name 12_Layer_Audio --from_scratch  --batch_size 8
python3 train.py --valid_mods audio --audio_vit_layers 16 --max_layerdrop 0 --num_epochs 200 --dir_name 16_Layer_Audio --from_scratch  --batch_size 8
