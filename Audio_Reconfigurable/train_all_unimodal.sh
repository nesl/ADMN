# Train unimodal image only first
python3 train.py --valid_mods image --vision_vit_layers 6 --max_layerdrop 0 --num_epochs 200
python3 train.py --valid_mods image --vision_vit_layers 8 --max_layerdrop 0 --num_epochs 200
python3 train.py --valid_mods image --vision_vit_layers 12 --max_layerdrop 0 --num_epochs 200
python3 train.py --valid_mods image --vision_vit_layers 16 --max_layerdrop 0 --num_epochs 200

# Train unimodal audio next
python3 train.py --valid_mods audio --audio_vit_layers 6 --max_layerdrop 0 --num_epochs 200
python3 train.py --valid_mods audio --audio_vit_layers 8 --max_layerdrop 0 --num_epochs 200
python3 train.py --valid_mods audio --audio_vit_layers 12 --max_layerdrop 0 --num_epochs 200
python3 train.py --valid_mods audio --audio_vit_layers 16 --max_layerdrop 0 --num_epochs 200
