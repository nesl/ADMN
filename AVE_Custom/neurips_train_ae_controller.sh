
python3 train_AE.py --num_epochs 100
python3 train_controller_AE.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --batch_size 8
python3 train_controller_AE.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --batch_size 8
python3 train_controller_AE.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --batch_size 8
python3 train_controller_AE.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --batch_size 8