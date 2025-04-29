mkdir ./logs/Progressive_Point1
python3 train_controller_AE.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --batch_size 8
python3 train_controller_AE.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --batch_size 8
python3 train_controller_AE.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --batch_size 8
python3 train_controller_AE.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --batch_size 8

python3 train_controller_AE.py --seed 200 --learning_rate 1e-3 --total_layers 6 --num_epochs 10  --batch_size 8
python3 train_controller_AE.py --seed 200 --learning_rate 1e-3 --total_layers 8 --num_epochs 10  --batch_size 8
python3 train_controller_AE.py --seed 200 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --batch_size 8
python3 train_controller_AE.py --seed 200 --learning_rate 1e-3 --total_layers 16 --num_epochs 10  --batch_size 8

python3 train_controller_AE.py --seed 300 --learning_rate 1e-3 --total_layers 6 --num_epochs 10  --batch_size 8
python3 train_controller_AE.py --seed 300 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --batch_size 8
python3 train_controller_AE.py --seed 300 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --batch_size 8
python3 train_controller_AE.py --seed 300 --learning_rate 1e-3 --total_layers 16 --num_epochs 10  --batch_size 8