python3 train_AE.py --num_epochs 100
python3 train_AE_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --discretization_method admn 
python3 train_AE_controller.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --discretization_method admn
python3 train_AE_controller.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --discretization_method admn
python3 train_AE_controller.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 -discretization_method admn