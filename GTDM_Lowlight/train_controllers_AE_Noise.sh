## train controllers for AE
# 6 layers
python3 train_AE_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 5
python3 train_AE_controller.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 5
python3 train_AE_controller.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 5
python3 train_AE_controller.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 5
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 5 
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 5 
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 5 
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 5 

python3 train_AE_controller.py --seed 200 --learning_rate 1e-3 --total_layers 6 --num_epochs 5
python3 train_AE_controller.py --seed 300 --learning_rate 1e-3 --total_layers 6 --num_epochs 5
python3 train_AE_controller.py --seed 400 --learning_rate 1e-3 --total_layers 6 --num_epochs 5
python3 train_AE_controller.py --seed 500 --learning_rate 1e-3 --total_layers 6 --num_epochs 5
python3 train_AE_controller.py --seed 600 --learning_rate 1e-3 --total_layers 6 --num_epochs 5
# 8 layers

python3 train_AE_controller.py --seed 200 --learning_rate 1e-3 --total_layers 8 --num_epochs 5
python3 train_AE_controller.py --seed 300 --learning_rate 1e-3 --total_layers 8 --num_epochs 5
python3 train_AE_controller.py --seed 400 --learning_rate 1e-3 --total_layers 8 --num_epochs 5
python3 train_AE_controller.py --seed 500 --learning_rate 1e-3 --total_layers 8 --num_epochs 5
python3 train_AE_controller.py --seed 600 --learning_rate 1e-3 --total_layers 8 --num_epochs 5
# 12 Layers

python3 train_AE_controller.py --seed 200 --learning_rate 1e-3 --total_layers 12 --num_epochs 5
python3 train_AE_controller.py --seed 300 --learning_rate 1e-3 --total_layers 12 --num_epochs 5
python3 train_AE_controller.py --seed 400 --learning_rate 1e-3 --total_layers 12 --num_epochs 5
python3 train_AE_controller.py --seed 500 --learning_rate 1e-3 --total_layers 12 --num_epochs 5
python3 train_AE_controller.py --seed 600 --learning_rate 1e-3 --total_layers 12 --num_epochs 5
# 16 Layers

python3 train_AE_controller.py --seed 200 --learning_rate 1e-3 --total_layers 16 --num_epochs 5
python3 train_AE_controller.py --seed 300 --learning_rate 1e-3 --total_layers 16 --num_epochs 5
python3 train_AE_controller.py --seed 400 --learning_rate 1e-3 --total_layers 16 --num_epochs 5
python3 train_AE_controller.py --seed 500 --learning_rate 1e-3 --total_layers 16 --num_epochs 5
python3 train_AE_controller.py --seed 600 --learning_rate 1e-3 --total_layers 16 --num_epochs 5

# controllers for noise
# 6 layers

python3 train_controller.py --seed 200 --learning_rate 1e-3 --total_layers 6 --num_epochs 5 
python3 train_controller.py --seed 300 --learning_rate 1e-3 --total_layers 6 --num_epochs 5 
python3 train_controller.py --seed 400 --learning_rate 1e-3 --total_layers 6 --num_epochs 5 
python3 train_controller.py --seed 500 --learning_rate 1e-3 --total_layers 6 --num_epochs 5 
python3 train_controller.py --seed 600 --learning_rate 1e-3 --total_layers 6 --num_epochs 5 
# 8 layers

python3 train_controller.py --seed 200 --learning_rate 1e-3 --total_layers 8 --num_epochs 5 
python3 train_controller.py --seed 300 --learning_rate 1e-3 --total_layers 8 --num_epochs 5 
python3 train_controller.py --seed 400 --learning_rate 1e-3 --total_layers 8 --num_epochs 5 
python3 train_controller.py --seed 500 --learning_rate 1e-3 --total_layers 8 --num_epochs 5 
python3 train_controller.py --seed 600 --learning_rate 1e-3 --total_layers 8 --num_epochs 5 
# 12 layers 

python3 train_controller.py --seed 200 --learning_rate 1e-3 --total_layers 12 --num_epochs 5 
python3 train_controller.py --seed 300 --learning_rate 1e-3 --total_layers 12 --num_epochs 5 
python3 train_controller.py --seed 400 --learning_rate 1e-3 --total_layers 12 --num_epochs 5
python3 train_controller.py --seed 500 --learning_rate 1e-3 --total_layers 12 --num_epochs 5
python3 train_controller.py --seed 600 --learning_rate 1e-3 --total_layers 12 --num_epochs 5
# 16 layers

python3 train_controller.py --seed 200 --learning_rate 1e-3 --total_layers 16 --num_epochs 5 
python3 train_controller.py --seed 300 --learning_rate 1e-3 --total_layers 16 --num_epochs 5 
python3 train_controller.py --seed 400 --learning_rate 1e-3 --total_layers 16 --num_epochs 5
python3 train_controller.py --seed 500 --learning_rate 1e-3 --total_layers 16 --num_epochs 5
python3 train_controller.py --seed 600 --learning_rate 1e-3 --total_layers 16 --num_epochs 5