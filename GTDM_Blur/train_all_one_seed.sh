mkdir ./logs/Progressive_Point1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=continuous --discretization_method progressive --temp 0.1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=continuous --discretization_method progressive --temp 0.1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=continuous --discretization_method progressive --temp 0.1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=continuous --discretization_method progressive --temp 0.1

python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=discrete --discretization_method progressive --temp 0.1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=discrete --discretization_method progressive --temp 0.1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=discrete --discretization_method progressive --temp 0.1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=discrete --discretization_method progressive --temp 0.1 

python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=finite --discretization_method progressive --temp 0.1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=finite --discretization_method progressive --temp 0.1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=finite --discretization_method progressive --temp 0.1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=finite --discretization_method progressive --temp 0.1

mv ./logs/Controller*/ ./logs/Progressive_Point1

mkdir ./logs/Progressive_10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=continuous --discretization_method progressive --temp 10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=continuous --discretization_method progressive --temp 10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=continuous --discretization_method progressive --temp 10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=continuous --discretization_method progressive --temp 10

python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=discrete --discretization_method progressive --temp 10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=discrete --discretization_method progressive --temp 10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=discrete --discretization_method progressive --temp 10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=discrete --discretization_method progressive --temp 10 

python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=finite --discretization_method progressive --temp 10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=finite --discretization_method progressive --temp 10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=finite --discretization_method progressive --temp 10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=finite --discretization_method progressive --temp 10

mv ./logs/Controller*/ ./logs/Progressive_10


mkdir ./logs/ADMN_Point1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=continuous --discretization_method admn --temp 0.1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=continuous --discretization_method admn --temp 0.1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=continuous --discretization_method admn --temp 0.1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=continuous --discretization_method admn --temp 0.1

python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=discrete --discretization_method admn --temp 0.1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=discrete --discretization_method admn --temp 0.1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=discrete --discretization_method admn --temp 0.1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=discrete --discretization_method admn --temp 0.1 

python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=finite --discretization_method admn --temp 0.1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=finite --discretization_method admn --temp 0.1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=finite --discretization_method admn --temp 0.1
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=finite --discretization_method admn --temp 0.1

mv ./logs/Controller*/ ./logs/ADMN_Point1

mkdir ./logs/ADMN_10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=continuous --discretization_method admn --temp 10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=continuous --discretization_method admn --temp 10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=continuous --discretization_method admn --temp 10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=continuous --discretization_method admn --temp 10

python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=discrete --discretization_method admn --temp 10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=discrete --discretization_method admn --temp 10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=discrete --discretization_method admn --temp 10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=discrete --discretization_method admn --temp 10 

python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=finite --discretization_method admn --temp 10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=finite --discretization_method admn --temp 10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=finite --discretization_method admn --temp 10
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=finite --discretization_method admn --temp 10

mv ./logs/Controller*/ ./logs/ADMN_10