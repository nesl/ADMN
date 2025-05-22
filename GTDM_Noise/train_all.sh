# Train continuous controller, 6 layers
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=continuous --discretization_method progressive
python3 train_controller.py --seed 200 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=continuous --discretization_method progressive
python3 train_controller.py --seed 300 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=continuous --discretization_method progressive
# Train continuous controller, 8 layers
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=continuous --discretization_method progressive
python3 train_controller.py --seed 200 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=continuous --discretization_method progressive
python3 train_controller.py --seed 300 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=continuous --discretization_method progressive
# Train continuous controller, 12 layers
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=continuous --discretization_method progressive
python3 train_controller.py --seed 200 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=continuous --discretization_method progressive
python3 train_controller.py --seed 300 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=continuous --discretization_method progressive
# Train continuous controller, 16 layers
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=continuous --discretization_method progressive
python3 train_controller.py --seed 200 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=continuous --discretization_method progressive
python3 train_controller.py --seed 300 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=continuous --discretization_method progressive

# Train discrete controller, 6 layers
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=discrete --discretization_method progressive
python3 train_controller.py --seed 200 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=discrete --discretization_method progressive
python3 train_controller.py --seed 300 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=discrete --discretization_method progressive
# Train discrete controller, 8 layers
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=discrete --discretization_method progressive
python3 train_controller.py --seed 200 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=discrete --discretization_method progressive
python3 train_controller.py --seed 300 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=discrete --discretization_method progressive
# Train discrete controller, 12 layers
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=discrete --discretization_method progressive
python3 train_controller.py --seed 200 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=discrete --discretization_method progressive
python3 train_controller.py --seed 300 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=discrete --discretization_method progressive
# Train discrete controller, 16 layers
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=discrete --discretization_method progressive
python3 train_controller.py --seed 200 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=discrete --discretization_method progressive
python3 train_controller.py --seed 300 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=discrete --discretization_method progressive

# Train finite controller, 6 layers
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=finite --discretization_method progressive
python3 train_controller.py --seed 200 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=finite --discretization_method progressive
python3 train_controller.py --seed 300 --learning_rate 1e-3 --total_layers 6 --num_epochs 10 --train_type=finite --discretization_method progressive
# Train finite controller, 8 layers
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=finite --discretization_method progressive
python3 train_controller.py --seed 200 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=finite --discretization_method progressive
python3 train_controller.py --seed 300 --learning_rate 1e-3 --total_layers 8 --num_epochs 10 --train_type=finite --discretization_method progressive
# Train finite controller, 12 layers
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=finite --discretization_method progressive
python3 train_controller.py --seed 200 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=finite --discretization_method progressive
python3 train_controller.py --seed 300 --learning_rate 1e-3 --total_layers 12 --num_epochs 10 --train_type=finite --discretization_method progressive
# Train finite controller, 16 layers
python3 train_controller.py --seed 100 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=finite --discretization_method progressive
python3 train_controller.py --seed 200 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=finite --discretization_method progressive
python3 train_controller.py --seed 300 --learning_rate 1e-3 --total_layers 16 --num_epochs 10 --train_type=finite --discretization_method progressive

#python3 train.py --learning_rate 1e-5 --vision_vit_layers 3 --depth_vit_layers 3 --max_layerdrop 0 --num_epochs 200
# python3 train.py --learning_rate 1e-5 --vision_vit_layers 4 --depth_vit_layers 4 --max_layerdrop 0 --num_epochs 200
# python3 train.py --learning_rate 1e-5 --vision_vit_layers 6 --depth_vit_layers 6 --max_layerdrop 0 --num_epochs 200
# python3 train.py --learning_rate 1e-5 --vision_vit_layers 8 --depth_vit_layers 8 --max_layerdrop 0 --num_epochs 200

