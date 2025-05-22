# train_adamml_subnets Naive Scratch Models
python3 train_adamml_subnets.py --dir_name 6_Layer_Naive_Scratch --use_img True --use_dep True --layer_budget 6 --learning_rate 1e-5  --num_epochs 200 
python3 train_adamml_subnets.py --dir_name 8_Layer_Naive_Scratch --use_img True --use_dep True  --layer_budget 8 --learning_rate 1e-5  --num_epochs 200 
python3 train_adamml_subnets.py --dir_name 12_Layer_Naive_Scratch --use_img True --use_dep True --layer_budget 12  --learning_rate 1e-5  --num_epochs 200 
python3 train_adamml_subnets.py --dir_name 16_Layer_Naive_Scratch --use_img True --use_dep True  --layer_budget 16 --learning_rate 1e-5  --num_epochs 200 

# Image Only
python3 train_adamml_subnets.py --dir_name 6_Layer_Image --use_img True --use_dep False --layer_budget 6 --learning_rate 1e-5  --num_epochs 200 
python3 train_adamml_subnets.py --dir_name 8_Layer_Image --use_img True --use_dep False --layer_budget 8 --learning_rate 1e-5  --num_epochs 200 
python3 train_adamml_subnets.py --dir_name 12_Layer_Image --use_img True --use_dep False --layer_budget 12 --learning_rate 1e-5  --num_epochs 200 
python3 train_adamml_subnets.py --dir_name 16_Layer_Image --use_img True --use_dep False --layer_budget 16 --learning_rate 1e-5  --num_epochs 200 

# Depth Only
python3 train_adamml_subnets.py --dir_name 6_Layer_Depth --use_img False --use_dep True --layer_budget 6 --learning_rate 1e-5  --num_epochs 200 
python3 train_adamml_subnets.py --dir_name 8_Layer_Depth --use_img False --use_dep True --layer_budget 8 --learning_rate 1e-5  --num_epochs 200 
python3 train_adamml_subnets.py --dir_name 12_Layer_Depth --use_img False --use_dep True --layer_budget 12  --learning_rate 1e-5  --num_epochs 200 
python3 train_adamml_subnets.py --dir_name 16_Layer_Depth --use_img False --use_dep True --layer_budget 16 --learning_rate 1e-5  --num_epochs 200 
