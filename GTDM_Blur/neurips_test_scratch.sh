# Naive Scratch
python3 batch_test_scratch.py --folder ./logs/6_Layer_Naive_Scratch --layer_budget 6 --use_img True --use_dep True --batch_size 1
python3 batch_test_scratch.py --folder ./logs/8_Layer_Naive_Scratch --layer_budget 8 --use_img True --use_dep True --batch_size 1
python3 batch_test_scratch.py --folder ./logs/12_Layer_Naive_Scratch --layer_budget 12 --use_img True --use_dep True --batch_size 1
python3 batch_test_scratch.py --folder ./logs/16_Layer_Naive_Scratch --layer_budget 16 --use_img True --use_dep True --batch_size 1

# Image Only
python3 batch_test_scratch.py --folder ./logs/6_Layer_Image --layer_budget 6 --use_img True --use_dep False --batch_size 1
python3 batch_test_scratch.py --folder ./logs/8_Layer_Image --layer_budget 8 --use_img True --use_dep False --batch_size 1
python3 batch_test_scratch.py --folder ./logs/12_Layer_Image --layer_budget 12 --use_img True --use_dep False --batch_size 1
python3 batch_test_scratch.py --folder ./logs/16_Layer_Image --layer_budget 16 --use_img True --use_dep False --batch_size 1

# Depth Only
python3 batch_test_scratch.py --folder ./logs/6_Layer_Depth --layer_budget 6 --use_img False --use_dep True --batch_size 1
python3 batch_test_scratch.py --folder ./logs/8_Layer_Depth --layer_budget 8 --use_img False --use_dep True  --batch_size 1
python3 batch_test_scratch.py --folder ./logs/12_Layer_Depth --layer_budget 12 --use_img False --use_dep True  --batch_size 1
python3 batch_test_scratch.py --folder ./logs/16_Layer_Depth --layer_budget 16 --use_img False --use_dep True  --batch_size 1