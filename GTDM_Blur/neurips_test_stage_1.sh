# 3 layers 1 2 3 4 5 6 7 8 9
# 4 layers 1 2 3 4 5 6 7 9
# 6 layers 2 4 5 6 7 9
# 8 layers 3 5 7 9

# UPPER BOUND
python3 batch_test.py --folder ./logs/Stage_1_Model --batch_size 1

# NAIVE ALLOC TESTS

# 3, 3 split
python3 batch_test.py --folder ./logs/Stage_1_Model --checkpoint last.pt --drop_layers_img 1 2 3 4 5 6 7 8 9 --drop_layers_depth 1 2 3 4 5 6 7 8 9 --batch_size 1
# 4, 4, split
python3 batch_test.py --folder ./logs/Stage_1_Model --checkpoint last.pt --drop_layers_img 1 2 3 4 5 6 7 9 --drop_layers_depth 1 2 3 4 5 6 7 9 --batch_size 1
#6, 6, split
python3 batch_test.py --folder ./logs/Stage_1_Model --checkpoint last.pt --drop_layers_img 2 4 5 6 7 9 --drop_layers_depth 2 4 5 6 7 9 --batch_size 1
#8, 8 split
python3 batch_test.py --folder ./logs/Stage_1_Model --checkpoint last.pt --drop_layers_img 3 5 7 9 --drop_layers_depth 3 5 7 9 --batch_size 1

# All Layers Img
# 6 layers Img, None Depth
python3 batch_test.py --folder ./logs/Stage_1_Model --checkpoint last.pt --drop_layers_img 2 4 5 6 7 9 --drop_layers_depth 0 1 2 3 4 5 6 7 8 9 10 11 --batch_size 1 
# 8 layers Img, None Depth
python3 batch_test.py --folder ./logs/Stage_1_Model --checkpoint last.pt --drop_layers_img 3 5 7 9 --drop_layers_depth 0 1 2 3 4 5 6 7 8 9 10 11 --batch_size 1
# 12 layers Img, None Depth
python3 batch_test.py --folder ./logs/Stage_1_Model --checkpoint last.pt  --drop_layers_depth 0 1 2 3 4 5 6 7 8 9 10 11 --batch_size 1

# All Layers Depth
# 6 layers Img, None Depth
python3 batch_test.py --folder ./logs/Stage_1_Model --checkpoint last.pt --drop_layers_depth 2 4 5 6 7 9 --drop_layers_img 0 1 2 3 4 5 6 7 8 9 10 11 --batch_size 1
# 8 layers Img, None Depth
python3 batch_test.py --folder ./logs/Stage_1_Model --checkpoint last.pt --drop_layers_depth 3 5 7 9 --drop_layers_img 0 1 2 3 4 5 6 7 8 9 10 11 --batch_size 1
# 12 layers Img, None Depth
python3 batch_test.py --folder ./logs/Stage_1_Model --checkpoint last.pt --drop_layers_img 0 1 2 3 4 5 6 7 8 9 10 11 --batch_size 1
