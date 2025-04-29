python3 batch_test.py --folder $1 --checkpoint last.pt --drop_layers 
python3 batch_test.py --folder $1 --checkpoint last.pt --drop_layers 6
python3 batch_test.py --folder $1 --checkpoint last.pt --drop_layers 6 8
python3 batch_test.py --folder $1 --checkpoint last.pt --drop_layers 4 6 8
python3 batch_test.py --folder $1 --checkpoint last.pt --drop_layers 2 4 6 8
python3 batch_test.py --folder $1 --checkpoint last.pt --drop_layers 2 4 6 8 9
python3 batch_test.py --folder $1 --checkpoint last.pt --drop_layers 2 4 6 7 8 9
python3 batch_test.py --folder $1 --checkpoint last.pt --drop_layers 1 2 4 6 7 8 9
python3 batch_test.py --folder $1 --checkpoint last.pt --drop_layers 1 2 4 5 6 7 8 9
python3 batch_test.py --folder $1 --checkpoint last.pt --drop_layers 1 2 3 4 5 6 7 8 9
python3 batch_test.py --folder $1 --checkpoint last.pt --drop_layers 1 2 3 4 5 6 7 8 9 10 # Change to drop layer 10 the last