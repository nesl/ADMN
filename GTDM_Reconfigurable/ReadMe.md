# IoBT Tracking

## Running

To train the multimodal model:
    python3 train.py

    To change the mdoality dropout, change the image mask and depth mask parameters passed to the PickelData transform_input function

To train the controller model:
    python3 train_controller.py
    To change the mdoality dropout, change the image mask and depth mask parameters passed to the PickelData transform_input function


python3 batch_test.py --folder ./logs/Controller_Multimodal_Noisy/ --checkpoint last_67.pt --batch_size 1