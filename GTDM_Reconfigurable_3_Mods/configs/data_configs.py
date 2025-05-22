#valid_mods=['mocap', 'zed_camera_left', 'range_doppler', 'mic_waveform', "realsense_camera_depth"]
valid_mods = [
    'mocap',
    'zed_camera_left',
    'realsense_camera_depth'
]

valid_nodes=[1, 2, 3]

cache_dir = '/mnt/ssd_8t/redacted/MotiviationalStudy/cached_dataset/'
base_root = '/mnt/ssd_8t/redacted/IROS/Raw_Datasets/single_view_1/'



data_root = base_root + "train"
trainset=dict(
    hdf5_fnames=[
        f'{data_root}/mocap.hdf5',
        f'{data_root}/node_1/mmwave.hdf5',
        f'{data_root}/node_2/mmwave.hdf5',
        f'{data_root}/node_3/mmwave.hdf5',
        # f'{data_root}/node_4/mmwave.hdf5',
        f'{data_root}/node_1/realsense.hdf5',
        f'{data_root}/node_2/realsense.hdf5',
        f'{data_root}/node_3/realsense.hdf5',
        # f'{data_root}/node_4/realsense.hdf5',
        f'{data_root}/node_1/respeaker.hdf5',
        f'{data_root}/node_2/respeaker.hdf5',
        f'{data_root}/node_3/respeaker.hdf5',
        # f'{data_root}/node_4/respeaker.hdf5',
        f'{data_root}/node_1/zed.hdf5',
        f'{data_root}/node_2/zed.hdf5',
        f'{data_root}/node_3/zed.hdf5',
        # f'{data_root}/node_4/zed.hdf5',
        # f'{data_root}/node_1/zed_r50.hdf5',
        # f'{data_root}/node_2/zed_r50.hdf5',
        # f'{data_root}/node_3/zed_r50.hdf5',
        # f'{data_root}/node_4/zed_r50.hdf5',
        # f'{data_root}/node_1/realsense_r50.hdf5',
        # f'{data_root}/node_2/realsense_r50.hdf5',
        # f'{data_root}/node_3/realsense_r50.hdf5',
        # f'{data_root}/node_4/realsense_r50.hdf5',
    ]
)

data_root = base_root + 'val'
valset=dict(
    hdf5_fnames=[
        f'{data_root}/mocap.hdf5',
        f'{data_root}/node_1/mmwave.hdf5',
        f'{data_root}/node_2/mmwave.hdf5',
        f'{data_root}/node_3/mmwave.hdf5',
        # f'{data_root}/node_4/mmwave.hdf5',
        f'{data_root}/node_1/realsense.hdf5',
        f'{data_root}/node_2/realsense.hdf5',
        f'{data_root}/node_3/realsense.hdf5',
        # f'{data_root}/node_4/realsense.hdf5',
        f'{data_root}/node_1/respeaker.hdf5',
        f'{data_root}/node_2/respeaker.hdf5',
        f'{data_root}/node_3/respeaker.hdf5',
        # f'{data_root}/node_4/respeaker.hdf5',
        f'{data_root}/node_1/zed.hdf5',
        f'{data_root}/node_2/zed.hdf5',
        f'{data_root}/node_3/zed.hdf5',
        # f'{data_root}/node_4/zed.hdf5',
        # f'{data_root}/node_1/zed_r50.hdf5',
        # f'{data_root}/node_2/zed_r50.hdf5',
        # f'{data_root}/node_3/zed_r50.hdf5',
        # f'{data_root}/node_4/zed_r50.hdf5',
        # f'{data_root}/node_1/realsense_r50.hdf5',
        # f'{data_root}/node_2/realsense_r50.hdf5',
        # f'{data_root}/node_3/realsense_r50.hdf5',
        # f'{data_root}/node_4/realsense_r50.hdf5',

    ]
)

data_root = base_root + 'test'
testset=dict(
    hdf5_fnames=[
        f'{data_root}/mocap.hdf5',
        f'{data_root}/node_1/mmwave.hdf5',
        f'{data_root}/node_2/mmwave.hdf5',
        f'{data_root}/node_3/mmwave.hdf5',
        # f'{data_root}/node_4/mmwave.hdf5',
        f'{data_root}/node_1/realsense.hdf5',
        f'{data_root}/node_2/realsense.hdf5',
        f'{data_root}/node_3/realsense.hdf5',
        # f'{data_root}/node_4/realsense.hdf5',
        f'{data_root}/node_1/respeaker.hdf5',
        f'{data_root}/node_2/respeaker.hdf5',
        f'{data_root}/node_3/respeaker.hdf5',
        # f'{data_root}/node_4/respeaker.hdf5',
        f'{data_root}/node_1/zed.hdf5',
        f'{data_root}/node_2/zed.hdf5',
        f'{data_root}/node_3/zed.hdf5',
        # f'{data_root}/node_4/zed.hdf5',
        # f'{data_root}/node_1/zed_r50.hdf5',
        # f'{data_root}/node_2/zed_r50.hdf5',
        # f'{data_root}/node_3/zed_r50.hdf5',
        # f'{data_root}/node_4/zed_r50.hdf5',
        # f'{data_root}/node_1/realsense_r50.hdf5',
        # f'{data_root}/node_2/realsense_r50.hdf5',
        # f'{data_root}/node_3/realsense_r50.hdf5',
        # f'{data_root}/node_4/realsense_r50.hdf5',
    ]
)


