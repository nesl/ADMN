import numpy as np
from viz import *
import cv2

class VideoGenerator:

    def __init__(self, log_dir, valid_mods, valid_nodes, dataset):
        self.valid_mods = valid_mods
        self.valid_nodes = valid_nodes
        self.valid_combinations = []
        for mod in self.valid_mods:
            if (mod == 'mocap'):
                self.valid_combinations.append(('mocap', 'mocap'))
                continue
            for nodes in self.valid_nodes:
                self.valid_combinations.append((mod, 'node_' + str(nodes)))
        self.log_dir = log_dir
        self.dataset = dataset
        self.truck_w = 30
        self.truck_h = 15
        self.colors = ['red', 'green', 'orange', 'black', 'yellow', 'blue']

    def write_video(self, outputs=None, **eval_kwargs): 
        video_length = len(self.dataset)
        fname = f'{self.log_dir}/latest_vid.mp4'
        fig, axes = init_fig(self.valid_combinations)
        size = (fig.get_figwidth()*50, fig.get_figheight()*50)
        size = tuple([int(s) for s in size])
        vid = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'mp4v'), 15, size) # 30 replaces fps

        markers, colors = [], []
        for i in range(100):
            markers.append(',')
            markers.append('o')
            colors.extend(['green', 'red', 'black', 'yellow'])

        frame_count = 0

        for i in trange(video_length):
            data = self.dataset[i]['data']
            save_frame = False
            for key, val in data.items():
                mod, node = key
                if (key not in self.valid_combinations):
                    continue
                if mod == 'mocap':
                    save_frame = True
                    axes[key].clear()
                    axes[key].grid('on', linewidth=3)
                    # axes[key].set_facecolor('gray')
                    #if self.limit_axis: TODO idk if this is good to remove
                    axes[key].set_xlim(0,700)
                    axes[key].set_ylim(0,500)
                    axes[key].set_aspect('equal')

                    num_nodes = len(val['node_pos'])
                    
                    for j in range(num_nodes):
                        pos = val['node_pos'][j]
                        pos = pos[[0, 2]]
                        # INVERT SECOND DIM
                        pos[1] = -pos[1]
                        node_id = val['node_ids'][j] + 1
                        pos = pos + 250
                        axes[key].scatter(pos[0], pos[1], marker='$N%d$' % node_id, color='black', lw=1, s=20*8**2,)
                    
                    num_gt = len(val['gt_positions'])
                    for j in range(num_gt):
                        pos = val['gt_positions'][j]
                        pos = pos[[0, 2]]
                        pos[1] = -pos[1]
                        pos = pos + 250
                        if pos[0] == -1:
                            continue
                        rot = val['gt_rot'][j]
                        ID = val['gt_ids'][j]
                        grid = val['gt_grids'][j]
                        marker = markers[ID]
                        color = colors[ID]
                        
                        axes[key].scatter(pos[0], pos[1], marker=markers[ID], color=color) 
                        
                        angle = rot2angle(rot, return_rads=False)
                        rec, _ = gen_rectange(pos, angle, w=self.truck_w, h=self.truck_h, color=color)
                        axes[key].add_patch(rec)

                        r=self.truck_w/2
                        axes[key].arrow(pos[0], pos[1], r*rot[0], r*rot[1], head_width=0.05*100, head_length=0.05*100, fc=color, ec=color)
                            
                    if outputs is not None: 
                        if len(outputs['det_means']) > 0:
                            pred_means = outputs['det_means'][i]
                            pred_means[:, 1] = -pred_means[:, 1]
                            pred_means = pred_means + 250
                            pred_covs = outputs['det_covs'][i]
                            for j in range(len(pred_means)):
                                mean = pred_means[j].cpu()
                                cov = pred_covs[j].cpu()
                                ID = str(j+1)
                                axes[key].scatter(mean[0], mean[1], color='black', marker='$%s$' % ID, lw=1, s=20*8**2)
                                ellipse = gen_ellipse(mean, cov, edgecolor='black', fc='None', lw=2, linestyle='--')
                                axes[key].add_patch(ellipse)
                        
                        # if 'track_means' in outputs.keys() and len(outputs['track_means'][i]) > 0:
                        pred_means = outputs['track_means'][i] 
                        pred_means[1] = -pred_means[1]
                        pred_means = pred_means + 250
                        pred_covs = outputs['track_covs'][i]
                        #pred_rots = outputs['track_rot'][i]
                        #ids = outputs['track_ids'][i].to(int)
                        # slot_ids = outputs['slot_ids'][i].to(int)
                        print(pred_means, pred_covs, val['gt_positions'] + 250)
                        #for j in range(len(pred_means)):
                            #rot = pred_rots[j]
                            #angle = torch.arctan(rot[0]/rot[1]) * 360

                        mean = pred_means
                        color = self.colors[0]
                        
                        #rec, _ = gen_rectange(mean, angle, w=self.truck_w, h=self.truck_h, color=color)
                        #axes[key].add_patch(rec)


                        # axes[key].scatter(mean[0], mean[1], color=color, marker=f'+', lw=1, s=20*4**2)
                        cov = pred_covs
                        #ID = ids[j]
                        # sID = slot_ids[j]
                        #axes[key].text(mean[0], mean[1], s=f'T${ID}$S{sID}', fontdict={'color': color})
                        axes[key].text(mean[0], mean[1], s=f'KF', fontdict={'color': color})
                        ellipse = gen_ellipse(mean, cov, edgecolor=color, fc='None', lw=2, linestyle='--')
                        axes[key].add_patch(ellipse)
                    
                    

                if mod in ['zed_camera_left', 'realsense_camera_img', 'realsense_camera_depth']:
                    # node_num = int(node[-1])
                    # A = outputs['attn_weights'][i]
                    # A = A.permute(1,0,2) 
                    # nO, nH, L = A.shape
                    # A = A.reshape(nO, nH, 4, 35)
                    # head_dists = A.sum(dim=-1)[..., node_num-1]
                    # head_dists = F.interpolate(head_dists.unsqueeze(0).unsqueeze(0), scale_factor=60)[0][0]
                    
                    # z = torch.zeros_like(head_dists)
                    # head_dists = torch.stack([head_dists,z,z], dim=-1)

                    # head_dists = (head_dists * 255).numpy()
                    # head_dists = (head_dists - 255) * -1
                    # head_dists = head_dists.astype(np.uint8)

                    axes[key].clear()
                    axes[key].axis('off')
                    axes[key].set_title(key) # code = data['zed_camera_left'][:]
                    code = data[key]
                    img = code
                    # img = data[key]['img'].data.cpu().squeeze()
                    # mean = data[key]['img_metas'].data['img_norm_cfg']['mean']
                    # std = data[key]['img_metas'].data['img_norm_cfg']['std']
                    # img = img.permute(1, 2, 0).numpy()
                    # img = (img * std) - mean
                    # img = img.astype(np.uint8)
                    #img = np.concatenate([img, head_dists], axis=0)
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if (len(img.shape) == 3):
                        img = np.transpose(img, (1, 2, 0))
                    axes[key].imshow(img)

                if 'r50' in mod:
                    axes[key].clear()
                    axes[key].axis('off')
                    axes[key].set_title(key) # code = data['zed_camera_left'][:]
                    feat = data[key]['img'].data#[0].cpu().squeeze()
                    feat = feat.mean(dim=0).cpu()
                    feat[feat > 1] = 1
                    feat = (feat * 255).numpy().astype(np.uint8)
                    feat = np.stack([feat]*3, axis=-1)
                    #axes[key].imshow(feat, cmap='turbo')
                    axes[key].imshow(feat)

                 
                if mod == 'zed_camera_depth':
                    axes[key].clear()
                    axes[key].axis('off')
                    axes[key].set_title(key)
                    dmap = data[key]['img'].data[0].cpu().squeeze()
                    axes[key].imshow(dmap, cmap='turbo')#vmin=0, vmax=10000)

                if mod == 'range_doppler':
                    axes[key].clear()
                    axes[key].axis('off')
                    axes[key].set_title(key)
                    # img = data[key]['img'].data[0].cpu().squeeze().numpy()
                    img = data[key]
                    axes[key].imshow(img, cmap='turbo', aspect='auto')

                if mod == 'azimuth_static':
                    axes[key].clear()
                    axes[key].axis('off')
                    axes[key].set_title(key)
                    img = data[key]['img'].data[0].cpu().squeeze().numpy()
                    axes[key].imshow(img, cmap='turbo', aspect='auto')

                if mod == 'mic_waveform':
                    axes[key].clear()
                    axes[key].set_title(key)
                    axes[key].set_ylim(-0.2,1)
                    img = data[key]#['img'].data[0].cpu().squeeze().numpy()
                    max_val = img[0].max()
                    min_val = img[0].min()
                    if max_val == min_val:
                        visual_sig = np.zeros(img[0].shape)
                    else:
                        visual_sig = (img[0] - min_val) / (max_val - min_val)
                    axes[key].plot(visual_sig, color='black')

            if save_frame:
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                data = cv2.resize(data, dsize=size)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                # fname = f'{logdir}/frame_{frame_count}.png'
                # cv2.imwrite(fname, data)
                frame_count += 1
                vid.write(data) 

        vid.release()

