import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import time
import random
import shutil
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.validation import make_valid
import concurrent.futures
import pickle


GAME_TO_SATE_KX = 1.8206
GAME_TO_SATE_BX = 7539.39
GAME_TO_SATE_KY = -1.8220
GAME_TO_SATE_BY = 15287.16
SATE_LENGTH = 24576


def order_points(points):
    hull = ConvexHull(points)
    ordered_points = [points[i] for i in hull.vertices]
    return ordered_points


def calc_intersect_area(poly1, poly2):
    poly2 = order_points(poly2)
    poly2 = Polygon(poly2)
    # 计算交集
    intersection = poly1.intersection(poly2)
    return intersection.area


def game_pos2sate_pos(game_pos_x, game_pos_y):
    sate_pos_x = game_pos_x * GAME_TO_SATE_KX + GAME_TO_SATE_BX
    sate_pos_y = game_pos_y * GAME_TO_SATE_KY + GAME_TO_SATE_BY
    return sate_pos_x, sate_pos_y


def game_pos2tile_pos(game_pos_x, game_pos_y, zoom_list):
    sate_pos_x = game_pos_x * GAME_TO_SATE_KX + GAME_TO_SATE_BX
    sate_pos_y = game_pos_y * GAME_TO_SATE_KY + GAME_TO_SATE_BY
    tile_xy_list = []
    for zoom_level in zoom_list:
        tile_length = SATE_LENGTH // int(2 ** zoom_level)
        tile_x = int(sate_pos_x + tile_length - 1) // tile_length - 1
        tile_y = int(sate_pos_y + tile_length - 1) // tile_length - 1
        # print(tile_x, tile_y)
        tile_xy_list.append((tile_x, tile_y, zoom_level))
    return tile_xy_list


def tile_expand(tile_xy_list, p_xy_list, debug=False):
    tile_expand_list = []
    tile_expand_weight_list = []
    for tile_x, tile_y, zoom_level in tile_xy_list:
        tile_length = SATE_LENGTH // (2 ** zoom_level)
        tile_size = tile_length ** 2
        tile_max_num = 2 ** zoom_level

        p_xy_list_order = order_points(p_xy_list)
        poly_p = Polygon(p_xy_list_order)
        poly_p_area = poly_p.area
        
        tile_tmp = [((tile_x    ) * tile_length, (tile_y    ) * tile_length), 
                    ((tile_x + 1) * tile_length, (tile_y    ) * tile_length), 
                    ((tile_x    ) * tile_length, (tile_y + 1) * tile_length), 
                    ((tile_x + 1) * tile_length, (tile_y + 1) * tile_length)]
        intersect_area = calc_intersect_area(poly_p, tile_tmp)
        max_rate = max(intersect_area / tile_size, intersect_area / poly_p_area)

        tile_expand_list.append((tile_x, tile_y, zoom_level, max_rate))
        if debug:
            print(tile_x, tile_y, max_rate)

        tile_l = max(0, tile_x - 5)
        tile_r = min(tile_x + 5, tile_max_num)
        tile_u = max(0, tile_y - 5)
        tile_d = min(tile_y + 5, tile_max_num)
        
        # # To Left
        # while tile_l >= 1:
        #     tile_tmp = [((tile_l - 1) * tile_length, (tile_y - 1) * tile_length), 
        #                 ((tile_l    ) * tile_length, (tile_y - 1) * tile_length), 
        #                 ((tile_l - 1) * tile_length, (tile_y    ) * tile_length), 
        #                 ((tile_l    ) * tile_length, (tile_y    ) * tile_length)]
        #     intersect_area = calc_intersect_area(p_xy_list, tile_tmp)
        #     if debug:
        #         print('To Left')
        #         print("tile_xy", tile_l, tile_y)
        #         print("tile_tmp", tile_tmp)
        #         print("intersect_area / tile_size", intersect_area, tile_size, intersect_area/tile_size)
        #     tile_l -= 1
        #     if intersect_area / tile_size < 0.4:
        #         break
        # tile_l += 1

        # Enumerate all LRUD
        for tile_x_i in range(tile_l, tile_r + 1):
            for tile_y_i in range(tile_u, tile_d + 1):
                if tile_x_i == tile_x and tile_y_i == tile_y:
                    continue
                tile_tmp = [((tile_x_i    ) * tile_length, (tile_y_i    ) * tile_length), 
                            ((tile_x_i + 1) * tile_length, (tile_y_i    ) * tile_length), 
                            ((tile_x_i    ) * tile_length, (tile_y_i + 1) * tile_length), 
                            ((tile_x_i + 1) * tile_length, (tile_y_i + 1) * tile_length)]
                intersect_area = calc_intersect_area(poly_p, tile_tmp)
                max_rate = max(intersect_area / tile_size, intersect_area / poly_p_area)
                if max_rate > 0.4:
                    # print('jyxjyx', intersect_area / tile_size, intersect_area / poly_p_area)
                    tile_expand_list.append((tile_x_i, tile_y_i, zoom_level, max_rate))
                # if debug:
                    # print('enumerate')
                    # print(tile_x_i, tile_y_i, intersect_area, tile_size, intersect_area / tile_size, poly_p_area, intersect_area/poly_p_area)

        # if debug:
        #     print('jyx tile lrud', zoom_level, tile_l, tile_r, tile_u, tile_d)
        #     print(tile_expand_list)

    return tile_expand_list


def process_per_drone_image(file_data):
    img_file, dir_img, dir_meta, dir_satellite, h, step, start_x, start_y, root, save_root, zoom_list = file_data

    meta_file_path = os.path.join(dir_meta, img_file.replace('.png', '.txt'))
    #### meta_data format
    #### loc_x, loc_y, loc_z, heightAboveGround, cam_x, cam_y, cam_z, cam_rot_x, cam_rot_y, cam_rot_z, P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y, time_hours, time_min, time_sec
    with open(meta_file_path, 'r') as file:
        meta_data = file.readline().strip().split()
        meta_data = [float(x) for x in meta_data]

    p_xy_game_list = meta_data[10:18]
    p_xy_sate_list = [
        game_pos2sate_pos(p_xy_game_list[0], p_xy_game_list[1]),
        game_pos2sate_pos(p_xy_game_list[2], p_xy_game_list[3]),
        game_pos2sate_pos(p_xy_game_list[4], p_xy_game_list[5]),
        game_pos2sate_pos(p_xy_game_list[6], p_xy_game_list[7])
    ]
    cam_x, cam_y = meta_data[4:6]
    tile_xy_list = game_pos2tile_pos(cam_x, cam_y, zoom_list)

    step_num = int(step.replace('step=', ''))
    x_id = round(cam_x - start_x) // step_num
    y_id = round(cam_y - start_y) // step_num
    ids = f"{x_id}_{y_id}"

    debug = (ids == "20_20")
    # debug = False
    # if not debug:  
    tile_expand_list = tile_expand(tile_xy_list, p_xy_sate_list, debug)

    save_drone_dir = os.path.join(save_root, 'drone', ids)
    save_sate_dir = os.path.join(save_root, 'satellite', ids)
    os.makedirs(save_drone_dir, exist_ok=True)
    os.makedirs(save_sate_dir, exist_ok=True)

    drone_img = os.path.join(dir_img, img_file)
    result = {
        "drone_img_dir": dir_img,
        "drone_img": img_file,
        "sate_img_dir": dir_satellite,
        "pair_sate_img_list": [],
        "pair_sate_weight_list": [],
    }
    for tile_x, tile_y, zoom_level, weight in tile_expand_list:
        tile_img = os.path.join(dir_satellite, f'level_{zoom_level}/{zoom_level}_{tile_x}_{tile_y}.png')
        save_drone_img = os.path.join(save_drone_dir, f'{h}_{img_file}')
        save_sate_img = os.path.join(save_sate_dir, f'{h}_{zoom_level}_{tile_x}_{tile_y}.png')
        result["pair_sate_img_list"].append(f'level_{zoom_level}/{zoom_level}_{tile_x}_{tile_y}.png')
        result["pair_sate_weight_list"].append(weight)
        # shutil.copy(drone_img, save_drone_img)
        # shutil.copy(tile_img, save_sate_img)

    if debug:
        print(p_xy_sate_list)
        print(drone_img)
        print('cam_pos', cam_x, cam_y)
        print('cam_pos sate', game_pos2sate_pos(cam_x, cam_y))
        print('tile_expand_list', tile_expand_list)
    return result


def save_pairs_meta_data(pairs_drone2sate_list, pkl_save_path):
    pairs_sate2drone_dict = {}
    pairs_drone2sate_dict = {}
    for pairs_drone2sate in pairs_drone2sate_list:
        
        pair_tile_img_list = pairs_drone2sate["pair_sate_img_list"]
        drone_img = pairs_drone2sate["drone_img"]
        for tile_img in pair_tile_img_list:
            pairs_drone2sate_dict.setdefault(drone_img, []).append(tile_img)
            pairs_sate2drone_dict.setdefault(tile_img, []).append(drone_img)

    pairs_match_set = set()

    for tile_img, tile2drone in pairs_sate2drone_dict.items():
        pairs_sate2drone_dict[tile_img] = list(set(tile2drone))
    for drone_img, drone2tile in pairs_drone2sate_dict.items():
        pairs_drone2sate_dict[drone_img] = list(set(drone2tile))
        for tile_img in pairs_drone2sate_dict[drone_img]:
            pairs_match_set.add((drone_img, tile_img))

    with open(pkl_save_path, 'wb') as f:
        pickle.dump({
            "pairs_drone2sate_list": pairs_drone2sate_list,
            "pairs_sate2drone_dict": pairs_sate2drone_dict,
            "pairs_drone2sate_dict": pairs_drone2sate_dict,
            "pairs_match_set": pairs_match_set,
        }, f)



def process_gta_data(root, save_root, h_list=[100, 200, 300], zoom_list=[5, 6, 7]):
    start_x = -1702
    start_y = -2587.6817

    pairs_drone2sate_list_train = []
    pairs_drone2sate_list_test = []

    for h in h_list:
        path_h = root + f'/drone/H={h}'
        steps = [name for name in os.listdir(path_h) if os.path.isdir(os.path.join(path_h, name))]
        for step in steps:
            dir_img = os.path.join(path_h, step, 'images')
            dir_meta = os.path.join(path_h, step, 'meta_data')
            dir_satellite = os.path.join(root, 'satellite')

            files = [f for f in os.listdir(dir_img)]
            files_num = len(files)
            files_train = files[: files_num//5*4]
            files_test = files[files_num//5*4: ]


            file_data_list_train = [(img_file, dir_img, dir_meta, dir_satellite, h, step, start_x, start_y, root, save_root, zoom_list) for img_file in files_train]
            file_data_list_test = [(img_file, dir_img, dir_meta, dir_satellite, h, step, start_x, start_y, root, save_root, zoom_list) for img_file in files_test]


            with concurrent.futures.ProcessPoolExecutor() as executor:
                for result in tqdm(executor.map(process_per_drone_image, file_data_list_train), total=len(file_data_list_train)):
                    pairs_drone2sate_list_train.append(result)

            with concurrent.futures.ProcessPoolExecutor() as executor:
                for result in tqdm(executor.map(process_per_drone_image, file_data_list_test), total=len(file_data_list_test)):
                    pairs_drone2sate_list_test.append(result)
    
    pkl_save_path = os.path.join(root, 'train_pair_meta_h200300.pkl')
    save_pairs_meta_data(pairs_drone2sate_list_train, pkl_save_path)
    pkl_save_path = os.path.join(root, 'test_pair_meta_h200300.pkl')
    save_pairs_meta_data(pairs_drone2sate_list_test, pkl_save_path)



def get_data(path):
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            data[name] = {"path": os.path.join(root, name)}
            for _, _, files in os.walk(data[name]["path"], topdown=False):
                data[name]["files"] = files
    return data

def get_sate_data(path, level_list=[5, 6, 7]):
    data = {
        "sate_img_dir": path,
        "sate_img": [],
    }
    for zoom_level in level_list:
        sate_img_list = os.listdir(os.path.join(path, f'level_{zoom_level}'))
        sate_img_list = [f'level_{zoom_level}/{sate_img}' for sate_img in sate_img_list]
        data["sate_img"].extend(sate_img_list)
    return data


class GTADatasetTrain(Dataset):
    
    def __init__(self,
                 pairs_meta_file,
                 drone2sate=True,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()
        
        with open(pairs_meta_file, 'rb') as f:
            pairs_meta_data = pickle.load(f)

        self.pairs = []
        pairs_drone2sate_list = pairs_meta_data['pairs_drone2sate_list']
        self.pairs_sate2drone_dict = pairs_meta_data['pairs_sate2drone_dict']
        self.pairs_drone2sate_dict = pairs_meta_data['pairs_drone2sate_dict']
        self.pairs_match_set = pairs_meta_data['pairs_match_set']

        self.drone2sate = drone2sate
        if drone2sate:
            for pairs_drone2sate in pairs_drone2sate_list:
                drone_img_dir = pairs_drone2sate['drone_img_dir']
                drone_img = pairs_drone2sate['drone_img']
                sate_img_dir = pairs_drone2sate['sate_img_dir']
                pair_sate_img_list = pairs_drone2sate['pair_sate_img_list']
                pair_sate_weight_list = pairs_drone2sate['pair_sate_weight_list']

                drone_img_file = f'{drone_img_dir}/{drone_img}'

                for pair_sate_img, pair_sate_weight in zip(pair_sate_img_list, pair_sate_weight_list):
                    sate_img_file = f'{sate_img_dir}/{pair_sate_img}'
                    self.pairs.append((drone_img_file, sate_img_file, pair_sate_weight))

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        
        self.samples = copy.deepcopy(self.pairs)
    
    def __getitem__(self, index):
        
        query_img_path, gallery_img_path, positive_weight = self.samples[index]
        
        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        gallery_img = cv2.imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1) 
        
        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
            
        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']
        
        return query_img, gallery_img, positive_weight

    def __len__(self):
        return len(self.samples)
    
    
    def shuffle(self, ):

            '''
            custom shuffle function for unique class_id sampling in batch
            '''
            
            print("\nShuffle Dataset:")
            
            pair_pool = copy.deepcopy(self.pairs)
              
            # Shuffle pairs order
            random.shuffle(pair_pool)
           
            sate_batch = set()
            drone_batch = set()
            
            # Lookup if already used in epoch
            pairs_epoch = set()   

            # buckets
            batches = []
            current_batch = []
             
            # counter
            break_counter = 0
            
            # progressbar
            pbar = tqdm()

            while True:
                
                pbar.update()
                
                if len(pair_pool) > 0:
                    pair = pair_pool.pop(0)
                    
                    drone_img, sate_img, _ = pair
                    drone_img_name = drone_img.split('/')[-1]
                    sate_img_name = sate_img.split('satellite')[-1][1:]
                    # print(sate_img_name)

                    pair_name = (drone_img_name, sate_img_name)

                    if drone_img_name not in drone_batch and sate_img_name not in sate_batch and pair_name not in pairs_epoch:

                        current_batch.append(pair)
                        pairs_epoch.add(pair_name)
                        
                        pairs_drone2sate = self.pairs_drone2sate_dict[drone_img_name]
                        for sate in pairs_drone2sate:
                            sate_batch.add(sate)
                        pairs_sate2drone = self.pairs_sate2drone_dict[sate_img_name]
                        for drone in pairs_sate2drone:
                            drone_batch.add(drone)
                        
                        break_counter = 0
                        
                    else:
                        # if pair fits not in batch and is not already used in epoch -> back to pool
                        if pair_name not in pairs_epoch:
                            pair_pool.append(pair)
                            
                        break_counter += 1
                        
                    if break_counter >= 16384:
                        break
                else:
                    break

                if len(current_batch) >= self.shuffle_batch_size:
                    # empty current_batch bucket to batches
                    batches.extend(current_batch)
                    sate_batch = set()
                    drone_batch = set()
                    current_batch = []
       
            pbar.close()
            
            # wait before closing progress bar
            time.sleep(0.3)
            
            self.samples = batches
            
            print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples))) 
            print("Break Counter:", break_counter)
            print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))  


class GTADatasetEval(Dataset):
    
    def __init__(self,
                 pairs_meta_file,
                 mode,
                 sate_img_dir='/home/xmuairmud/data/GTA-UAV-data/randcam2_std5_stable/satellite',
                 transforms=None,
                 ):
        super().__init__()
        
        with open(pairs_meta_file, 'rb') as f:
            pairs_meta_data = pickle.load(f)         

        self.images_name = []
        self.images = []

        if mode == 'drone':
            pairs_drone2sate_list = pairs_meta_data['pairs_drone2sate_list']
            self.pairs_sate2drone_dict = pairs_meta_data['pairs_sate2drone_dict']
            self.pairs_drone2sate_dict = pairs_meta_data['pairs_drone2sate_dict']
            self.pairs_match_set = pairs_meta_data['pairs_match_set']
            for pairs_drone2sate in pairs_drone2sate_list:
                self.images.append(os.path.join(pairs_drone2sate['drone_img_dir'], pairs_drone2sate['drone_img']))
                self.images_name.append(pairs_drone2sate['drone_img'])
        elif mode == 'sate':
            sate_datas = get_sate_data(path=sate_img_dir)
            # print('???????', sate_datas['sate_img'])
            for sate_img in sate_datas['sate_img']:
                self.images.append(os.path.join(sate_datas['sate_img_dir'], sate_img))
                self.images_name.append(sate_img)

        self.transforms = transforms


    def __getitem__(self, index):
        
        img_path = self.images[index]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        #if self.mode == "sat":
        
        #    img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #    img180 = cv2.rotate(img90, cv2.ROTATE_90_CLOCKWISE)
        #    img270 = cv2.rotate(img180, cv2.ROTATE_90_CLOCKWISE)
            
        #    img_0_90 = np.concatenate([img, img90], axis=1)
        #    img_180_270 = np.concatenate([img180, img270], axis=1)
            
        #    img = np.concatenate([img_0_90, img_180_270], axis=0)
        
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        
        return img

    def __len__(self):
        return len(self.images)
    
    def get_sample_ids(self):
        return set(self.sample_ids)

    
def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    

    val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])
                                
                             
    train_sat_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*img_size[0]),
                                                               max_width=int(0.2*img_size[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*img_size[0]),
                                                               min_width=int(0.1*img_size[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.RandomRotate90(p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                      ])
    
    train_drone_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                        A.OneOf([
                                                 A.AdvancedBlur(p=1.0),
                                                 A.Sharpen(p=1.0),
                                              ], p=0.3),
                                        A.OneOf([
                                                 A.GridDropout(ratio=0.4, p=1.0),
                                                 A.CoarseDropout(max_holes=25,
                                                                 max_height=int(0.2*img_size[0]),
                                                                 max_width=int(0.2*img_size[0]),
                                                                 min_holes=10,
                                                                 min_height=int(0.1*img_size[0]),
                                                                 min_width=int(0.1*img_size[0]),
                                                                 p=1.0),
                                              ], p=0.3),
                                        A.Normalize(mean, std),
                                        ToTensorV2(),
                                        ])
    
    return val_transforms, train_sat_transforms, train_drone_transforms


def move_file():
    import os
    import shutil

    # 定义源目录和目标目录
    source_directory = "/home/xmuairmud/data/GTA-UAV-data/randcam2_std5/train_new/drone"
    target_directory = "/home/xmuairmud/data/GTA-UAV-data/randcam2_std5/test_new/drone"

    # 检查目标目录是否存在，如果不存在则创建
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # 遍历范围为25到32的所有目录
    for x in range(20, 33):
        for y in range(20, 33):
            directory_name = f"{x}_{y}"
            source_path = os.path.join(source_directory, directory_name)
            
            # 检查目录是否存在
            if os.path.exists(source_path) and os.path.isdir(source_path):
                # 移动目录到目标目录
                shutil.move(source_path, target_directory)
                print(f"Moved: {source_path} to {target_directory}")

    print("All directories moved successfully.")



if __name__ == "__main__":
    root = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std5_stable'
    save_root = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std5/train_h200300'
    process_gta_data(root, save_root, h_list=[200, 300])

    # x = -100
    # y = 314

    # print(game_pos2sate_pos(x, y))
    # print(game_pos2tile_pos(x, y, [5, 6, 7]))

    # from shapely.geometry import Polygon
    # from shapely.validation import make_valid

    # p_xy_list = [(9485.994310144275, 15556.90363294211), (9693.898883986512, 16153.238791564307), (9848.754275218602, 15429.67346430581), (10056.65884906084, 16026.008622928004)]
    # tile_x = 50
    # tile_y = 81
    # tile_length = SATE_LENGTH // int(2 ** 7)
    # tile_tmp = [((tile_x    ) * tile_length, (tile_y    ) * tile_length), 
    #             ((tile_x + 1) * tile_length, (tile_y    ) * tile_length), 
    #             ((tile_x    ) * tile_length, (tile_y + 1) * tile_length), 
    #             ((tile_x + 1) * tile_length, (tile_y + 1) * tile_length)]
    # print(tile_tmp)

    # poly1_points = order_points(tile_tmp)
    # poly2_points = order_points(p_xy_list)

    # poly1 = Polygon(poly1_points)
    # poly2 = Polygon(poly2_points)


    # print(poly1)
    # print(poly2)
    
    # inter1 = poly1.intersection(poly2).area
    # inter2 = poly2.intersection(poly1).area
    # print(inter1, inter2)

    # move_file()