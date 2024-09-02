import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns
from PIL import Image
import pickle
import cv2

import os

Image.MAX_IMAGE_PIXELS = None


def gaussin_random_truncted(lower_bound, upper_bound, mean, std_dev):
    while True:
        val = random.gauss(mean, std_dev)
        if lower_bound <= val <= upper_bound:
            return val


def draw_backgroud():
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # 设置图形背景
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.set_zlim([0, 150])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


    # 设置z轴在左边
    ax.zaxis.set_rotate_label(True)  # disable automatic rotation
    ax.set_zlabel('Z', rotation=0)

    # # 强化格子线条
    # ax.xaxis._axinfo['grid'].update(color='k', linestyle='-', linewidth=1.0)
    # ax.yaxis._axinfo['grid'].update(color='k', linestyle='-', linewidth=1.0)
    # ax.zaxis._axinfo['grid'].update(color='k', linestyle='-', linewidth=1.0)

    # 强化坐标轴的粗线
    ax.xaxis._axinfo['axisline'].update(color='k', linewidth=2)
    ax.yaxis._axinfo['axisline'].update(color='k', linewidth=2)
    ax.zaxis._axinfo['axisline'].update(color='k', linewidth=2)

    # 保留格子线条
    ax.grid(True)

    # 去掉背景
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # 保存为无背景的png图片，高分辨率
    plt.savefig('3d_background.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()


def draw_attitude_roll_pitch_2():
    yaw_range = [-180, 180]
    pitch_range = [-100, -80]
    roll_range = [-10, 10]
    std = 5
    pitchs = [gaussin_random_truncted(pitch_range[0], pitch_range[1], -90, std) for _ in range(50000)]
    rolls = [gaussin_random_truncted(roll_range[0], roll_range[1], 0, std) for _ in range(50000)]
    yaws = [random.uniform(-180.0, 180.0) for _ in range(50000)]
    
    data = pd.DataFrame({
        'Pitch': pitchs,
        'Roll': rolls
    })

    sns.set_theme(style="ticks")

    ## #7970A6 #4CB391 #183E67
    g = sns.jointplot(x=rolls, y=pitchs, kind="hex", color="#5975A4", marginal_ticks=True)
    
    g.set_axis_labels('Roll', 'Pitch')

    plt.savefig('attitude_angles_roll_pitch.pdf', transparent=True, bbox_inches='tight', pad_inches=0)
    # plt.savefig('attitude_angles_roll_pitch_2.pdf')

def draw_attitude_roll_pitch():
    # 生成一些示例数据
    yaw_range = [-180, 180]
    pitch_range = [-110, -70]
    roll_range = [-10, 10]
    std = 5
    pitchs = [gaussin_random_truncted(pitch_range[0], pitch_range[1], -90, std) for _ in range(50000)]
    rolls = [gaussin_random_truncted(roll_range[0], roll_range[1], 0, std) for _ in range(50000)]
    yaws = [random.uniform(-180.0, 180.0) for _ in range(50000)]

    y = rolls
    x = pitchs

    # 创建图表
    fig = plt.figure(figsize=(5, 5), dpi=300)
    grid = plt.GridSpec(4, 4, hspace=0.4, wspace=0.4)

    # Y 维度的分布图（放在左边，反转左右）
    y_hist = fig.add_subplot(grid[1:, 0])
    y_hist.hist(y, bins=2000, orientation='horizontal', color='#F8D795', alpha=1.0)
    y_hist.invert_xaxis()  # 反转左右
    y_hist.axis('off')

    pos1 = y_hist.get_position() # get the original position 
    pos2 = [pos1.x0 - 0.08, pos1.y0, pos1.width, pos1.height] 
    y_hist.set_position(pos2) # set a new position

    # 主散点图
    main_ax = fig.add_subplot(grid[1:, 1:])
    main_ax.scatter(x, y, alpha=0.3, color='#C4C0D9')
    main_ax.axis('on')

    main_ax.set_xlabel('Pitch')
    main_ax.set_ylabel('Roll', labelpad=-5)
    # main_ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°'])
    # main_ax.set_yticklabels(['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°'])

    # X 维度的分布图
    x_hist = fig.add_subplot(grid[0, 1:], sharex=main_ax)
    x_hist.hist(x, bins=2000, color='#7970A6', alpha=1.0)
    x_hist.axis('off')

    legend_elements = [
        Patch(facecolor='#7970A6', label='Pitch'),
        Patch(facecolor='#F8D795', label='Roll')
    ]
    main_ax.legend(handles=legend_elements, loc='upper right')

    plt.savefig('attitude_angles_roll_pitch.pdf', transparent=True, bbox_inches='tight', pad_inches=0)


def draw_attitude_yaw():
    yaw_range = [-180, 180]
    pitch_range = [-110, -70]
    roll_range = [-10, 10]
    std = 5
    pitchs = [gaussin_random_truncted(pitch_range[0], pitch_range[1], -90, std) for _ in range(50000)]
    rolls = [gaussin_random_truncted(roll_range[0], roll_range[1], 0, std) for _ in range(50000)]
    yaws = [random.uniform(-180.0, 180.0) for _ in range(50000)]

    print(pitchs[0], rolls[0], yaws[0])

    pitchs_rad = np.radians(pitchs)
    rolls_rad = np.radians(rolls)
    yaws_rad = np.radians(yaws)


    # plt.figure(figsize=(12, 4), dpi=300)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw=dict(projection='polar'), dpi=300)

    # bins = np.linspace(-115, 15, 1000)
    # ax[0].hist(rolls, bins, alpha=0.6, label='Roll', color='#7970A6')
    # ax[0].hist(pitchs, bins, alpha=0.6, label='Pitch', color='#E76768')
    # ax[0].hist(yaws, bins, alpha=0.6, label='Yaw', color='#F8D795')

    # ax = fig.add_subplot(122, projection='polar')
    # ax[1].hist(rolls_rad, bins=10, color='#7970A6', label='Roll')

    # ax[1].hist(pitchs_rad, bins=20, color='#E76768', label='Pitch')

    ax.hist(yaws_rad, bins=30, label='Yaw', color='#4C84B6', edgecolor='k')

    ax.set_theta_zero_location('N')  # 设置0度位置为北（即顶部）
    ax.set_theta_direction(-1)  # 设置角度增加方向为顺时针
    ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°'])
    # ax.set_yticklabels([])

    # fig.legend(loc='upper center')

    plt.show()
    plt.savefig('attitude_angles_yaw.pdf', transparent=True, bbox_inches='tight', pad_inches=0)


def draw_altitude():
    # altitude = []
    
    # directory = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable_all/drone/meta_data'
    # for filename in os.listdir(directory):
    #     if filename.endswith('.txt'):
    #         filepath = os.path.join(directory, filename)
    #         with open(filepath, 'r') as file:
    #             line = file.readline().strip()
    #             values = line.split()
    #             if len(values) >= 4:
    #                 h = float(values[3])
    #                 altitude.append(h)
    #                 if h > 150 and h < 250:
    #                     altitude.append(h - 80)

    # print(len(altitude))
    # sns.set_theme(style="darkgrid")
    plt.figure(figsize=(4, 6))

    alttitude = list(reversed([5263 * 5 // 3.6, 5963 * 5 // 3.6, 6058 * 5 // 3.6, 6106 * 5 // 3.6, 6166 * 5 // 3.6, 6187 * 5 // 3.6]))
    height = list(reversed(['100m', '200m', '300m', '400m', '500m', '600m']))
    data = pd.DataFrame({
        'Altitude': height,
        'Count': alttitude,
    })
    sns.barplot(data, x="Count", y="Altitude", color='#4C84B6')
    plt.grid(True, axis='x', color='gray', linestyle='-', linewidth=0.5)

    # plt.savefig('altitude.pdf')

    # counts, bin_edges = np.histogram(altitude, bins=bins)

    # plt.figure(figsize=(10, 5), dpi=300)

    # sns.set_theme(style="darkgrid")
    # # plt.barh(bin_edges[:-1], counts, height=(bin_edges[1] - bin_edges[0]))
    
    # ax.set_xlim(100, 630)
    # ax.set_xlabel('Altitude (m)')
    # ax.set_ylabel('Count')

    # plt.xlabel('Altitude (m)')
    # plt.xlabel('Count')
    # plt.xticks([3000, 6000], ['4000', '8000'])
    # plt.ylim(0, 600)

    plt.show()
    plt.savefig('altitude_hist.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)


def draw_scenes_pie():
    # 示例数据
    labels = ['City', 'Mountain', 'Desert', 'Forest', 'Field', 'Seaside']
    sizes = [56.13, 12.52, 10.19, 4.27, 8.29, 8.60]
    colors = ['#6a5acd', '#ff6f61', '#00bfff', '#ff69b4', '#ffcc99', '#dcdcdc']

    # 绘制饼图
    fig, ax = plt.subplots(figsize=(5,5), dpi=300)
    wedges, texts, autotexts = ax.pie(sizes, colors=colors, autopct='%1.2f%%',
                                    startangle=140, pctdistance=0.85, wedgeprops=dict(width=0.3))

    # 添加中心的文本
    plt.setp(autotexts, size=6, weight="bold", color="white")
    ax.text(0, 0, 'Scene\nDistribution', ha='center', va='center', fontsize=13, fontweight='bold')
    # plt.setp(autotexts, size=6, weight="bold", color="white")

    # 添加标签
    # for i, text in enumerate(texts):
    #     text.set_text(labels[i])
    #     text.set_fontsize(10)
    #     text.set_color('gray')
    
    plt.savefig('scene_pie.png', transparent=True, bbox_inches='tight', pad_inches=0)


def plot_k_ablation():
    altitude = []
    

def f(k, x):
    return 1 / (1 + np.exp(-k * x))
    
def plot_func():
    pass


def gen_attitudes():
    yaw_range = [-180, 180]
    pitch_range = [-110, -70]
    roll_range = [-10, 10]
    std = 5
    pitchs = [gaussin_random_truncted(pitch_range[0], pitch_range[1], -90, std) for _ in range(50000)]
    rolls = [gaussin_random_truncted(roll_range[0], roll_range[1], 0, std) for _ in range(50000)]
    yaws = [random.uniform(-180.0, 180.0) for _ in range(50000)]

    # 创建一个简单的数据框
    data = {
        'Pitch': pitchs,
        'Roll': rolls,
        'Yaw': yaws
    }

    df = pd.DataFrame(data)

    # 写出数据到CSV文件
    df.to_csv('output.csv', index=False)


def draw_loc():

    train_pickle = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable_all/cross_h23456_z41_iou4_oc4/train_pair_meta.pkl'
    test_pickle = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable_all/cross_h23456_z41_iou4_oc4/test_pair_meta.pkl'
    with open(train_pickle, 'rb') as f:
        data_train = pickle.load(f)
    with open(test_pickle, 'rb') as f:
        data_test = pickle.load(f)

    x_coords_train = []
    y_coords_train = []
    x_coords_test = []
    y_coords_test = []
    for data_sate in data_train['pairs_drone2sate_list']:
        x_coords_train.append(data_sate['drone_loc_x_y'][0]/0.45)
        y_coords_train.append(data_sate['drone_loc_x_y'][1]/0.45)
    for data_sate in data_test['pairs_drone2sate_list']:
        x_coords_test.append(data_sate['drone_loc_x_y'][0]/0.45)
        y_coords_test.append(data_sate['drone_loc_x_y'][1]/0.45)


    image_path = '/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-sate.png'  # 替换为你的图片路径
    image = Image.open(image_path)

    # 获取图片的分辨率
    width, height = image.size

    plt.figure(figsize=(width / 1000, height / 1000), dpi=1000)

    plt.imshow(image)

    plt.scatter(x_coords_train, y_coords_train, color='red', s=10)
    plt.scatter(x_coords_test, y_coords_test, color='blue', s=10)
    # plt.scatter(x_coords_train, y_coords_train, color='red', s=5)

    # 设置图片的边界与比例
    plt.xlim(0, width)
    plt.ylim(height, 0)  # 因为图像坐标系的原点在左上角，所以要反转 y 轴
    plt.gca().set_aspect('equal', adjustable='box')

    # 隐藏坐标轴
    plt.axis('off')
    plt.savefig('GTA-UAV-sample-dist-cross.png', transparent=True, bbox_inches='tight', pad_inches=0)


def resize_img():
    a = cv2.imread('GTA-UAV-sample-dist-cross.png')
    height, width, _ = a.shape
    print(height, width)
    a = cv2.resize(a, (width // 8, height // 8))
    cv2.imwrite('GTA-UAV-sample-dist-cross-resize.png', a)


if __name__ == '__main__':
    # gen_attitudes()
    draw_attitude_roll_pitch_2()
    draw_loc()
    resize_img()
    # gen_attitudes()
    # draw_attitude_roll_pitch_2()
    # draw_attitude_yaw()
    # draw_attitude_roll_pitch()
    # draw_altitude()
    # draw_scenes_pie()
