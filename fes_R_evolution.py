import math
import random

import cv2
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from fes_utils import extract_contours
from fes_utils import twist_curve


class FiniteElementStress():

    def __init__(self, number=512, line_points=None, H=1024, x=None, is_ILM=False,
                 swit_direction = 1.,
                 start=100, end=300, force_magnitude=1.0, correct_angle=0., correct_force=False):

        # 定义材料和几何参数
        self.L = number   # 杆件的长度
        self.H = H
        self.N = number - 1  # 节点数量
        self.E = 1e6  # 杨氏模量
        self.A = 1.0  # 横截面积

        # 计算每个节点的坐标
        self.x = np.linspace(0, self.L, self.N + 1) if x is None else x
        self.y = line_points

        # 初始化位移和载荷向量
        self.U = np.zeros(self.N + 1)

        # 刚度矩阵和载荷向量
        self.K_horizontal = np.zeros((self.N + 1, self.N + 1))  # 水平方向的刚度矩阵
        self.K_vertical = np.zeros((self.N + 1, self.N + 1))  # 竖直方向的刚度矩阵
        self.F_vertical = np.zeros(self.N + 1)  # 竖直方向的载荷向量
        self.F_horizontal = np.zeros(self.N + 1)  # 水平方向的载荷向量

        # 应力位置区间
        self.start = start
        self.end = end

        # 修正力的角度 [-30,30]
        self.correct_angle = correct_angle
        self.swit_direction = swit_direction

        # ILM fix_force_type
        self.is_ILM = is_ILM

        # 力的大小
        self.force_magnitude = force_magnitude
        self.force_weight = 100000.0
        self.correct_force = correct_force

    def init_unit_K(self):
        for i in range(self.N):
            dx = self.x[i + 1] - self.x[i]

            dan = np.array([[1, -1], [-1, 1]])

            k_local = self.A * self.E / dx * dan  # 单元刚度矩阵

            self.K_horizontal[i:i + 2, i:i + 2] += k_local
            self.K_vertical[i:i + 2, i:i + 2] += k_local

    def init(self):
        # 初始化位移和载荷向量
        self.U = np.zeros(self.N + 1)
        # 刚度矩阵和载荷向量
        self.K_horizontal = np.zeros((self.N + 1, self.N + 1))  # 水平方向的刚度矩阵
        self.K_vertical = np.zeros((self.N + 1, self.N + 1))  # 竖直方向的刚度矩阵
        self.F_vertical = np.zeros(self.N + 1)  # 竖直方向的载荷向量
        self.F_horizontal = np.zeros(self.N + 1)  # 水平方向的载荷向量

        self.init_unit_K()

    def random_swit_curve(self, sequence, fix_sequence):
        # 随机扭曲边缘

        for ud, index in enumerate(sequence):
            length = fix_sequence[ud+1] - fix_sequence[ud] + 1

            force = round(random.uniform(10, 20), 1) *  length / 12
            # f = self.swit_direction  #if random.randint(0, 1) == 1 else -1
            force = force * self.swit_direction
            # 自定义力的方向角度（相对于水平方向）
            angle_degrees = self.get_angle()
            angle_radians = np.radians(angle_degrees)  # 将角度转换为弧度

            force_x = force * np.cos(angle_radians) * self.force_magnitude * self.force_weight
            force_y = force * np.sin(angle_radians) * self.force_magnitude * self.force_weight

            self.F_vertical[index] += force_y
            self.F_horizontal[index] += force_x


    def customize_angle_force(self):

        # 自定义力的方向角度（相对于水平方向）
        angle_degrees = self.get_angle()
        angle_radians = np.radians(angle_degrees)  # 将角度转换为弧度

        # 定义正态分布的均值和标准差
        mu = 0
        sigma = 2.5
        # 生成一组符合正态分布的随机数

        # 在指定范围内施加力，形成形变
        x_start_index = self.start  # 开始位置索引
        x_end_index = self.end  # 结束位置索引
        coord = np.linspace(-5, 5, x_end_index - x_start_index + 1)

        if random.randint(0,1) == 1 or self.is_ILM:
            force = norm.pdf(coord, mu, sigma) - 0.05  # 使用概率密度函数计算对应的y值
        else:
            force = 0.04 * coord ** 2

        # reversed
        if self.correct_force:
            force += 0.01 * coord ** 2

        # force[50:55] = -force[50:55]

        # plt.plot(coord, force)
        # plt.xlabel('x')
        # plt.ylabel('Probability Density')
        # plt.title('Normal Distribution')
        # plt.grid(True)
        # plt.savefig(r'D:\data\edema_demo\force_Distribution.png')
        # plt.show()

        force_x = force * np.cos(angle_radians) * self.force_magnitude * self.force_weight
        force_y = force * np.sin(angle_radians) * self.force_magnitude * self.force_weight

        self.F_vertical[x_start_index: x_end_index + 1] += force_y
        self.F_horizontal[x_start_index: x_end_index + 1] += force_x

    def fix_specific_point(self, sequence):
        def fix_func(F, K, index):
            K[:, index] = 0  # 将与该节点相关的刚度矩阵列置零
            K[index, :] = 0  # 将与该节点相关的刚度矩阵行置零
            K[index, index] = 1  # 将该节点对应的刚度矩阵元素置为1
            F[index] = 0  # 将与该节点相关的载荷向量元素置零
            return F, K

        for index in sequence:
            self.F_horizontal, self.K_horizontal = fix_func(F=self.F_horizontal, K=self.K_horizontal, index=index)
            self.F_vertical, self.K_vertical = fix_func(F=self.F_vertical, K=self.K_vertical, index=index)

    def fix_both_ends(self):

        def fix_func(F, K, x, L=1, R=10, l=3, r=7):
            fixed_start_index = 0  # 开始位置索引
            if len(np.where(x < l)[0]) == 0:
                k = np.where(x < l)[0]
                print("ok")

            fixed_end_index = l  # 结束位置索引
            K[fixed_start_index: fixed_end_index + 1, :] = 0.0
            K[fixed_start_index: fixed_end_index + 1, fixed_start_index: fixed_end_index + 1] = np.eye(
                fixed_end_index - fixed_start_index + 1)
            F[fixed_start_index: fixed_end_index + 1] = 0.0

            fixed_start_index = r + 1  # 开始位置索引
            fixed_end_index = R  # 结束
            K[fixed_start_index: fixed_end_index + 1, :] = 0.0
            K[fixed_start_index: fixed_end_index + 1, fixed_start_index: fixed_end_index + 1] = np.eye(
                fixed_end_index - fixed_start_index + 1)
            F[fixed_start_index: fixed_end_index + 1] = 0.0

            return F, K

        # 边界条件：固定首尾节点
        # 固定区间x=[1,3)的节点
        self.F_horizontal, self.K_horizontal = fix_func(F=self.F_horizontal, K=self.K_horizontal, x=self.x,
                                                   L=0, R=self.N, l=self.start, r=self.end)
        self.F_vertical, self.K_vertical = fix_func(F=self.F_vertical, K=self.K_vertical, x=self.x,
                                               L=0, R=self.N, l=self.start, r=self.end)

    def get_angle(self):
        # 自定义力的方向角度（相对于水平方向）
        tranx = (self.end - self.start) / (self.y[self.end] - self.y[self.start] + 0.1)
        angle_degrees = 90 + math.atan(tranx) + round(random.uniform(-10, 10), 1)
        #print(angle_degrees)
        return angle_degrees

    def solve_offset(self):
        Uy = np.linalg.solve(self.K_vertical, self.F_vertical)
        Ux = np.linalg.solve(self.K_horizontal, self.F_horizontal)

        # 计算形变后的节点坐标
        deformed_y = self.y + Uy
        deformed_x = self.x + Ux

        return deformed_x, deformed_y

    def check_valid_indices(self, x, y):
        valid_indices = np.where((x >= 0) & (x < self.N) & (y >= 0) & (y < self.H))
        filtered_x = x[valid_indices]
        filtered_y = y[valid_indices]
        assert(min(filtered_x)>=0 and max(filtered_x)<512 )
        assert (min(filtered_y) >= 0 and max(filtered_y) < 1024)

        return filtered_x, filtered_y

    def random_sequence(self):
        # 生成序列个数在10-30之间的随机整数
        seq_length = random.randint(4, 10) * 2 + 1 # [8,30]
        # 生成一个有序的序列，元素为（0，1）之间的两位小数且互不相同
        seq_length = 21
        sequence = sorted(random.sample([random.randint(self.start, self.end) for _ in range(1000)], k=seq_length))

        fix_sequence = sequence[::2]  # 偶数下标序列
        unfix_sequence = sequence[1::2]  # 奇数下标序列

        return fix_sequence, unfix_sequence


    def __call__(self, *args, **kwargs):
        self.init()
        self.customize_angle_force()
        self.fix_both_ends()
        deformed_x, deformed_y =self.solve_offset()
        #random.seed()
        if random.randint(1, 3) > 0 and self.end - self.start > 30:
            self.x = deformed_x
            self.y = deformed_y
            self.init()
            fix_sequence, unfix_sequence = self.random_sequence()
            self.random_swit_curve(sequence=unfix_sequence, fix_sequence=fix_sequence)
            self.fix_both_ends()
            self.fix_specific_point(fix_sequence)
            deformed_x, deformed_y = self.solve_offset()
        #random.seed(473)
        #filtered_x, filtered_y = self.check_valid_indices(deformed_x, deformed_y)

        return deformed_x, deformed_y




def algo_polygon(x_coords, y_coords, padding):
    # 提取点的 x 和 y 坐标


    # x_coords = x_coords.astype(np.uint8)
    # y_coords = y_coords.astype(np.uint8)
    points = [(int(item), int(y_coords[idx])) for idx, item in enumerate(x_coords)]
    image = np.zeros((1024 + padding * 2, 512 + padding * 2, 3), dtype=np.uint8)
    # for x,y in points:
    #     image[y][x] = 255
    # 绘制连线
    for i in range(len(points)-1):
        # 获取当前点和下一个点的坐标
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        # 在图像上连接相邻点
        cv2.line(image, (x1 + padding, y1 + padding), (x2 + padding, y2 + padding), (255, 255, 255), thickness=2)

    # # 绘制起点和终点之间的连线（闭合图形）
    x1, y1 = points[0]
    x2, y2 = points[-1]
    cv2.line(image, (x1 + padding, y1 + padding), (x2 + padding, y2 + padding), (255, 255, 255), thickness=2)
    # 显示图像

    contours = extract_contours(image)

    # cv2.fillPoly(image, contours, (255, 255, 255))

    image = image[padding:-padding, padding:-padding]
    # plt.imshow(image)
    # plt.show()

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def check_coord(y, x):
    return x<0 or x>=512 or y<0 or y>=1024

# 假设水肿影响区域为[l-30, r+30]

# 区间长度 受力大小 受力方向
def draw_layer_with_srf(layer, pd):

    layer = np.array(layer)

    srf_width_limit = 500
    offset = 30 # 左右受力范围
    every_layer_point_number = 512 # 点个数
    layer_number = 6 # 11层个数

    # s_start = 200
    # s_end = 260

    correct_angle = round(random.uniform(-10, 10), 1) # 修正角度

    top_force_weight = round(random.uniform(0.3, 0.6), 2)
    bottom_force_weight = round(random.uniform(0.6, 2.0), 2)

    #bottom_force_weight = 1.5
    top_force_weight = 0.3 + pd * 0.03
    bottom_force_weight = 0.6 + pd * 0.3

    random.seed(473) # 473 500 511(s_end=350)

    # 初始化水肿区间
    s_start = random.randint(1, every_layer_point_number - offset * 5)
    s_end = random.randint(min(s_start + offset // 2, every_layer_point_number - 1),
                           min(s_start + srf_width_limit, every_layer_point_number - 1))
    s_start = 200
    s_end = 400
    if random.randint(0, 2) == 0:
    # 小型扁平形状水肿受力
        bottom_force_magnitude = -1.5 * 90 * 60 / ((s_end - s_start - 5)**2) * bottom_force_weight  # 层受力大小
    else:
    # 中型扁平形状水肿受力
        bottom_force_magnitude = -1.5 * 50 / ((s_end - s_start + 1 )) * bottom_force_weight  # 层受力大小

    top_force_magnitude = -2 * 90 * top_force_weight / (s_end - s_start)   # 层受力大小


    srf_vertical_offset = np.sum(layer[-1] - layer[-2]) // 512


    assert(layer.shape == (layer_number, every_layer_point_number))


    # top 受力建模
    fes = FiniteElementStress(line_points=layer[0], start=max(s_start - offset, 1), end=min(s_end + offset, every_layer_point_number-1),
                              swit_direction=0., is_ILM=True,
                              force_magnitude=top_force_magnitude, correct_angle=correct_angle)
    deformed_x, deformed_y = fes()

    # srf top 受力建模
    fes1 = FiniteElementStress(line_points=layer[-1], start=s_start, end=s_end, correct_force=True,
                               swit_direction=1.,
                               force_magnitude=bottom_force_magnitude, correct_angle=correct_angle)
    deformed_x1, deformed_y1 = fes1()

    # srf bottom 受力建模
    fes2 = FiniteElementStress(line_points=layer[-1], start=s_start, end=s_end,
                               swit_direction=-1.,
                               force_magnitude=top_force_magnitude * -0.3, correct_angle=-correct_angle)
    deformed_x2, deformed_y2 = fes2()

    # get_srf_line
    srf_x, srf_y = [], []

    # origin
    # x_start_index = np.where(fes1.x >= s_start)[0][0]  # 开始位置索引
    # x_end_index = np.where(fes1.x <= s_end)[0][-1]  # 结束位置索引

    x_start_index = np.where(fes2.x >= s_start)[0][0]  # 开始位置索引
    x_end_index = np.where(fes2.x <= s_end)[0][-1]  # 结束位置索引

    x1_start_index = np.where(deformed_x1 >= s_start)[0][0]  # 开始位置索引
    x1_end_index = np.where(deformed_x1 <= s_end)[0][-1]  # 结束位置索引

    # # srf bottom
    # bottom_x, bottom_y = twist_curve(deformed_x2[x_start_index:x_end_index + 1],
    #                                  deformed_y2[x_start_index:x_end_index + 1],
    #                                  noise_factor=0.5)
    # srf_x.extend(bottom_x)
    # srf_y.extend(bottom_y)
    #
    # # srf top
    #
    # top_x, top_y = twist_curve(deformed_x1[x1_start_index:x1_end_index + 1:-1],
    #                            deformed_y1[x1_start_index:x1_end_index + 1:-1],
    #                            noise_factor=0.5)
    # srf_x.extend(top_x)
    # srf_y.extend(top_y)

    # srf bottom
    srf_x.extend(deformed_x2[x_start_index:x_end_index + 1])
    srf_y.extend(deformed_y2[x_start_index:x_end_index + 1])

    # srf top
    srf_x.extend(reversed(deformed_x1[x1_start_index:x1_end_index + 1]))
    srf_y.extend(reversed(deformed_y1[x1_start_index:x1_end_index + 1]))

    srf_y = np.array(srf_y)
    srf_x = np.array(srf_x)
    srf_y -= random.randint(int(srf_vertical_offset * 0.3), int(srf_vertical_offset * 0.8))
    srf_x += random.randint(-offset, offset)

    srf_x, srf_y = twist_curve(srf_x, srf_y, noise_factor=0.5)


    padding = 50
    srf_l, srf_r = min(srf_x), max(srf_x)
    # add padding for prevent coordinate overflow
    res = np.zeros((1024 + padding * 2, 512 + padding * 2))

    for ud, item in enumerate(layer):
        if ud == 0: continue
        for id, x in enumerate(fes.x):
            if x > srf_l - offset and x < srf_r + offset: continue
            #print(int(x))
            res[int(item[id])+padding][int(x)+padding] = 255

    for ud, item in enumerate(deformed_y):
        res[int(item)+padding][int(deformed_x[ud])+padding] = 255

    for ud, item in enumerate(srf_y):
        res[int(item)+padding][int(srf_x[ud])+padding] = 255

    mask = algo_polygon(srf_x, srf_y, padding)
    #print(mask.shape)

    res = res[padding:-padding, padding:-padding]

    res = res.astype(np.uint8) | mask.astype(np.uint8)
    #print(res.shape)
    cv2.imwrite(f"C:/Users/MAC/Desktop/evaluation/draw/DragProcess/evolution_7/plus/2/FES_{pd}.bmp", res)
    plt.imshow(res)
    plt.show()
    return res, mask





if __name__ == '__main__':

    from scipy import io
    cube_name = 'P5578066_Macular Cube 512x128_12-1-2015_9-25-7_OS_sn70487_cube_z'
    dict_ = io.loadmat(
        f'./data/layer_mat/{cube_name}_Surfaces.mat')
    data = dict_['data']
    data = data.transpose(1, 0, 2)

    # B-scan ID
    id = 10
    y = []
    y_arr = []
    origin_layer = np.zeros((1024, 512))

    for c, bscan in enumerate(data):
        if c < id: continue
        k = bscan[0]
        y = np.array(k)
        layerArr = [0, 1, 3, 5, 7, 10] # ILM GCL INL BM  OPR RPE
        for id, item in enumerate(bscan):
            if id not in layerArr: continue
            for x,y in enumerate(item):
                origin_layer[y][x] = 255
            y_arr.append(np.array(item))
        break
    cv2.imwrite(r'C:\Users\MAC\Desktop\evaluation\draw\o_layer.bmp', origin_layer)
    for i in range(10):
        draw_layer_with_srf(np.array(y_arr), pd=i)
        # break









