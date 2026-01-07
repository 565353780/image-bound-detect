import cv2
import numpy as np

# 8邻域方向偏移量（按顺时针顺序）
# 从右开始：右、右下、下、左下、左、左上、上、右上
NEIGHBOR_8 = [
    (0, 1),   # 右
    (1, 1),   # 右下
    (1, 0),   # 下
    (1, -1),  # 左下
    (0, -1),  # 左
    (-1, -1), # 左上
    (-1, 0),  # 上
    (-1, 1),  # 右上
]


def find_boundary_pixels(binary_img):
    """
    找到所有边界像素：值为1且8邻域中包含0的像素点
    
    Args:
        binary_img: 二值化图像，值为0或255
        
    Returns:
        boundary_set: 边界像素的集合，每个元素是(y, x)坐标
    """
    h, w = binary_img.shape
    boundary_set = set()
    
    # 将图像归一化到0和1
    img_normalized = (binary_img > 127).astype(np.uint8)
    
    for y in range(h):
        for x in range(w):
            if img_normalized[y, x] == 1:  # 当前像素值为1
                # 检查8邻域中是否有0
                has_zero_neighbor = False
                for dy, dx in NEIGHBOR_8:
                    ny, nx = y + dy, x + dx
                    # 检查是否在图像范围内
                    if 0 <= ny < h and 0 <= nx < w:
                        if img_normalized[ny, nx] == 0:
                            has_zero_neighbor = True
                            break
                    else:
                        # 边界外的像素视为0
                        has_zero_neighbor = True
                        break
                
                if has_zero_neighbor:
                    boundary_set.add((y, x))
    
    return boundary_set


def trace_boundary(binary_img, boundary_set, start_pixel):
    """
    从起始像素开始，沿着连通性追踪边界，形成一个多边形
    
    Args:
        binary_img: 二值化图像
        boundary_set: 所有边界像素的集合（会被修改，移除已使用的像素）
        start_pixel: 起始像素坐标 (y, x)
        
    Returns:
        polygon: 多边形顶点列表，每个元素是(x, y)坐标
    """
    h, w = binary_img.shape
    img_normalized = (binary_img > 127).astype(np.uint8)
    
    polygon = []
    current = start_pixel
    visited_in_trace = set()  # 本次追踪中访问过的像素
    
    # 如果起始像素不在边界集合中，直接返回
    if current not in boundary_set:
        return polygon
    
    # 找到初始方向：从当前像素的8邻域中找到一个值为1的边界像素
    # 从右方向（索引0）开始顺时针搜索
    initial_direction = None
    for i in range(8):
        dy, dx = NEIGHBOR_8[i]
        ny, nx = current[0] + dy, current[1] + dx
        if 0 <= ny < h and 0 <= nx < w:
            neighbor = (ny, nx)
            if neighbor in boundary_set and img_normalized[ny, nx] == 1:
                initial_direction = i
                break
    
    # 如果没有找到初始方向，说明是孤立点
    if initial_direction is None:
        boundary_set.discard(current)
        return [(current[1], current[0])]  # 返回单个点，转换为(x, y)
    
    # 开始追踪
    prev_direction = initial_direction
    polygon.append((current[1], current[0]))  # 添加起始点，转换为(x, y)
    visited_in_trace.add(current)
    
    # 记录上一个移动的方向，用于确定搜索起始方向
    last_move_dir = initial_direction
    
    while True:
        found_next = False
        
        # 从上一个移动方向的反方向开始，顺时针搜索下一个边界像素
        # 这样可以保持沿着边界追踪
        start_search_dir = (last_move_dir + 6) % 8  # 反方向再往前一个位置
        
        for offset in range(8):
            search_dir = (start_search_dir + offset) % 8
            dy, dx = NEIGHBOR_8[search_dir]
            ny, nx = current[0] + dy, current[1] + dx
            
            if 0 <= ny < h and 0 <= nx < w:
                neighbor = (ny, nx)
                # 检查是否是边界像素且值为1
                if neighbor in boundary_set and img_normalized[ny, nx] == 1:
                    # 如果已经访问过这个邻居
                    if neighbor in visited_in_trace:
                        # 检查是否回到起点
                        if neighbor == start_pixel and len(polygon) > 2:
                            # 形成闭环，移除所有访问过的像素
                            boundary_set -= visited_in_trace
                            return polygon
                        else:
                            # 遇到已访问点但不是起点，停止追踪
                            boundary_set -= visited_in_trace
                            return polygon
                    
                    # 找到下一个像素
                    current = neighbor
                    last_move_dir = search_dir
                    
                    # 如果回到起点，形成闭环
                    if current == start_pixel and len(polygon) > 1:
                        boundary_set -= visited_in_trace
                        return polygon
                    
                    polygon.append((current[1], current[0]))  # 转换为(x, y)
                    visited_in_trace.add(current)
                    found_next = True
                    break
        
        # 如果没有找到下一个像素，追踪结束
        if not found_next:
            # 移除所有访问过的像素
            boundary_set -= visited_in_trace
            return polygon


def extract_polygons(img, threshold=None):
    """
    从图像中提取所有多边形
    
    算法流程：
    1. 将图像转为灰度图后二值化
    2. 标记出所有邻域包含0的数值为1的像素点（边界像素）
    3. 每次从中取出一个像素，沿着一个方向将其按照连通性连接起来
    4. 得到一个不一定封闭的多边形
    5. 直到所有像素被选完
    6. 最终得到N个多边形
    
    Args:
        img: 输入图像（可以是彩色或灰度）
        threshold: 二值化阈值，如果为None则使用Otsu自动计算
        
    Returns:
        polygons: 多边形列表，每个多边形是一个numpy数组，形状为(N, 2)，每行是(x, y)坐标
    """
    # 1. 转为灰度图
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    # 2. 二值化
    if threshold is None:
        _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, img_binary = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
    
    # 3. 找到所有边界像素
    boundary_set = find_boundary_pixels(img_binary)
    
    # 4. 追踪所有多边形
    polygons = []
    
    while boundary_set:
        # 取出一个边界像素作为起始点
        start_pixel = next(iter(boundary_set))
        
        # 追踪边界形成多边形
        polygon = trace_boundary(img_binary, boundary_set, start_pixel)
        
        # 如果多边形至少有两个点，则添加到结果中
        if len(polygon) >= 2:
            polygons.append(np.array(polygon, dtype=np.float32))
    
    return polygons

def main():
    image_file_path = "/Users/chli/Downloads/mayi/img/10309_train_stage2_512_init_map_315_robot_2_rotate.png"
    save_image_file_path = "/Users/chli/Downloads/mayi/test_bound.png"

    img = cv2.imread(image_file_path)
    if len(img.shape) == 3:
        H, W = img.shape[:2]
    else:
        H, W = img.shape

    polygons = extract_polygons(img)
    import random

    print(f"找到 {len(polygons)} 个多边形")

    for poly in polygons:
        # poly 形状为 (N, 2)，每行是 (x, y)
        # 确保坐标在图像范围内
        poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)  # x坐标
        poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)  # y坐标
        pts = poly.astype(int)
        
        # 为每个多边形生成不同的颜色
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        
        # 绘制多边形（不一定是封闭的）
        # 检查是否封闭：如果首尾点相同或很接近，则封闭
        is_closed = False
        if len(pts) >= 3:
            first = pts[0]
            last = pts[-1]
            dist = np.sqrt((first[0] - last[0])**2 + (first[1] - last[1])**2)
            if dist < 2:  # 如果首尾距离小于2像素，视为封闭
                is_closed = True
        
        cv2.polylines(img, [pts], isClosed=is_closed, color=color, thickness=1)

    cv2.imwrite(save_image_file_path, img)
    print(f"结果已保存到: {save_image_file_path}")

if __name__ == "__main__":
    main()
