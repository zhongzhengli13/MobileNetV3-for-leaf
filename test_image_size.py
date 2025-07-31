from PIL import Image
import os


def check_image_size(image_path):
    """
    检查并打印单张图片的尺寸。

    参数:
        image_path (str): 图片文件的路径。
    """
    if not os.path.exists(image_path):
        print(f"错误: 文件 '{image_path}' 不存在。")
        return

    try:
        # 打开图片文件
        with Image.open(image_path) as img:
            # 获取图片的宽度和高度
            width, height = img.size
            print(f"图片路径: {image_path}")
            print(f"尺寸 (宽 x 高): {width} x {height} 像素")
    except Exception as e:
        print(f"无法打开或处理图片 '{image_path}'。错误: {e}")


# --- 使用示例 ---
if __name__ == '__main__':
    # !! 请将此路径替换为您要测试的图片路径 !!
    path_to_your_image = '/root/autodl-tmp/mobileNetV3-plant/plant-dataset/Train/Healthy/851aa29da98f734b.jpg'

    check_image_size(path_to_your_image)
