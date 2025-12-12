#!/usr/bin/env python3
"""
测试mask生成功能的简单脚本
"""

import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_extract_classes():
    """测试类别名提取功能"""
    from generate_masks import extract_classes_from_filename
    
    test_cases = [
        ("apple_banana.png", ["apple", "banana"]),
        ("cat_dog-neg.jpg", ["cat", "dog"]),
        ("car.bmp", ["car"]),
        ("house_tree_flower-neg.tiff", ["house", "tree", "flower"])
    ]
    
    print("测试类别名提取功能:")
    for filename, expected in test_cases:
        result = extract_classes_from_filename(filename)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {filename} -> {result} (期望: {expected})")

def test_imports():
    """测试导入功能"""
    print("\n测试导入功能:")
    try:
        from generate_masks import (
            load_image, load_model, get_grounding_output,
            create_mask_image, save_mask_data
        )
        print("  ✓ 核心函数导入成功")
    except ImportError as e:
        print(f"  ✗ 导入失败: {e}")

def test_model_paths():
    """测试模型路径检查"""
    print("\n测试模型路径检查:")
    
    # 检查模型文件是否存在
    model_files = [
        "groundingdino_swint_ogc.pth",
        "sam_vit_h_4b8939.pth",
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    ]
    
    for file_path in model_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path} 存在")
        else:
            print(f"  ✗ {file_path} 不存在")

def test_directory_structure():
    """测试目录结构"""
    print("\n测试目录结构:")
    
    directories = ["original_images", "neg_images", "masks"]
    for dir_name in directories:
        if os.path.exists(dir_name):
            print(f"  ✓ {dir_name}/ 目录存在")
            # 检查目录中是否有文件
            files = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
            print(f"    包含 {len(files)} 个文件")
        else:
            print(f"  ✗ {dir_name}/ 目录不存在")

def main():
    """主测试函数"""
    print("=" * 50)
    print("Mask生成功能测试")
    print("=" * 50)
    
    # 运行各项测试
    test_extract_classes()
    test_imports()
    test_model_paths()
    test_directory_structure()
    
    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)
    
    # 给出下一步建议
    print("\n下一步建议:")
    if not os.path.exists("original_images"):
        print("1. 请先运行 generate_images.py 生成图像")
    else:
        print("1. 可以运行 generate_masks.py 生成mask")
    
    print("2. 确保已安装必要的依赖包:")
    print("   pip install torch torchvision opencv-python")
    print("   pip install groundingdino-py segment-anything")
    print("   pip install Pillow numpy")

if __name__ == "__main__":
    main()