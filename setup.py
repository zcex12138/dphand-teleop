from setuptools import setup, find_packages

setup(
    name="dphandteleop",       # 包名（pip list 显示的名称）
    version="0.1.0",           # 版本号
    packages=find_packages(),  # 自动发现所有包（包含 __init__.py 的目录）
    include_package_data=True, # 包含非代码文件（如 assets/data）
    install_requires=[         # 依赖项（可替换为 requirements.txt）
        "avp_stream==1.0",
        "gymnasium==1.1.1",
        "mujoco==3.3.0",
        "numpy==2.2.4",
        "scipy==1.15.2",
    ],
    python_requires=">=3.6",  # Python 版本要求
    author="Yu Hexi",        # 作者信息（可选）
    description="Teleoperation for DPhand",
)