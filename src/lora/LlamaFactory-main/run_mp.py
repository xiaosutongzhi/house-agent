# run_mp.py
from llamafactory.train.tuner import run_exp

if __name__ == "__main__":
    # 直接调用底层训练函数，彻底绕过 torchrun 启动器
    run_exp()