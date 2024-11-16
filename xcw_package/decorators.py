from .dependencies import np
from .dependencies import wraps
from .dependencies import inspect


def Check_Params(*var_checks):
    # 根据指定的变量名和维度进行对应的装饰器检查
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数参数名和对应的值
            params = inspect.signature(func)
            bound_args = params.bind(*args, **kwargs)
            bound_args.apply_defaults()
            # 按指定方式检查指定的变量
            for var_name, expected_dim in var_checks:
                data = bound_args.arguments.get(var_name)
                if data is not None:
                    if not isinstance(data, np.ndarray):
                        raise ValueError(
                            f"输入变量 '{var_name}' 不是array数组, 可能导致后续计算错误"
                        )
                    if data.ndim != expected_dim:
                        raise ValueError(
                            f"输入变量 '{var_name}' 维度不为 {expected_dim}, 不符合算法要求"
                        )

            return func(*args, **kwargs)
        return wrapper
    return decorator
