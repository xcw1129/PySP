from .dependencies import np
from .dependencies import inspect
from .dependencies import wraps


# --------------------------------------------------------------------------------------------#
# --## ---------------------------------------------------------------------------------------#
# ------## -----------------------------------------------------------------------------------#
# ----------## -------------------------------------------------------------------------------#
def Check_Vars(*var_checks):
    # 根据指定的变量名和维度进行对应的装饰器检查
    # ---------------------------------------------------------------------------------------#
    def decorator(func):
        @wraps(func)
        # -----------------------------------------------------------------------------------#
        def wrapper(*args, **kwargs):
            # -------------------------------------------------------------------------------#
            # 获取函数输入变量
            Vars = inspect.signature(func)
            bound_args = Vars.bind(*args, **kwargs)
            bound_args.apply_defaults()
            # 获取变量的类型注解
            annotations = func.__annotations__
            var_checks_json = var_checks[0]
            # -------------------------------------------------------------------------------#
            # 按指定方式检查指定的变量
            for var_name in var_checks_json:
                var_value = bound_args.arguments.get(var_name)  # 变量实际值
                var_type = annotations.get(var_name)  # 变量预设类型
                var_cond = var_checks_json[var_name]  # 变量额外检查条件
                if var_value is not None:
                    # 检查输入值类型是否为预设类型
                    if var_type and not isinstance(var_value, var_type):
                        raise TypeError(
                            f"输入变量 '{var_name}' 类型不为 {var_type.__name__}, 实际为 {type(var_value).__name__}"
                        )
                    # 针对某些变量类型进行额外检查
                    # array类检查维度
                    if isinstance(var_value, np.ndarray):
                        # 条件1：数组维度
                        if var_value.ndim != var_cond["ndim"]:
                            raise ValueError(
                                f"输入array数组 '{var_name}' 维度不为 {var_cond['ndim']}, 实际为{var_value.ndim}"
                            )
                        # 条件2：...
                    # int类
                    # float类...
            # -------------------------------------------------------------------------------#
            return func(*args, **kwargs)  # 检查通过，执行函数

        return wrapper
        # -----------------------------------------------------------------------------------#

    return decorator
    # ---------------------------------------------------------------------------------------#
