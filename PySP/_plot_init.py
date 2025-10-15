from PySP._Assist_Module.Dependencies import cycler, font_manager, plt, resources

font_name = None
try:
    with resources.path("PySP._Assist_Module", "times+simsun.ttf") as font_path:
        font_manager.fontManager.addfont(str(font_path))
        prop = font_manager.FontProperties(fname=str(font_path))
        font_name = prop.get_name()
except Exception:
    pass

font_sans_serif = [font_name] if font_name else []
font_sans_serif += ["SimSun", "Microsoft YaHei", "Arial"]

# 全局配置
plt.rcParams.update(
    {
        "font.family": "sans-serif",  # 设置全局字体
        "font.sans-serif": font_sans_serif,  # 优先自定义字体
        "axes.unicode_minus": False,  # 负号正常显示
        "font.size": 18,  # 设置全局字体大小
        "axes.titlesize": 20,  # 标题字体大小
        "axes.labelsize": 18,  # 轴标签字体大小
        "xtick.labelsize": 16,  # x轴刻度标签字体大小
        "ytick.labelsize": 16,  # y轴刻度标签字体大小
        "legend.fontsize": 16,  # 图例字体大小
        "figure.figsize": (12, 5),  # 默认图形大小，12cm x 5cm
        "figure.dpi": 100,  # 显示分辨率
        "savefig.dpi": 600,  # 保存分辨率
        "axes.prop_cycle": cycler(
            color=[
                "#1f77b4",  # 蓝
                "#ff7f0e",  # 橙
                "#2ca02c",  # 绿
                "#d62728",  # 红
                "#a77ece",  # 紫
                "#8c564b",  # 棕
                "#520e8e",  # 粉
                "#7f7f7f",  # 灰
                "#bcbd22",  # 橄榄
                "#17becf",  # 青
            ]
        ),  # 设置颜色循环
        "axes.grid": True,  # 显示网格
        "axes.grid.axis": "y",  # 只显示y轴网格
        "grid.linestyle": (0, (8, 6)),  # 网格线为虚线
        "xtick.direction": "in",  # x轴刻度线朝内
        "ytick.direction": "in",  # y轴刻度线朝内
        "mathtext.fontset": "custom",  # 公式字体设置
        "mathtext.rm": "Times New Roman",  # 数学公式字体 - 正常
        "mathtext.it": "Times New Roman:italic",  # 数学公式字体 - 斜体
        "mathtext.bf": "Times New Roman:bold",  # 数学公式字体 - 粗体
    }
)
