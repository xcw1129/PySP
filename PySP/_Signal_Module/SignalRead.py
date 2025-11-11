"""
数据集文件夹结构扫描工具。

扩展 DataSet 类，实现数据集结构对象化表示。
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Set, Optional, List
from concurrent.futures import ThreadPoolExecutor
from numpy import random
from IPython.display import display
from copy import deepcopy


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
# 数据文件管理类
class Files:

    def __init__(self, names: List[str], root: str, type: str):
        self.filepaths = []  # 使用列表管理保留文件顺序
        typeList = list(Files.show_read_params().keys())
        if type not in typeList:
            raise ValueError(f"不支持的文件类型: {type}，仅支持: {typeList}")
        else:
            self.filetype = type
        self.rootpath = Path(root)
        if not self.rootpath.exists() or not self.rootpath.is_dir():
            raise ValueError(f"指定的root路径: {root} 不存在或不是文件夹")
        # 路径生成与扩展名检查合并
        for filename in names:
            fp = self.rootpath / filename
            if fp.suffix.lower() != self.filetype:
                print(
                    f"文件: {filename} 的类型与指定的filetype: {self.filetype} 不一致"
                )
                continue
            if not fp.exists() or not fp.is_file():
                print(f"文件: {filename} 不存在于路径: {self.rootpath} 下")
                continue
            self.filepaths.append(fp)

    # --------------------------------------------------------------------------------#
    # Files类Python特性支持
    def __len__(self):
        return len(self.filepaths)

    def __iter__(self):
        for fp in self.filepaths:
            yield fp

    @property
    def names(self) -> List[str]:
        return [fp.name for fp in self.filepaths]

    def __getitem__(self, item) -> "Files":
        existing_filenames = [fp.name for fp in self.filepaths]
        rootpath_str = str(self.rootpath)
        # 支持整数/切片索引
        if isinstance(item, (int, slice)):
            select_filenames = existing_filenames[item]
            if isinstance(select_filenames, str):
                select_filenames = [select_filenames]
            return Files(select_filenames, rootpath_str, self.filetype)
        # 支持文件名字符串/字符串列表
        elif isinstance(item, (str, list)):
            # 检查文件是否存在
            select_filenames = []
            item = [item] if isinstance(item, str) else item
            for filename in item:
                if filename not in existing_filenames:
                    raise IndexError(f"文件: {filename} 不存在于Files对象中")
                select_filenames.append(filename)
            return Files(select_filenames, rootpath_str, self.filetype)
        else:
            raise TypeError("Files仅支持整数、切片、字符串和字符串列表索引")

    def __repr__(self) -> str:
        return f"Files(x{len(self.filepaths)} {self.filetype} [{str(self.rootpath)}])"

    # --------------------------------------------------------------------------------#
    # 数据加载等外部常用接口
    def load(
        self,
        merge: bool = False,
        isParallel: bool = False,
        parallelCores: Optional[int] = None,
    ):
        if self.filetype == ".csv":
            dfs = Files._read_csv_batch(self.filepaths, isParallel, parallelCores)
        else:
            raise NotImplementedError(f"暂不支持 {self.filetype} 文件读取")
        if not dfs:
            return pd.DataFrame()
        if merge:  # 并排合并为单个DataFrame返回
            all_df = pd.concat(dfs, axis=1)
        else:  # 转为字典返回
            all_df: Dict[str, pd.DataFrame] = {}
            for fp, df in zip(self.filepaths, dfs):
                prefix = fp.name
                if len(df.columns) > 0:
                    df.columns = [
                        col.split("/", 1)[1] if "/" in col else col
                        for col in df.columns
                    ]
                all_df[prefix] = df
        return all_df

    def example(self, num: int = 1, **kwargs) -> None:
        if not self.filepaths:
            print("Files对象为空，无可预览文件。")
            return
        n = min(num, len(self.filepaths))
        # 随机n抽样预览
        sample_filepaths = random.choice(self.filepaths, n, replace=False)
        for fp in sample_filepaths:
            print(f"\n---\n预览文件: {fp}\n---")
            if self.filetype == ".csv":
                df = Files._read_csv_once(fp, **kwargs)
                try:
                    display(df)
                finally:
                    pass

    def save(self):
        all_df = self.load(merge=True, isParallel=True)
        # 保持为excel格式, 便于软件打开查看
        all_df.to_excel(self.rootpath / "merged_result.csv", index=False)

    # --------------------------------------------------------------------------------#
    # 数据文件读取内部方法
    _read_params = {
        ".csv": {},
        ".txt": {},
        ".xlsx": {},
        ".mat": {},
    }  # 数据读取全局参数（私有）

    @staticmethod
    def show_read_params() -> Dict:
        return deepcopy(Files._read_params)

    @staticmethod
    def set_read_params(filetype: str, **kwargs) -> None:
        typeList = list(Files._read_params.keys())
        if filetype not in typeList:
            raise ValueError(f"不支持的文件类型: {filetype}，仅支持: {typeList}")
        Files._read_params[filetype].update(kwargs)

    @staticmethod
    def clean_read_params(filetype: str) -> None:
        typeList = list(Files._read_params.keys())
        if filetype not in typeList:
            raise ValueError(f"不支持的文件类型: {filetype}，仅支持: {typeList}")
        Files._read_params[filetype] = {}

    @staticmethod
    def _read_csv_once(fp: Path, **kwargs) -> pd.DataFrame:
        csv_read_params = Files._read_params[".csv"].copy()
        csv_read_params.update(kwargs)
        # drop_cols_idx: 按列索引删除指定列
        drop_cols_idx = csv_read_params.pop("drop_cols_idx", None)
        # 执行读取操作
        try:
            df = pd.read_csv(fp, **csv_read_params)
        except Exception as e:
            print(f"读取CSV文件: {fp} 失败，错误信息: {e}")
            return pd.DataFrame()
        # 读取结果后处理
        if drop_cols_idx:
            cols_to_drop = [df.columns[i] for i in drop_cols_idx if i < len(df.columns)]
            df = df.drop(columns=cols_to_drop, errors="ignore")
        return df

    @staticmethod
    def _read_csv_batch(
        filepaths: List[Path],
        isParallel: bool = False,
        parallelCores: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        if isParallel:  # 多线程读取
            max_workers = (
                parallelCores
                if parallelCores is not None
                else min(os.cpu_count() * 2, len(filepaths))  # 默认2倍CPU核数线程
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                dfs = list(executor.map(Files._read_csv_once, filepaths))
        else:
            dfs = [Files._read_csv_once(fp) for fp in filepaths]
        # 为避免列名冲突，添加文件名前缀
        for i, df in enumerate(dfs):
            prefix = filepaths[i].stem
            df.columns = [f"{prefix}/{col}" for col in df.columns]
        return dfs


# --------------------------------------------------------------------------------------------#
# 数据集结构管理类
class Dataset:

    def __init__(self, root: str, type: str = "csv", label: str = "") -> None:
        self.rootpath = Path(root)
        if not self.rootpath.exists() or not self.rootpath.is_dir():
            raise ValueError(f"指定的root路径: {root} 不存在或不是文件夹")
        self.filetype = Dataset._standardize_filetype(type)
        self.label = label
        self.content = Dataset._get_dataset_structure(str(self.rootpath), self.filetype)
        if self.content == {}:
            raise ValueError(
                f"[{self.label}] {self.rootpath} 下未找到格式为 {self.filetype} 的文件。"
            )
        self.content = Dataset._convert_Files(
            self.content, str(self.rootpath), self.filetype
        )

    # --------------------------------------------------------------------------------#
    # Dataset类Python特性支持
    def __getitem__(self, key: str) -> Dict:
        # 允许多级索引
        return self.content[key]

    # --------------------------------------------------------------------------------#
    # 数据集读写转换等外部常用接口
    def refresh(self) -> None:
        structure = Dataset._get_dataset_structure(str(self.rootpath), self.filetype)
        self.content = Dataset._convert_Files(
            structure, str(self.rootpath), self.filetype
        )

    def info(self) -> None:
        print(f"[{self.label}] rootpath: {self.rootpath}")
        print(f"[{self.label}] filetype: {self.filetype}")
        Dataset._print_tree(self.content, depth=1)

    # --------------------------------------------------------------------------------#
    # 数据集结构字典处理内部方法
    @staticmethod
    def _print_tree(node: Dict, depth: int) -> None:
        indent = "    " * depth
        folders = sorted(key for key in node.keys() if key != "Files")
        for folder in folders:
            print(f"{indent}{folder}\\")
            Dataset._print_tree(node[folder], depth + 1)
        files = node.get("Files")
        if files:
            print(f"{indent}{files}")

    @staticmethod
    def _convert_Files(node: Dict, folder: str, type: str) -> Dict:
        out = {}
        for label, content in node.items():
            if label == "Files":
                out[label] = Files(content, folder, type)
            elif isinstance(content, Dict):  # 递归转换子目录
                out[label] = Dataset._convert_Files(content, folder + "/" + label, type)
            else:
                out[label] = content
        return out

    # --------------------------------------------------------------------------------#
    # windows文件操作内部方法
    @staticmethod
    def _standardize_filetype(filetype: str) -> str:
        ext = filetype.lower()
        if not ext.startswith("."):
            ext = "." + ext
        # 常见传感器时序数据格式
        supported = list(Files.show_read_params().keys())
        if ext not in supported:
            raise ValueError(f"文件类型: {ext} 通常不为数据文件格式: {supported}")
        return ext

    @staticmethod
    def _get_dataset_structure(rootpath: str, filetype: str = "csv") -> Dict:
        # 路径与类型检查
        base = Path(rootpath)
        if not base.exists():
            raise FileNotFoundError(f"未找到路径对应的文件夹: {rootpath}")
        if not base.is_dir():
            raise NotADirectoryError(f"非文件夹路径: {rootpath}")
        ext = Dataset._standardize_filetype(filetype)

        # 扫描所有包含目标文件的目录，记录相对路径和文件名列表
        contains_dirs: Set[Path] = set()
        files_in_dir = {}
        for dirpath, _, filenames in os.walk(base):
            # 收集目标类型文件名:　默认文件名字符串排序
            filelist = [fn for fn in filenames if Path(fn).suffix.lower() == ext]
            # 解析收集文件
            if filelist:
                rel = Path(dirpath).relative_to(base)  # 计算相对路径
                if str(rel) == ".":
                    contains_dirs.add(Path("."))  # 根目录
                else:
                    contains_dirs.add(rel)
                files_in_dir[rel] = filelist  # 记录该目录下所有目标文件

        # 若无包含目标文件的目录，返回空字典
        if not contains_dirs:
            return {}

        # 构建嵌套字典树，顶层为数据集根目录
        contentTree: Dict = {}
        # 先插入所有目录节点，确保父节点先于子节点
        for rel in sorted(
            contains_dirs, key=lambda p: len(p.parts) if p != Path(".") else 0
        ):
            if rel == Path("."):
                continue
            cur = contentTree
            for part in rel.parts:
                cur = cur.setdefault(part, {})  # 递归插入子目录
        # 插入文件列表到对应目录
        for rel, filelist in files_in_dir.items():
            if rel == Path("."):
                cur = contentTree
            else:
                cur = contentTree
                for part in rel.parts:
                    cur = cur.setdefault(part, {})
            cur["Files"] = filelist  # 该目录下所有目标文件
        return contentTree


__all__ = ["Files", "Dataset"]
