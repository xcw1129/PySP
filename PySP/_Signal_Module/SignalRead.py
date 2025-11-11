"""
# SignalRead
数据文件与数据集结构管理模块

## 内容
    - class:
        1. Files: 数据文件批量管理与读取类，支持多种格式文件的批量加载、预览、保存
        2. Dataset: 数据集文件夹结构对象化管理类，递归扫描并组织多层级数据集
"""

from PySP._Assist_Module.Dependencies import (
    Dict,
    List,
    Optional,
    Path,
    Set,
    ThreadPoolExecutor,
    Union,
    deepcopy,
    os,
    pd,
    random,
)


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
class Files:
    """
    数据文件批量管理与读取类，支持多种格式文件的批量加载、预览、保存

    Parameters
    ----------
    names : List[str]
        文件名列表
    root : str
        文件所在根目录路径
    type : str
        文件类型（扩展名），如 'csv', 'txt', 'xlsx', 'mat'

    Attributes
    ----------
    filepaths : List[Path]
        有效文件的完整路径列表
    filetype : str
        文件类型（标准化扩展名）
    rootpath : Path
        根目录路径

    Methods
    -------
    load(merge=False, isParallel=False, parallelCores=None)
        批量加载文件为DataFrame或字典
    example(num=1, **kwargs)
        随机预览部分文件内容
    save()
        合并保存所有文件为Excel
    show_read_params()
        显示各类型文件的读取参数
    set_read_params(filetype, **kwargs)
        设置指定类型文件的读取参数
    clean_read_params(filetype)
        清空指定类型文件的读取参数
    """

    def __init__(self, names: List[str], root: str, type: str):
        self.filepaths = []  # 使用列表管理保留文件顺序
        self.filetype = Files._standardize_filetype(type)
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
    # Python特性支持
    def __len__(self):
        """返回文件数量"""
        return len(self.filepaths)

    def __iter__(self):
        """迭代器，遍历文件路径"""
        for fp in self.filepaths:
            yield fp

    @property
    def names(self) -> List[str]:
        """所有文件名列表"""
        return [fp.name for fp in self.filepaths]

    def __getitem__(self, item) -> "Files":
        """支持整数/切片/文件名/文件名列表索引，返回新的Files对象"""
        existing_filenames = [fp.name for fp in self.filepaths]
        rootpath_str = str(self.rootpath)
        if isinstance(item, (int, slice)):
            select_filenames = existing_filenames[item]
            if isinstance(select_filenames, str):
                select_filenames = [select_filenames]
            return Files(select_filenames, rootpath_str, self.filetype)
        elif isinstance(item, (str, list)):
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
    # 外部常用接口
    def load(
        self,
        merge: bool = False,
        isParallel: bool = False,
        parallelCores: Optional[int] = None,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        批量加载文件为DataFrame或字典

        Parameters
        ----------
        merge : bool, default: False
            是否将所有文件按列合并为单个DataFrame返回
        isParallel : bool, default: False
            是否并行读取
        parallelCores : int, optional
            并行读取时的线程数

        Returns
        -------
        all_df : pd.DataFrame or Dict[str, pd.DataFrame]
            合并时为单个DataFrame，否则为文件名到DataFrame的字典
        """
        if self.filetype == ".csv":
            dfs = Files._read_csv_batch(self.filepaths, isParallel, parallelCores)
        elif self.filetype == ".txt":
            dfs = Files._read_txt_batch(self.filepaths, isParallel, parallelCores)
        elif self.filetype == ".xlsx":
            dfs = Files._read_xlsx_batch(self.filepaths, isParallel, parallelCores)
        elif self.filetype == ".mat":
            dfs = Files._read_mat_batch(self.filepaths, isParallel, parallelCores)
        else:
            raise NotImplementedError(f"暂不支持 {self.filetype} 文件读取")
        if not dfs:
            return pd.DataFrame()
        if merge:
            all_df = pd.concat(dfs, axis=1)
        else:
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
        """
        随机抽样预览部分文件内容

        Parameters
        ----------
        num : int, default: 1
            预览文件数量
        **kwargs :
            传递给读取函数的参数
        """
        if not self.filepaths:
            print("Files对象为空，无可预览文件。")
            return
        n = min(num, len(self.filepaths))
        sample_filepaths = random.choice(self.filepaths, n, replace=False)
        for fp in sample_filepaths:
            print(f"\n---\n预览文件: {fp}\n---")
            if self.filetype == ".csv":
                df = Files._read_csv_once(fp, **kwargs)
                try:
                    from IPython.display import display

                    display(df)
                finally:
                    pass

    def save(self) -> None:
        """合并保存所有文件为Excel文件"""
        all_df = self.load(merge=True, isParallel=True)
        all_df.to_excel(self.rootpath / "merged_result.csv", index=False)

    # --------------------------------------------------------------------------------#
    # 数据文件读取参数与内部方法
    _read_params = {
        ".csv": {},
        ".txt": {"sep": "\t"},  # 默认tab分隔
        ".xlsx": {},
        ".mat": {"variable": None},  # variable: 指定mat变量名
    }  # 数据读取全局参数（私有）

    @staticmethod
    def _standardize_filetype(filetype: str) -> str:
        """标准化文件类型扩展名"""
        ext = filetype.lower()
        if not ext.startswith("."):
            ext = "." + ext
        supported = list(Files.show_read_params().keys())
        if ext not in supported:
            raise ValueError(f"文件类型: {ext} 通常不为数据文件格式: {supported}")
        return ext

    @staticmethod
    def show_read_params() -> Dict:
        """显示所有支持文件类型的读取参数"""
        return deepcopy(Files._read_params)

    @staticmethod
    def set_read_params(filetype: str, **kwargs) -> None:
        """设置指定类型文件的读取参数"""
        typeList = list(Files._read_params.keys())
        if filetype not in typeList:
            raise ValueError(f"不支持的文件类型: {filetype}，仅支持: {typeList}")
        Files._read_params[filetype].update(kwargs)

    @staticmethod
    def clean_read_params(filetype: str) -> None:
        """清空指定类型文件的读取参数"""
        typeList = list(Files._read_params.keys())
        if filetype not in typeList:
            raise ValueError(f"不支持的文件类型: {filetype}，仅支持: {typeList}")
        Files._read_params[filetype] = {}

    @staticmethod
    def _read_txt_once(fp: Path, **kwargs) -> pd.DataFrame:
        """读取单个TXT文件为DataFrame"""
        txt_read_params = Files._read_params[".txt"].copy()
        txt_read_params.update(kwargs)
        drop_cols_idx = txt_read_params.pop("drop_cols_idx", None)
        try:
            df = pd.read_csv(fp, **txt_read_params)
        except Exception as e:
            print(f"读取TXT文件: {fp} 失败，错误信息: {e}")
            return pd.DataFrame()
        if drop_cols_idx:
            cols_to_drop = [df.columns[i] for i in drop_cols_idx if i < len(df.columns)]
            df = df.drop(columns=cols_to_drop, errors="ignore")
        return df

    @staticmethod
    def _read_txt_batch(
        filepaths: List[Path],
        isParallel: bool = False,
        parallelCores: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        """批量读取TXT文件"""
        if isParallel:
            max_workers = (
                parallelCores
                if parallelCores is not None
                else min(os.cpu_count() * 2, len(filepaths))
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                dfs = list(executor.map(Files._read_txt_once, filepaths))
        else:
            dfs = [Files._read_txt_once(fp) for fp in filepaths]
        for i, df in enumerate(dfs):
            prefix = filepaths[i].stem
            df.columns = [f"{prefix}/{col}" for col in df.columns]
        return dfs

    @staticmethod
    def _read_xlsx_once(fp: Path, **kwargs) -> pd.DataFrame:
        """读取单个XLSX文件为DataFrame"""
        xlsx_read_params = Files._read_params[".xlsx"].copy()
        xlsx_read_params.update(kwargs)
        drop_cols_idx = xlsx_read_params.pop("drop_cols_idx", None)
        try:
            df = pd.read_excel(fp, **xlsx_read_params)
        except Exception as e:
            print(f"读取XLSX文件: {fp} 失败，错误信息: {e}")
            return pd.DataFrame()
        if drop_cols_idx:
            cols_to_drop = [df.columns[i] for i in drop_cols_idx if i < len(df.columns)]
            df = df.drop(columns=cols_to_drop, errors="ignore")
        return df

    @staticmethod
    def _read_xlsx_batch(
        filepaths: List[Path],
        isParallel: bool = False,
        parallelCores: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        """批量读取XLSX文件"""
        if isParallel:
            max_workers = (
                parallelCores
                if parallelCores is not None
                else min(os.cpu_count() * 2, len(filepaths))
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                dfs = list(executor.map(Files._read_xlsx_once, filepaths))
        else:
            dfs = [Files._read_xlsx_once(fp) for fp in filepaths]
        for i, df in enumerate(dfs):
            prefix = filepaths[i].stem
            df.columns = [f"{prefix}/{col}" for col in df.columns]
        return dfs

    @staticmethod
    def _read_mat_once(fp: Path, **kwargs) -> pd.DataFrame:
        """读取单个MAT文件为DataFrame"""
        try:
            from scipy.io import loadmat
        except ImportError:
            print("请先安装scipy以支持mat文件读取。")
            return pd.DataFrame()
        mat_read_params = Files._read_params[".mat"].copy()
        mat_read_params.update(kwargs)
        variable = mat_read_params.pop("variable", None)
        try:
            mat = loadmat(fp)
        except Exception as e:
            print(f"读取MAT文件: {fp} 失败，错误信息: {e}")
            return pd.DataFrame()
        if variable is not None:
            if variable not in mat:
                print(f"MAT文件: {fp} 不包含变量: {variable}")
                return pd.DataFrame()
            arr = mat[variable]
        else:
            user_vars = [k for k in mat.keys() if not k.startswith("__")]
            if not user_vars:
                print(f"MAT文件: {fp} 未找到有效变量")
                return pd.DataFrame()
            arr = mat[user_vars[0]]
        try:
            df = pd.DataFrame(arr)
        except Exception as e:
            print(f"MAT变量转DataFrame失败: {e}")
            return pd.DataFrame()
        return df

    @staticmethod
    def _read_mat_batch(
        filepaths: List[Path],
        isParallel: bool = False,
        parallelCores: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        """批量读取MAT文件"""
        if isParallel:
            max_workers = (
                parallelCores
                if parallelCores is not None
                else min(os.cpu_count() * 2, len(filepaths))
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                dfs = list(executor.map(Files._read_mat_once, filepaths))
        else:
            dfs = [Files._read_mat_once(fp) for fp in filepaths]
        for i, df in enumerate(dfs):
            prefix = filepaths[i].stem
            df.columns = [f"{prefix}/{col}" for col in df.columns]
        return dfs

    @staticmethod
    def _read_csv_once(fp: Path, **kwargs) -> pd.DataFrame:
        """读取单个CSV文件为DataFrame"""
        csv_read_params = Files._read_params[".csv"].copy()
        csv_read_params.update(kwargs)
        drop_cols_idx = csv_read_params.pop("drop_cols_idx", None)
        try:
            df = pd.read_csv(fp, **csv_read_params)
        except Exception as e:
            print(f"读取CSV文件: {fp} 失败，错误信息: {e}")
            return pd.DataFrame()
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
        """批量读取CSV文件"""
        if isParallel:
            max_workers = (
                parallelCores
                if parallelCores is not None
                else min(os.cpu_count() * 2, len(filepaths))
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                dfs = list(executor.map(Files._read_csv_once, filepaths))
        else:
            dfs = [Files._read_csv_once(fp) for fp in filepaths]
        for i, df in enumerate(dfs):
            prefix = filepaths[i].stem
            df.columns = [f"{prefix}/{col}" for col in df.columns]
        return dfs


# --------------------------------------------------------------------------------------------#
class Dataset:
    """
    数据集文件夹结构对象化管理类，递归扫描并组织多层级数据集

    Parameters
    ----------
    root : str
        数据集根目录路径
    type : str, default: 'csv'
        文件类型（扩展名）
    label : str, optional
        数据集标签

    Attributes
    ----------
    rootpath : Path
        数据集根目录路径
    filetype : str
        文件类型（标准化扩展名）
    label : str
        数据集标签
    content : dict
        多层级嵌套的目录-文件结构（Files对象递归嵌套）

    Methods
    -------
    refresh()
        重新扫描数据集结构
    info()
        打印数据集结构树
    """

    def __init__(self, root: str, type: str = "csv", label: str = "") -> None:
        self.rootpath = Path(root)
        if not self.rootpath.exists() or not self.rootpath.is_dir():
            raise ValueError(f"指定的root路径: {root} 不存在或不是文件夹")
        self.filetype = Files._standardize_filetype(type)
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
    # Python特性支持
    def __getitem__(self, key: str) -> Dict:
        """支持多级索引访问子目录或Files对象"""
        return self.content[key]

    # --------------------------------------------------------------------------------#
    # 外部常用接口
    def refresh(self) -> None:
        """重新扫描数据集结构"""
        structure = Dataset._get_dataset_structure(str(self.rootpath), self.filetype)
        self.content = Dataset._convert_Files(
            structure, str(self.rootpath), self.filetype
        )

    def info(self) -> None:
        """打印数据集结构树"""
        print(f"[{self.label}] rootpath: {self.rootpath}")
        print(f"[{self.label}] filetype: {self.filetype}")
        Dataset._print_tree(self.content, depth=1)

    # --------------------------------------------------------------------------------#
    # 内部方法
    @staticmethod
    def _print_tree(node: Dict, depth: int) -> None:
        """递归打印目录树结构"""
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
        """递归将文件名列表转为Files对象"""
        out = {}
        for label, content in node.items():
            if label == "Files":
                out[label] = Files(content, folder, type)
            elif isinstance(content, Dict):
                out[label] = Dataset._convert_Files(content, folder + "/" + label, type)
            else:
                out[label] = content
        return out

    # --------------------------------------------------------------------------------#
    # windows文件操作内部方法
    @staticmethod
    def _get_dataset_structure(rootpath: str, filetype: str = "csv") -> Dict:
        """递归扫描文件夹，构建多层级数据集结构树"""
        base = Path(rootpath)
        if not base.exists():
            raise FileNotFoundError(f"未找到路径对应的文件夹: {rootpath}")
        if not base.is_dir():
            raise NotADirectoryError(f"非文件夹路径: {rootpath}")
        ext = Dataset._standardize_filetype(filetype)

        contains_dirs: Set[Path] = set()
        files_in_dir = {}
        for dirpath, _, filenames in os.walk(base):
            filelist = [fn for fn in filenames if Path(fn).suffix.lower() == ext]
            if filelist:
                rel = Path(dirpath).relative_to(base)
                if str(rel) == ".":
                    contains_dirs.add(Path("."))
                else:
                    contains_dirs.add(rel)
                files_in_dir[rel] = filelist

        if not contains_dirs:
            return {}

        contentTree: Dict = {}
        for rel in sorted(
            contains_dirs, key=lambda p: len(p.parts) if p != Path(".") else 0
        ):
            if rel == Path("."):
                continue
            cur = contentTree
            for part in rel.parts:
                cur = cur.setdefault(part, {})
        for rel, filelist in files_in_dir.items():
            if rel == Path("."):
                cur = contentTree
            else:
                cur = contentTree
                for part in rel.parts:
                    cur = cur.setdefault(part, {})
            cur["Files"] = filelist
        return contentTree


# --------------------------------------------------------------------------------------------#
__all__ = ["Files", "Dataset"]
