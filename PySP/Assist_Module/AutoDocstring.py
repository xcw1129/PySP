"""
# AutoDocstring
自动文档字符串生成器模块, 用于扫描Python文件并生成模块级注释文档

## 内容
    - class:
        1. DocstringGenerator: 自动文档字符串生成器类，用于扫描Python文件并生成模块级注释文档
    - function:
        1. extract_docstring: 提取函数或类的首段注释文档
        2. generate_module_docstring: 生成指定文件的模块级注释文档
        3. update_file_docstring: 更新文件内容中的模块注释
        4. auto_update_target_module: 自动更新本文件开头指定的目标模块文件的模块注释（支持多个文件）
"""




import ast
import os
import re
from typing import Dict, List, Tuple, Optional

# ----------- 需操作的目标模块文件名及描述（字典形式，键为文件名，值为描述） -----------
TARGET_MODULES = {
    "AutoDocstring.py": "自动文档字符串生成器模块, 用于扫描Python文件并生成模块级注释文档",
    "Signal.py": "信号数据模块, 定义了PySP库中的核心信号处理对象Signal的基本结构, 以及一些信号预处理函数",
    "Analysis.py": "分析处理方法模块, 定义了PySP库中高级分析处理方法模块的基本类结构Analysis",
    "Plot.py": "绘图方法模块, 定义了PySP库中绘图方法模块的基本类结构Plot. 提供了常用绘图方法的类和函数接口, 以及辅助插件",
    "BasicSP.py": "基础信号分析及处理方法模块",
}
# ---------------------------------------------------------------

class DocstringGenerator:
    """
    自动文档字符串生成器类，用于扫描Python文件并生成模块级注释文档
    
    参数:
    --------
    file_path : str
        需要处理的Python文件路径
        
    属性:
    --------
    file_path : str
        文件路径
    tree : ast.AST
        抽象语法树
    classes : list
        类信息列表
    functions : list
        函数信息列表
        
    方法:
    --------
    parse_file() -> None
        解析Python文件，提取类和函数信息
    generate_docstring() -> str
        生成模块级注释文档字符串
    update_file() -> None
        更新文件的模块注释
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.tree = None
        self.classes = []
        self.functions = []
        
    def parse_file(self) -> None:
        """
        解析Python文件，提取类和函数信息
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.tree = ast.parse(content)
            self.classes = []
            self.functions = []
            
            # 遍历AST节点，提取一级类和函数
            for node in ast.walk(self.tree):
                if isinstance(node, ast.ClassDef):
                    # 只处理顶级类（不在其他类或函数内部）
                    if self._is_top_level(node):
                        class_info = {
                            'name': node.name,
                            'docstring': extract_docstring(node),
                            'line': node.lineno
                        }
                        self.classes.append(class_info)
                        
                elif isinstance(node, ast.FunctionDef):
                    # 只处理顶级函数（不在类内部）
                    if self._is_top_level(node):
                        func_info = {
                            'name': node.name,
                            'docstring': extract_docstring(node),
                            'line': node.lineno
                        }
                        self.functions.append(func_info)
                        
        except Exception as e:
            print(f"解析文件时出错: {e}")
            
    def _is_top_level(self, node) -> bool:
        """
        判断节点是否为顶级节点（不在其他类或函数内部）
        """
        for parent in ast.walk(self.tree):
            if hasattr(parent, 'body') and node in parent.body:
                # 如果父节点是Module，则为顶级节点
                return isinstance(parent, ast.Module)
        return False
    
    def generate_docstring(self, module_name: str = None, module_description: str = None) -> str:
        """
        生成模块级注释文档字符串
        
        参数:
        --------
        module_name : str, 可选
            模块名称，默认从文件名提取
        module_description : str, 可选
            模块描述，默认生成通用描述
            
        返回:
        --------
        docstring : str
            生成的模块注释字符串
        """
        if module_name is None:
            module_name = os.path.splitext(os.path.basename(self.file_path))[0]
            
        if module_description is None:
            module_description = f"{module_name}模块的相关功能和方法"
            
        # 构建注释文档
        docstring_parts = [
            '"""',
            f'# {module_name}',
            module_description,
            '',
            '## 内容'
        ]
        
        # 添加类信息
        if self.classes:
            docstring_parts.append('    - class:')
            for i, class_info in enumerate(self.classes, 1):
                class_desc = self._extract_brief_description(class_info['docstring'])
                docstring_parts.append(f'        {i}. {class_info["name"]}: {class_desc}')
                
        # 添加函数信息
        if self.functions:
            docstring_parts.append('    - function:')
            for i, func_info in enumerate(self.functions, 1):
                func_desc = self._extract_brief_description(func_info['docstring'])
                docstring_parts.append(f'        {i}. {func_info["name"]}: {func_desc}')
                
        docstring_parts.append('"""')
        
        return '\n'.join(docstring_parts)
    
    def _extract_brief_description(self, docstring: str) -> str:
        """
        从docstring中提取简要描述（第一行非空内容）
        """
        if not docstring:
            return "待添加描述"
            
        # 清理docstring，移除三引号
        clean_docstring = docstring.strip().strip('"""').strip("'''").strip()
        
        if not clean_docstring:
            return "待添加描述"
            
        # 获取第一行非空内容
        lines = clean_docstring.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                return line
                
        return "待添加描述"
    
    def update_file(self, module_name: str = None, module_description: str = None) -> None:
        """
        更新文件的模块注释
        
        参数:
        --------
        module_name : str, 可选
            模块名称
        module_description : str, 可选
            模块描述
        """
        try:
            # 先解析文件
            self.parse_file()
            
            # 生成新的docstring
            new_docstring = self.generate_docstring(module_name, module_description)
            
            # 读取原文件内容
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 更新文件内容
            updated_content = update_file_docstring(content, new_docstring)
            
            # 写回文件
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
                
            print(f"成功更新 {self.file_path} 的模块注释")
            
        except Exception as e:
            print(f"更新文件时出错: {e}")


def extract_docstring(node) -> str:
    """
    提取函数或类的首段注释文档
    
    参数:
    --------
    node : ast.AST
        AST节点（ClassDef或FunctionDef）
        
    返回:
    --------
    docstring : str
        提取的文档字符串，如果没有则返回空字符串
    """
    if (hasattr(node, 'body') and 
        node.body and 
        isinstance(node.body[0], ast.Expr) and 
        isinstance(node.body[0].value, ast.Constant)):
        return node.body[0].value.value
    return ""


def generate_module_docstring(file_path: str, module_name: str = None, 
                            module_description: str = None) -> str:
    """
    生成指定文件的模块级注释文档
    
    参数:
    --------
    file_path : str
        Python文件路径
    module_name : str, 可选
        模块名称，默认从文件名提取
    module_description : str, 可选
        模块描述
        
    返回:
    --------
    docstring : str
        生成的模块注释字符串
    """
    generator = DocstringGenerator(file_path)
    generator.parse_file()
    return generator.generate_docstring(module_name, module_description)


def update_file_docstring(content: str, new_docstring: str) -> str:
    """
    更新文件内容中的模块注释
    
    参数:
    --------
    content : str
        原文件内容
    new_docstring : str
        新的模块注释字符串
        
    返回:
    --------
    updated_content : str
        更新后的文件内容
    """
    # 使用正则表达式查找现有的模块级docstring
    # 模式：文件开头的三引号注释
    pattern = r'^(\s*#.*?\n)*\s*""".*?"""\s*\n'
    match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
    
    if match:
        # 如果找到现有的docstring，替换它
        # 保留文件路径注释
        lines = content.split('\n')
        filepath_comment = ""
        start_idx = 0
        
        # 查找filepath注释
        for i, line in enumerate(lines):
            if line.strip().startswith('# filepath:'):
                filepath_comment = line + '\n'
                start_idx = i + 1
                break
                
        # 找到docstring结束位置
        in_docstring = False
        docstring_end = start_idx
        for i, line in enumerate(lines[start_idx:], start_idx):
            if '"""' in line:
                if not in_docstring:
                    in_docstring = True
                else:
                    docstring_end = i + 1
                    break
                    
        # 重构内容
        before_docstring = '\n'.join(lines[:start_idx])
        after_docstring = '\n'.join(lines[docstring_end:])
        
        if filepath_comment:
            return filepath_comment + new_docstring + '\n\n' + after_docstring
        else:
            return new_docstring + '\n\n' + after_docstring
    else:
        # 如果没有找到现有的docstring，在文件开头添加
        lines = content.split('\n')
        
        # 查找是否有filepath注释
        if lines and lines[0].strip().startswith('# filepath:'):
            return lines[0] + '\n' + new_docstring + '\n\n' + '\n'.join(lines[1:])
        else:
            return new_docstring + '\n\n' + content

def auto_update_target_module():
    """
    自动更新本文件开头指定的目标模块文件的模块注释（支持多个文件）
    """
    current_dir = os.path.dirname(__file__)
    for filename, module_desc in TARGET_MODULES.items():
        target_path = os.path.join(current_dir, filename)
        if not os.path.exists(target_path):
            print(f"目标模块文件不存在: {target_path}")
            continue

        module_name = os.path.splitext(os.path.basename(target_path))[0]
        generator = DocstringGenerator(target_path)
        generator.update_file(module_name, module_desc)

# 使用示例
if __name__ == "__main__":
    # 自动更新本文件开头指定的目标模块文件
    auto_update_target_module()