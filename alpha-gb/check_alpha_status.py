"""
Alpha表达式验证器
用于验证WorldQuant BRAIN alpha表达式的语法和语义正确性
"""

import re
import json
import pandas as pd
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass


# ============================================================================
# 第一部分：词法分析器（Tokenizer）
# ============================================================================

class TokenType(Enum):
    """Token类型枚举"""
    # 字面量
    NUMBER = "NUMBER"           # 123, 0.5, 1.23
    STRING = "STRING"           # "gaussian", "NAN"
    BOOL = "BOOL"              # true, false
    NAN = "NAN"                # nan

    # 标识符
    IDENTIFIER = "IDENTIFIER"   # close, open, rank, my_var

    # 运算符
    PLUS = "PLUS"              # +
    MINUS = "MINUS"            # -
    MULTIPLY = "MULTIPLY"      # *
    DIVIDE = "DIVIDE"          # /
    LESS = "LESS"              # <
    LESS_EQUAL = "LESS_EQUAL"  # <=
    GREATER = "GREATER"        # >
    GREATER_EQUAL = "GREATER_EQUAL"  # >=
    EQUAL = "EQUAL"            # ==
    NOT_EQUAL = "NOT_EQUAL"    # !=

    # 分隔符
    LPAREN = "LPAREN"          # (
    RPAREN = "RPAREN"          # )
    COMMA = "COMMA"            # ,
    SEMICOLON = "SEMICOLON"    # ;
    ASSIGN = "ASSIGN"          # =

    # 特殊
    EOF = "EOF"                # 结束符


@dataclass
class Token:
    """Token数据类"""
    type: TokenType
    value: Any
    position: int = 0


class Tokenizer:
    """词法分析器"""

    def __init__(self, expression: str):
        self.expression = expression
        self.pos = 0
        self.current_char = expression[0] if expression else None

    def error(self, msg: str):
        """抛出词法错误"""
        raise SyntaxError(f"Lexical error at position {self.pos}: {msg}")

    def advance(self):
        """前进到下一个字符"""
        self.pos += 1
        if self.pos < len(self.expression):
            self.current_char = self.expression[self.pos]
        else:
            self.current_char = None

    def peek(self, offset: int = 1) -> Optional[str]:
        """向前查看字符，不移动位置"""
        peek_pos = self.pos + offset
        if peek_pos < len(self.expression):
            return self.expression[peek_pos]
        return None

    def skip_whitespace(self):
        """跳过空白字符"""
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def read_number(self) -> Token:
        """读取数字（整数或浮点数）"""
        start_pos = self.pos
        num_str = ''

        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            num_str += self.current_char
            self.advance()

        try:
            value = float(num_str) if '.' in num_str else int(num_str)
            return Token(TokenType.NUMBER, value, start_pos)
        except ValueError:
            self.error(f"Invalid number: {num_str}")

    def read_identifier(self) -> Token:
        """读取标识符或关键字"""
        start_pos = self.pos
        ident = ''

        # 标识符可以包含字母、数字、下划线
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            ident += self.current_char
            self.advance()

        # 检查是否是关键字（不区分大小写）
        ident_lower = ident.lower()
        if ident_lower == 'true':
            return Token(TokenType.BOOL, True, start_pos)
        elif ident_lower == 'false':
            return Token(TokenType.BOOL, False, start_pos)
        elif ident_lower == 'nan':
            return Token(TokenType.NAN, float('nan'), start_pos)
        else:
            return Token(TokenType.IDENTIFIER, ident, start_pos)

    def read_string(self, quote_char: str) -> Token:
        """读取字符串（支持单引号和双引号）

        Args:
            quote_char: 引号字符（' 或 "）
        """
        start_pos = self.pos
        self.advance()  # 跳过开始的引号

        string_val = ''
        while self.current_char is not None and self.current_char != quote_char:
            string_val += self.current_char
            self.advance()

        if self.current_char != quote_char:
            self.error(f"Unterminated string (expected closing {quote_char})")

        self.advance()  # 跳过结束的引号
        return Token(TokenType.STRING, string_val, start_pos)

    def tokenize(self) -> List[Token]:
        """将表达式转换为token列表"""
        tokens = []

        while self.current_char is not None:
            # 跳过空白字符
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            # 数字（包括负数）
            # 检测负数：如果是'-'后跟数字，且前面是开始/左括号/逗号，则读取为负数
            if self.current_char == '-':
                next_char = self.peek()
                if next_char and (next_char.isdigit() or next_char == '.'):
                    # 检查'-'前面的token，判断是否应该作为负号
                    if not tokens or tokens[-1].type in [TokenType.LPAREN, TokenType.COMMA]:
                        # 这是负数，不是减号
                        self.advance()  # 跳过'-'
                        num_token = self.read_number()
                        # 将数字值设为负数
                        num_token.value = -num_token.value
                        tokens.append(num_token)
                        continue

            if self.current_char.isdigit() or (self.current_char == '.' and self.peek() and self.peek().isdigit()):
                tokens.append(self.read_number())
                continue

            # 标识符或关键字
            if self.current_char.isalpha() or self.current_char == '_':
                tokens.append(self.read_identifier())
                continue

            # 字符串（支持单引号和双引号）
            if self.current_char == '"':
                tokens.append(self.read_string('"'))
                continue

            if self.current_char == "'":
                tokens.append(self.read_string("'"))
                continue

            # 双字符运算符
            if self.current_char == '<' and self.peek() == '=':
                tokens.append(Token(TokenType.LESS_EQUAL, '<=', self.pos))
                self.advance()
                self.advance()
                continue

            if self.current_char == '>' and self.peek() == '=':
                tokens.append(Token(TokenType.GREATER_EQUAL, '>=', self.pos))
                self.advance()
                self.advance()
                continue

            if self.current_char == '=' and self.peek() == '=':
                tokens.append(Token(TokenType.EQUAL, '==', self.pos))
                self.advance()
                self.advance()
                continue

            if self.current_char == '!' and self.peek() == '=':
                tokens.append(Token(TokenType.NOT_EQUAL, '!=', self.pos))
                self.advance()
                self.advance()
                continue

            # 单字符运算符和分隔符
            char_map = {
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.MULTIPLY,
                '/': TokenType.DIVIDE,
                '<': TokenType.LESS,
                '>': TokenType.GREATER,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                ',': TokenType.COMMA,
                ';': TokenType.SEMICOLON,
                '=': TokenType.ASSIGN,
            }

            if self.current_char in char_map:
                token_type = char_map[self.current_char]
                tokens.append(Token(token_type, self.current_char, self.pos))
                self.advance()
                continue

            # 未知字符
            self.error(f"Unknown character: '{self.current_char}'")

        tokens.append(Token(TokenType.EOF, None, self.pos))
        return tokens


# ============================================================================
# 第二部分：抽象语法树（AST）节点
# ============================================================================

class ASTNode:
    """AST节点基类"""
    pass


class NumberNode(ASTNode):
    """数字节点"""
    def __init__(self, value: float):
        self.value = value

    def __repr__(self):
        return f"NumberNode({self.value})"


class StringNode(ASTNode):
    """字符串节点"""
    def __init__(self, value: str):
        self.value = value

    def __repr__(self):
        return f"StringNode('{self.value}')"


class BoolNode(ASTNode):
    """布尔节点"""
    def __init__(self, value: bool):
        self.value = value

    def __repr__(self):
        return f"BoolNode({self.value})"


class NanNode(ASTNode):
    """NaN节点"""
    def __repr__(self):
        return "NanNode()"


class IdentifierNode(ASTNode):
    """标识符节点（可能是数据字段或变量）"""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"IdentifierNode('{self.name}')"


class BinaryOpNode(ASTNode):
    """二元运算符节点"""
    def __init__(self, left: ASTNode, op: str, right: ASTNode):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"BinaryOpNode({self.left} {self.op} {self.right})"


class UnaryOpNode(ASTNode):
    """一元运算符节点"""
    def __init__(self, op: str, operand: ASTNode):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"UnaryOpNode({self.op}{self.operand})"


class FunctionCallNode(ASTNode):
    """函数调用节点"""
    def __init__(self, name: str, args: List[ASTNode], kwargs: Dict[str, ASTNode]):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        args_str = ', '.join(repr(arg) for arg in self.args)
        kwargs_str = ', '.join(f"{k}={repr(v)}" for k, v in self.kwargs.items())
        all_args = ', '.join(filter(None, [args_str, kwargs_str]))
        return f"FunctionCallNode({self.name}({all_args}))"


class AssignmentNode(ASTNode):
    """赋值节点"""
    def __init__(self, var_name: str, value: ASTNode):
        self.var_name = var_name
        self.value = value

    def __repr__(self):
        return f"AssignmentNode({self.var_name} = {self.value})"


class ProgramNode(ASTNode):
    """程序节点（整个表达式）"""
    def __init__(self, statements: List[AssignmentNode], final_expr: ASTNode):
        self.statements = statements
        self.final_expr = final_expr

    def __repr__(self):
        stmts = '; '.join(repr(s) for s in self.statements)
        return f"ProgramNode({stmts}; {self.final_expr})"


# ============================================================================
# 第三部分：语法分析器（Parser）
# ============================================================================

class Parser:
    """语法分析器"""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = tokens[0] if tokens else Token(TokenType.EOF, None)

    def error(self, msg: str):
        """抛出语法错误"""
        raise SyntaxError(f"Syntax error at position {self.current_token.position}: {msg}")

    def advance(self):
        """前进到下一个token"""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = Token(TokenType.EOF, None)

    def peek(self, offset: int = 1) -> Token:
        """向前查看token"""
        peek_pos = self.pos + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return Token(TokenType.EOF, None)

    def expect(self, token_type: TokenType):
        """期望特定类型的token"""
        if self.current_token.type != token_type:
            self.error(f"Expected {token_type}, got {self.current_token.type}")
        self.advance()

    def is_assignment(self) -> bool:
        """检查当前是否是赋值语句"""
        if self.current_token.type == TokenType.IDENTIFIER:
            next_token = self.peek()
            return next_token.type == TokenType.ASSIGN
        return False

    def parse(self) -> ProgramNode:
        """解析整个程序"""
        statements = []

        # 解析赋值语句
        while self.current_token.type != TokenType.EOF and self.is_assignment():
            stmt = self.parse_assignment()
            statements.append(stmt)

            # 赋值语句后必须跟分号
            if self.current_token.type == TokenType.SEMICOLON:
                self.advance()
            else:
                self.error("Expected ';' after assignment")

        # 解析最终表达式
        if self.current_token.type == TokenType.EOF:
            self.error("Expected expression after assignments")

        final_expr = self.parse_expression()

        # 最终表达式后不能有分号
        if self.current_token.type == TokenType.SEMICOLON:
            self.error("Final expression cannot end with semicolon")

        # 检查是否到达结尾
        if self.current_token.type != TokenType.EOF:
            self.error(f"Unexpected token: {self.current_token.type}")

        return ProgramNode(statements, final_expr)

    def parse_assignment(self) -> AssignmentNode:
        """解析赋值语句"""
        var_name = self.current_token.value
        self.advance()
        self.expect(TokenType.ASSIGN)
        value = self.parse_expression()
        return AssignmentNode(var_name, value)

    def parse_expression(self) -> ASTNode:
        """解析表达式（处理比较运算符）"""
        return self.parse_comparison()

    def parse_comparison(self) -> ASTNode:
        """解析比较运算"""
        left = self.parse_additive()

        while self.current_token.type in [TokenType.LESS, TokenType.LESS_EQUAL,
                                          TokenType.GREATER, TokenType.GREATER_EQUAL,
                                          TokenType.EQUAL, TokenType.NOT_EQUAL]:
            op_token = self.current_token
            self.advance()
            right = self.parse_additive()
            left = BinaryOpNode(left, op_token.value, right)

        return left

    def parse_additive(self) -> ASTNode:
        """解析加减运算"""
        left = self.parse_multiplicative()

        while self.current_token.type in [TokenType.PLUS, TokenType.MINUS]:
            op_token = self.current_token
            self.advance()
            right = self.parse_multiplicative()
            left = BinaryOpNode(left, op_token.value, right)

        return left

    def parse_multiplicative(self) -> ASTNode:
        """解析乘除运算"""
        left = self.parse_unary()

        while self.current_token.type in [TokenType.MULTIPLY, TokenType.DIVIDE]:
            op_token = self.current_token
            self.advance()
            right = self.parse_unary()
            left = BinaryOpNode(left, op_token.value, right)

        return left

    def parse_unary(self) -> ASTNode:
        """解析一元运算（如负号）"""
        # 检查是否是一元负号
        if self.current_token.type == TokenType.MINUS:
            op_token = self.current_token
            self.advance()
            # 递归解析操作数
            operand = self.parse_unary()
            return UnaryOpNode(op_token.value, operand)

        # 不是一元运算符，继续解析主表达式
        return self.parse_primary()

    def parse_primary(self) -> ASTNode:
        """解析主表达式"""
        token = self.current_token

        # 数字
        if token.type == TokenType.NUMBER:
            self.advance()
            return NumberNode(token.value)

        # 字符串
        if token.type == TokenType.STRING:
            self.advance()
            return StringNode(token.value)

        # 布尔值
        if token.type == TokenType.BOOL:
            self.advance()
            return BoolNode(token.value)

        # NaN
        if token.type == TokenType.NAN:
            self.advance()
            return NanNode()

        # 标识符（可能是函数调用、数据字段或变量）
        if token.type == TokenType.IDENTIFIER:
            name = token.value
            self.advance()

            # 检查是否是函数调用
            if self.current_token.type == TokenType.LPAREN:
                return self.parse_function_call(name)
            else:
                # 否则是数据字段或变量引用
                return IdentifierNode(name)

        # 括号表达式
        if token.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr

        self.error(f"Unexpected token: {token.type}")

    def parse_function_call(self, func_name: str) -> FunctionCallNode:
        """解析函数调用"""
        self.expect(TokenType.LPAREN)

        args = []
        kwargs = {}

        # 解析参数
        while self.current_token.type != TokenType.RPAREN:
            # 检查是否是关键字参数
            if self.current_token.type == TokenType.IDENTIFIER and self.peek().type == TokenType.ASSIGN:
                key = self.current_token.value
                self.advance()
                self.expect(TokenType.ASSIGN)
                value = self.parse_expression()
                kwargs[key] = value
            else:
                # 位置参数
                args.append(self.parse_expression())

            # 检查是否有更多参数
            if self.current_token.type == TokenType.COMMA:
                self.advance()
            elif self.current_token.type != TokenType.RPAREN:
                self.error("Expected ',' or ')' in function call")

        self.expect(TokenType.RPAREN)
        return FunctionCallNode(func_name, args, kwargs)


# ============================================================================
# 第四部分：操作符规格系统
# ============================================================================

class ParamType(Enum):
    """参数类型枚举"""
    MATRIX = "MATRIX"
    VECTOR = "VECTOR"
    GROUP = "GROUP"
    INT = "INT"
    FLOAT = "FLOAT"
    BOOL = "BOOL"
    STRING = "STRING"
    ANY = "ANY"  # 任意类型


@dataclass
class ParamSpec:
    """参数规格"""
    name: str
    param_type: ParamType
    optional: bool = False
    default_value: Any = None
    value_constraint: Optional[Callable[[Any], bool]] = None

    def validate_value(self, value: Any) -> Tuple[bool, str]:
        """验证参数值"""
        if self.value_constraint and not self.value_constraint(value):
            return False, f"Value {value} does not meet constraint for parameter '{self.name}'"
        return True, ""


@dataclass
class OperatorSpec:
    """操作符规格"""
    name: str
    positional_params: List[ParamSpec]
    keyword_params: Dict[str, ParamSpec]
    variadic: bool = False
    return_type: ParamType = ParamType.MATRIX
    min_args: int = 0  # 可变参数的最小数量


class OperatorSpecBuilder:
    """操作符规格构建器 - 根据operators.json和特殊规则构建操作符规格"""

    @staticmethod
    def build_all_specs() -> Dict[str, OperatorSpec]:
        """构建所有操作符的规格"""
        specs = {}

        # Arithmetic operators
        specs.update(OperatorSpecBuilder._build_arithmetic_specs())
        # Logical operators
        specs.update(OperatorSpecBuilder._build_logical_specs())
        # Time Series operators
        specs.update(OperatorSpecBuilder._build_time_series_specs())
        # Cross Sectional operators
        specs.update(OperatorSpecBuilder._build_cross_sectional_specs())
        # Vector operators
        specs.update(OperatorSpecBuilder._build_vector_specs())
        # Transformational operators
        specs.update(OperatorSpecBuilder._build_transformational_specs())
        # Group operators
        specs.update(OperatorSpecBuilder._build_group_specs())

        return specs

    @staticmethod
    def _build_arithmetic_specs() -> Dict[str, OperatorSpec]:
        """构建算术运算符规格"""
        return {
            'add': OperatorSpec(
                name='add',
                positional_params=[
                    ParamSpec('x', ParamType.ANY),
                    ParamSpec('y', ParamType.ANY)
                ],
                keyword_params={'filter': ParamSpec('filter', ParamType.BOOL, optional=True, default_value=False)},
                variadic=True,
                min_args=2
            ),
            'multiply': OperatorSpec(
                name='multiply',
                positional_params=[
                    ParamSpec('x', ParamType.ANY),
                    ParamSpec('y', ParamType.ANY)
                ],
                keyword_params={'filter': ParamSpec('filter', ParamType.BOOL, optional=True, default_value=False)},
                variadic=True,
                min_args=2
            ),
            'sign': OperatorSpec(
                name='sign',
                positional_params=[ParamSpec('x', ParamType.ANY)],
                keyword_params={}
            ),
            'subtract': OperatorSpec(
                name='subtract',
                positional_params=[
                    ParamSpec('x', ParamType.ANY),
                    ParamSpec('y', ParamType.ANY)
                ],
                keyword_params={'filter': ParamSpec('filter', ParamType.BOOL, optional=True, default_value=False)}
            ),
            'log': OperatorSpec(
                name='log',
                positional_params=[ParamSpec('x', ParamType.ANY)],
                keyword_params={}
            ),
            'max': OperatorSpec(
                name='max',
                positional_params=[
                    ParamSpec('x', ParamType.ANY),
                    ParamSpec('y', ParamType.ANY)
                ],
                keyword_params={},
                variadic=True,
                min_args=2
            ),
            'to_nan': OperatorSpec(
                name='to_nan',
                positional_params=[ParamSpec('x', ParamType.ANY)],
                keyword_params={
                    'value': ParamSpec('value', ParamType.FLOAT, optional=True, default_value=0),
                    'reverse': ParamSpec('reverse', ParamType.BOOL, optional=True, default_value=False)
                }
            ),
            'abs': OperatorSpec(
                name='abs',
                positional_params=[ParamSpec('x', ParamType.ANY)],
                keyword_params={}
            ),
            'divide': OperatorSpec(
                name='divide',
                positional_params=[
                    ParamSpec('x', ParamType.ANY),
                    ParamSpec('y', ParamType.ANY)
                ],
                keyword_params={}
            ),
            'min': OperatorSpec(
                name='min',
                positional_params=[
                    ParamSpec('x', ParamType.ANY),
                    ParamSpec('y', ParamType.ANY)
                ],
                keyword_params={},
                variadic=True,
                min_args=2
            ),
            'signed_power': OperatorSpec(
                name='signed_power',
                positional_params=[
                    ParamSpec('x', ParamType.ANY),
                    ParamSpec('y', ParamType.ANY)
                ],
                keyword_params={}
            ),
            'inverse': OperatorSpec(
                name='inverse',
                positional_params=[ParamSpec('x', ParamType.ANY)],
                keyword_params={}
            ),
            'sqrt': OperatorSpec(
                name='sqrt',
                positional_params=[ParamSpec('x', ParamType.ANY)],
                keyword_params={}
            ),
            'reverse': OperatorSpec(
                name='reverse',
                positional_params=[ParamSpec('x', ParamType.ANY)],
                keyword_params={}
            ),
            'power': OperatorSpec(
                name='power',
                positional_params=[
                    ParamSpec('x', ParamType.ANY),
                    ParamSpec('y', ParamType.ANY)
                ],
                keyword_params={}
            ),
            'densify': OperatorSpec(
                name='densify',
                positional_params=[ParamSpec('x', ParamType.GROUP)],
                keyword_params={},
                return_type=ParamType.GROUP
            ),
        }

    @staticmethod
    def _build_logical_specs() -> Dict[str, OperatorSpec]:
        """构建逻辑运算符规格"""
        return {
            'or': OperatorSpec(
                name='or',
                positional_params=[
                    ParamSpec('input1', ParamType.ANY),
                    ParamSpec('input2', ParamType.ANY)
                ],
                keyword_params={},
                return_type=ParamType.MATRIX
            ),
            'and': OperatorSpec(
                name='and',
                positional_params=[
                    ParamSpec('input1', ParamType.ANY),
                    ParamSpec('input2', ParamType.ANY)
                ],
                keyword_params={},
                return_type=ParamType.MATRIX
            ),
            'not': OperatorSpec(
                name='not',
                positional_params=[ParamSpec('x', ParamType.ANY)],
                keyword_params={},
                return_type=ParamType.MATRIX
            ),
            'is_nan': OperatorSpec(
                name='is_nan',
                positional_params=[ParamSpec('input', ParamType.ANY)],
                keyword_params={},
                return_type=ParamType.MATRIX
            ),
            'if_else': OperatorSpec(
                name='if_else',
                positional_params=[
                    ParamSpec('input1', ParamType.ANY),
                    ParamSpec('input2', ParamType.ANY),
                    ParamSpec('input3', ParamType.ANY)
                ],
                keyword_params={},
                return_type=ParamType.MATRIX
            ),
            # 比较运算符的函数形式（也支持中缀形式 <, <=, >, >=, ==, !=）
            'less': OperatorSpec(
                name='less',
                positional_params=[
                    ParamSpec('input1', ParamType.ANY),
                    ParamSpec('input2', ParamType.ANY)
                ],
                keyword_params={},
                return_type=ParamType.MATRIX
            ),
            'less_equal': OperatorSpec(
                name='less_equal',
                positional_params=[
                    ParamSpec('input1', ParamType.ANY),
                    ParamSpec('input2', ParamType.ANY)
                ],
                keyword_params={},
                return_type=ParamType.MATRIX
            ),
            'greater': OperatorSpec(
                name='greater',
                positional_params=[
                    ParamSpec('input1', ParamType.ANY),
                    ParamSpec('input2', ParamType.ANY)
                ],
                keyword_params={},
                return_type=ParamType.MATRIX
            ),
            'greater_equal': OperatorSpec(
                name='greater_equal',
                positional_params=[
                    ParamSpec('input1', ParamType.ANY),
                    ParamSpec('input2', ParamType.ANY)
                ],
                keyword_params={},
                return_type=ParamType.MATRIX
            ),
            'equal': OperatorSpec(
                name='equal',
                positional_params=[
                    ParamSpec('input1', ParamType.ANY),
                    ParamSpec('input2', ParamType.ANY)
                ],
                keyword_params={},
                return_type=ParamType.MATRIX
            ),
            'not_equal': OperatorSpec(
                name='not_equal',
                positional_params=[
                    ParamSpec('input1', ParamType.ANY),
                    ParamSpec('input2', ParamType.ANY)
                ],
                keyword_params={},
                return_type=ParamType.MATRIX
            ),
        }

    @staticmethod
    def _build_time_series_specs() -> Dict[str, OperatorSpec]:
        """构建时间序列运算符规格"""
        return {
            'ts_corr': OperatorSpec(
                name='ts_corr',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('y', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={}
            ),
            'ts_zscore': OperatorSpec(
                name='ts_zscore',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={}
            ),
            'ts_product': OperatorSpec(
                name='ts_product',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={}
            ),
            'ts_std_dev': OperatorSpec(
                name='ts_std_dev',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={}
            ),
            'ts_backfill': OperatorSpec(
                name='ts_backfill',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, optional=True, value_constraint=lambda v: v > 0)
                ],
                keyword_params={
                    'lookback': ParamSpec('lookback', ParamType.INT, optional=True, value_constraint=lambda v: v > 0),
                    'k': ParamSpec('k', ParamType.INT, optional=True, default_value=1, value_constraint=lambda v: v > 0)
                }
            ),
            'days_from_last_change': OperatorSpec(
                name='days_from_last_change',
                positional_params=[ParamSpec('x', ParamType.MATRIX)],
                keyword_params={}
            ),
            'last_diff_value': OperatorSpec(
                name='last_diff_value',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={}
            ),
            'ts_scale': OperatorSpec(
                name='ts_scale',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={
                    'constant': ParamSpec('constant', ParamType.FLOAT, optional=True, default_value=0)
                }
            ),
            'ts_step': OperatorSpec(
                name='ts_step',
                positional_params=[ParamSpec('n', ParamType.INT, value_constraint=lambda v: v > 0)],
                keyword_params={}
            ),
            'ts_sum': OperatorSpec(
                name='ts_sum',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={}
            ),
            'ts_av_diff': OperatorSpec(
                name='ts_av_diff',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={}
            ),
            'ts_mean': OperatorSpec(
                name='ts_mean',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={}
            ),
            'ts_arg_max': OperatorSpec(
                name='ts_arg_max',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={}
            ),
            'ts_max': OperatorSpec(
                name='ts_max',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={}
            ),
            'ts_rank': OperatorSpec(
                name='ts_rank',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={
                    'constant': ParamSpec('constant', ParamType.FLOAT, optional=True, default_value=0)
                }
            ),
            'ts_delay': OperatorSpec(
                name='ts_delay',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={}
            ),
            'ts_quantile': OperatorSpec(
                name='ts_quantile',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={
                    'driver': ParamSpec('driver', ParamType.STRING, optional=True, default_value="gaussian",
                                       value_constraint=lambda v: v in ['gaussian', 'cauchy', 'uniform'])
                }
            ),
            'ts_min': OperatorSpec(
                name='ts_min',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={}
            ),
            'ts_count_nans': OperatorSpec(
                name='ts_count_nans',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={}
            ),
            'ts_covariance': OperatorSpec(
                name='ts_covariance',
                positional_params=[
                    ParamSpec('y', ParamType.MATRIX),
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={}
            ),
            'ts_decay_linear': OperatorSpec(
                name='ts_decay_linear',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={
                    'dense': ParamSpec('dense', ParamType.BOOL, optional=True, default_value=False)
                }
            ),
            'jump_decay': OperatorSpec(
                name='jump_decay',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={
                    'stddev': ParamSpec('stddev', ParamType.BOOL, optional=True, default_value=True),
                    'sensitivity': ParamSpec('sensitivity', ParamType.FLOAT, optional=False,
                                            value_constraint=lambda v: 0 < v < 1),
                    'force': ParamSpec('force', ParamType.FLOAT, optional=False,
                                      value_constraint=lambda v: 0 < v < 1)
                }
            ),
            'ts_arg_min': OperatorSpec(
                name='ts_arg_min',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={}
            ),
            'ts_regression': OperatorSpec(
                name='ts_regression',
                positional_params=[
                    ParamSpec('y', ParamType.MATRIX),
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={
                    'lag': ParamSpec('lag', ParamType.INT, optional=True, default_value=0,
                                    value_constraint=lambda v: v >= 0),
                    'rettype': ParamSpec('rettype', ParamType.INT, optional=True, default_value=0,
                                        value_constraint=lambda v: 0 <= v <= 9)
                }
            ),
            'kth_element': OperatorSpec(
                name='kth_element',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={
                    'k': ParamSpec('k', ParamType.INT, optional=False, value_constraint=lambda v: v > 0)
                }
            ),
            'hump': OperatorSpec(
                name='hump',
                positional_params=[ParamSpec('x', ParamType.MATRIX)],
                keyword_params={
                    'hump': ParamSpec('hump', ParamType.FLOAT, optional=False,
                                     value_constraint=lambda v: 0 < v < 1)
                }
            ),
            'ts_delta': OperatorSpec(
                name='ts_delta',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={}
            ),
            'ts_target_tvr_decay': OperatorSpec(
                name='ts_target_tvr_decay',
                positional_params=[ParamSpec('x', ParamType.MATRIX)],
                keyword_params={
                    'lambda_min': ParamSpec('lambda_min', ParamType.FLOAT, optional=False, default_value=0),
                    'lambda_max': ParamSpec('lambda_max', ParamType.FLOAT, optional=False, default_value=1),
                    'target_tvr': ParamSpec('target_tvr', ParamType.FLOAT, optional=False, default_value=0.1)
                }
            ),
            'ts_target_tvr_delta_limit': OperatorSpec(
                name='ts_target_tvr_delta_limit',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('y', ParamType.MATRIX)
                ],
                keyword_params={
                    'lambda_min': ParamSpec('lambda_min', ParamType.FLOAT, optional=False, default_value=0),
                    'lambda_max': ParamSpec('lambda_max', ParamType.FLOAT, optional=False, default_value=1),
                    'target_tvr': ParamSpec('target_tvr', ParamType.FLOAT, optional=False, default_value=0.1)
                }
            ),
        }

    @staticmethod
    def _build_cross_sectional_specs() -> Dict[str, OperatorSpec]:
        """构建截面运算符规格"""
        return {
            'winsorize': OperatorSpec(
                name='winsorize',
                positional_params=[ParamSpec('x', ParamType.MATRIX)],
                keyword_params={
                    'std': ParamSpec('std', ParamType.FLOAT, optional=False,
                                    value_constraint=lambda v: v > 0)
                }
            ),
            'rank': OperatorSpec(
                name='rank',
                positional_params=[ParamSpec('x', ParamType.MATRIX)],
                keyword_params={
                    'rate': ParamSpec('rate', ParamType.INT, optional=True, default_value=2,
                                     value_constraint=lambda v: v in [0, 2])
                }
            ),
            'vector_neut': OperatorSpec(
                name='vector_neut',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('y', ParamType.MATRIX)
                ],
                keyword_params={}
            ),
            'zscore': OperatorSpec(
                name='zscore',
                positional_params=[ParamSpec('x', ParamType.MATRIX)],
                keyword_params={}
            ),
            'scale_down': OperatorSpec(
                name='scale_down',
                positional_params=[ParamSpec('x', ParamType.MATRIX)],
                keyword_params={
                    'constant': ParamSpec('constant', ParamType.FLOAT, optional=True, default_value=0)
                }
            ),
            'scale': OperatorSpec(
                name='scale',
                positional_params=[ParamSpec('x', ParamType.MATRIX)],
                keyword_params={
                    'scale': ParamSpec('scale', ParamType.INT, optional=True, default_value=1,
                                      value_constraint=lambda v: v > 0),
                    'longscale': ParamSpec('longscale', ParamType.INT, optional=True, default_value=1,
                                          value_constraint=lambda v: v > 0),
                    'shortscale': ParamSpec('shortscale', ParamType.INT, optional=True, default_value=1,
                                           value_constraint=lambda v: v > 0)
                }
            ),
            'normalize': OperatorSpec(
                name='normalize',
                positional_params=[ParamSpec('x', ParamType.MATRIX)],
                keyword_params={
                    'useStd': ParamSpec('useStd', ParamType.BOOL, optional=True, default_value=False),
                    'limit': ParamSpec('limit', ParamType.FLOAT, optional=True, default_value=0.0)
                }
            ),
            'quantile': OperatorSpec(
                name='quantile',
                positional_params=[ParamSpec('x', ParamType.MATRIX)],
                keyword_params={
                    'driver': ParamSpec('driver', ParamType.STRING, optional=True, default_value="gaussian",
                                       value_constraint=lambda v: v in ['gaussian', 'cauchy', 'uniform']),
                    'sigma': ParamSpec('sigma', ParamType.FLOAT, optional=True, default_value=1.0)
                }
            ),
        }

    @staticmethod
    def _build_vector_specs() -> Dict[str, OperatorSpec]:
        """构建向量运算符规格"""
        return {
            'vec_min': OperatorSpec(
                name='vec_min',
                positional_params=[ParamSpec('x', ParamType.VECTOR)],
                keyword_params={},
                return_type=ParamType.MATRIX
            ),
            'vec_sum': OperatorSpec(
                name='vec_sum',
                positional_params=[ParamSpec('x', ParamType.VECTOR)],
                keyword_params={},
                return_type=ParamType.MATRIX
            ),
            'vec_max': OperatorSpec(
                name='vec_max',
                positional_params=[ParamSpec('x', ParamType.VECTOR)],
                keyword_params={},
                return_type=ParamType.MATRIX
            ),
            'vec_avg': OperatorSpec(
                name='vec_avg',
                positional_params=[ParamSpec('x', ParamType.VECTOR)],
                keyword_params={},
                return_type=ParamType.MATRIX
            ),
        }

    @staticmethod
    def _build_transformational_specs() -> Dict[str, OperatorSpec]:
        """构建变换运算符规格"""
        return {
            'bucket': OperatorSpec(
                name='bucket',
                positional_params=[ParamSpec('x', ParamType.MATRIX)],
                keyword_params={
                    'range': ParamSpec('range', ParamType.STRING, optional=False)
                },
                return_type=ParamType.GROUP
            ),
            'trade_when': OperatorSpec(
                name='trade_when',
                positional_params=[
                    ParamSpec('x', ParamType.ANY),
                    ParamSpec('y', ParamType.ANY),
                    ParamSpec('z', ParamType.ANY)
                ],
                keyword_params={}
            ),
        }

    @staticmethod
    def _build_group_specs() -> Dict[str, OperatorSpec]:
        """构建分组运算符规格"""
        return {
            'group_min': OperatorSpec(
                name='group_min',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('group', ParamType.GROUP)
                ],
                keyword_params={}
            ),
            'group_mean': OperatorSpec(
                name='group_mean',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('weight', ParamType.ANY),  # 可以是常数或matrix
                    ParamSpec('group', ParamType.GROUP)
                ],
                keyword_params={}
            ),
            'group_max': OperatorSpec(
                name='group_max',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('group', ParamType.GROUP)
                ],
                keyword_params={}
            ),
            'group_rank': OperatorSpec(
                name='group_rank',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('group', ParamType.GROUP)
                ],
                keyword_params={}
            ),
            'group_backfill': OperatorSpec(
                name='group_backfill',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('group', ParamType.GROUP),
                    ParamSpec('d', ParamType.INT, value_constraint=lambda v: v > 0)
                ],
                keyword_params={
                    'std': ParamSpec('std', ParamType.FLOAT, optional=True, default_value=4.0)
                }
            ),
            'group_scale': OperatorSpec(
                name='group_scale',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('group', ParamType.GROUP)
                ],
                keyword_params={}
            ),
            'group_zscore': OperatorSpec(
                name='group_zscore',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('group', ParamType.GROUP)
                ],
                keyword_params={}
            ),
            'group_neutralize': OperatorSpec(
                name='group_neutralize',
                positional_params=[
                    ParamSpec('x', ParamType.MATRIX),
                    ParamSpec('group', ParamType.GROUP)
                ],
                keyword_params={}
            ),
            'group_cartesian_product': OperatorSpec(
                name='group_cartesian_product',
                positional_params=[
                    ParamSpec('g1', ParamType.GROUP),
                    ParamSpec('g2', ParamType.GROUP)
                ],
                keyword_params={},
                return_type=ParamType.GROUP
            ),
        }


# ============================================================================
# 第五部分：数据上下文（Data Context）
# ============================================================================

class DataContext:
    """数据上下文 - 加载和管理数据字段和操作符信息"""

    def __init__(self, csv_path: str = 'EUR_TOP2500_1.csv'):
        """初始化数据上下文"""
        self.datafields = self._load_datafields(csv_path)
        self.operators = OperatorSpecBuilder.build_all_specs()

    def _load_datafields(self, csv_path: str) -> Dict[str, Dict]:
        """从CSV加载数据字段信息"""
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            datafields = {}

            for _, row in df.iterrows():
                field_id = row['id']
                datafields[field_id] = {
                    'type': row['type'],  # MATRIX, VECTOR, GROUP
                    'description': row.get('description', '')
                }

            return datafields
        except Exception as e:
            raise RuntimeError(f"Failed to load datafields from {csv_path}: {e}")

    def is_datafield(self, name: str) -> bool:
        """检查是否是数据字段（不区分大小写）"""
        name_lower = name.lower()
        return any(field.lower() == name_lower for field in self.datafields)

    def get_datafield_type(self, name: str) -> Optional[ParamType]:
        """获取数据字段类型（不区分大小写）"""
        name_lower = name.lower()
        for field_name, field_info in self.datafields.items():
            if field_name.lower() == name_lower:
                type_str = field_info['type']
                return ParamType[type_str]
        return None

    def is_operator(self, name: str) -> bool:
        """检查是否是操作符（不区分大小写）"""
        name_lower = name.lower()
        return any(op.lower() == name_lower for op in self.operators)

    def get_operator_spec(self, name: str) -> Optional[OperatorSpec]:
        """获取操作符规格（不区分大小写）"""
        name_lower = name.lower()
        for op_name, op_spec in self.operators.items():
            if op_name.lower() == name_lower:
                return op_spec
        return None




# ============================================================================
# 第六部分：语义分析器（Semantic Analyzer）
# ============================================================================

class SemanticAnalyzer:
    """语义分析器 - 执行类型检查、参数验证等"""

    def __init__(self, context: DataContext):
        self.context = context
        self.defined_vars = {}  # {var_name: ParamType}
        self.used_vars = set()
        self.errors = []

    def analyze(self, ast: ProgramNode) -> Tuple[bool, List[str]]:
        """分析AST，返回(是否通过, 错误列表)"""
        self.errors = []
        self.defined_vars = {}
        self.used_vars = set()

        # 分析赋值语句
        for stmt in ast.statements:
            self._analyze_assignment(stmt)

        # 分析最终表达式
        final_expr_type = self._analyze_expression(ast.final_expr)

        # 检查未使用的变量
        self._check_unused_variables()

        # 检查caution3规则：最终表达式不能是赋值
        if isinstance(ast.final_expr, AssignmentNode):
            self.errors.append("Final expression cannot be an assignment (caution3)")

        # 检查最终表达式不能是GROUP类型
        if final_expr_type == ParamType.GROUP:
            self.errors.append("Final expression cannot return GROUP type. GROUP values must be used with group_* operators only")

        return (len(self.errors) == 0, self.errors)

    def _analyze_assignment(self, node: AssignmentNode):
        """分析赋值语句"""
        var_name = node.var_name

        # 检查变量名冲突
        if self.context.is_operator(var_name):
            self.errors.append(f"Variable name '{var_name}' conflicts with operator")
            return

        if self.context.is_datafield(var_name):
            self.errors.append(f"Variable name '{var_name}' conflicts with datafield")
            return

        # 检查禁用的变量名（不区分大小写）
        forbidden_names = {'delta', 'sum', 'covariance', 'delay'}
        if var_name.lower() in forbidden_names:
            self.errors.append(f"Variable name '{var_name}' is reserved and cannot be used")
            return

        # 分析右侧表达式
        expr_type = self._analyze_expression(node.value)

        # 记录变量类型
        self.defined_vars[var_name] = expr_type

    def _analyze_expression(self, node: ASTNode) -> ParamType:
        """分析表达式，返回表达式的类型"""
        if isinstance(node, NumberNode):
            if isinstance(node.value, int):
                return ParamType.INT
            return ParamType.FLOAT

        elif isinstance(node, StringNode):
            return ParamType.STRING

        elif isinstance(node, BoolNode):
            return ParamType.BOOL

        elif isinstance(node, NanNode):
            return ParamType.FLOAT

        elif isinstance(node, IdentifierNode):
            return self._analyze_identifier(node)

        elif isinstance(node, FunctionCallNode):
            return self._analyze_function_call(node)

        elif isinstance(node, BinaryOpNode):
            return self._analyze_binary_op(node)

        elif isinstance(node, UnaryOpNode):
            return self._analyze_unary_op(node)

        else:
            self.errors.append(f"Unknown node type: {type(node).__name__}")
            return ParamType.ANY

    def _analyze_identifier(self, node: IdentifierNode) -> ParamType:
        """分析标识符（数据字段或变量）"""
        name = node.name

        # 检查是否是变量
        if name in self.defined_vars:
            self.used_vars.add(name)
            return self.defined_vars[name]

        # 检查是否是数据字段
        if self.context.is_datafield(name):
            return self.context.get_datafield_type(name)

        # 未定义的标识符
        self.errors.append(f"Undefined identifier '{name}'")
        return ParamType.ANY

    def _analyze_binary_op(self, node: BinaryOpNode) -> ParamType:
        """分析二元运算"""
        left_type = self._analyze_expression(node.left)
        right_type = self._analyze_expression(node.right)

        # 算术运算和比较运算通常返回MATRIX类型
        return ParamType.MATRIX

    def _analyze_unary_op(self, node: UnaryOpNode) -> ParamType:
        """分析一元运算"""
        operand_type = self._analyze_expression(node.operand)

        # 一元负号运算返回与操作数相同的类型（或MATRIX类型）
        # 如果操作数是数值类型，保持其类型；否则返回MATRIX
        if operand_type in [ParamType.INT, ParamType.FLOAT]:
            return operand_type
        return ParamType.MATRIX

    def _analyze_function_call(self, node: FunctionCallNode) -> ParamType:
        """分析函数调用"""
        func_name = node.name

        # 检查操作符是否存在
        if not self.context.is_operator(func_name):
            self.errors.append(f"Unknown operator: '{func_name}'")
            return ParamType.ANY

        spec = self.context.get_operator_spec(func_name)

        # 检查参数数量
        self._check_param_count(node, spec)

        # 分析并检查位置参数类型
        arg_types = []
        for i, arg in enumerate(node.args):
            arg_type = self._analyze_expression(arg)
            arg_types.append(arg_type)

            # 检查参数类型
            if i < len(spec.positional_params):
                param_spec = spec.positional_params[i]
                self._check_param_type(arg, arg_type, param_spec, func_name, i)

        # 分析并检查关键字参数
        for key, value in node.kwargs.items():
            if key not in spec.keyword_params:
                self.errors.append(f"Unknown keyword argument '{key}' for operator '{func_name}'")
                continue

            param_spec = spec.keyword_params[key]
            value_type = self._analyze_expression(value)
            self._check_param_type(value, value_type, param_spec, func_name, key)

            # 检查参数值约束
            if isinstance(value, (NumberNode, StringNode, BoolNode)):
                is_valid, msg = param_spec.validate_value(value.value)
                if not is_valid:
                    self.errors.append(f"{func_name}: {msg}")

        # 检查必需的关键字参数是否都被提供
        for key, param_spec in spec.keyword_params.items():
            # 如果参数不是可选的，且没有默认值，则必须提供
            if not param_spec.optional and param_spec.default_value is None:
                if key not in node.kwargs:
                    self.errors.append(
                        f"{func_name}: Missing required keyword argument '{key}'"
                    )

        # 检查关键字参数是否使用了位置传递
        self._check_keyword_params_format(node, spec)

        # 特殊规则检查
        self._check_special_rules(node, spec, arg_types)

        # 返回函数返回类型
        return spec.return_type

    def _check_param_count(self, node: FunctionCallNode, spec: OperatorSpec):
        """检查参数数量"""
        func_name = node.name
        arg_count = len(node.args)

        if spec.variadic:
            # 可变参数，检查最小数量
            if arg_count < spec.min_args:
                self.errors.append(
                    f"{func_name}: Expected at least {spec.min_args} arguments, got {arg_count}"
                )
        else:
            # 固定参数，考虑可选参数
            # 计算必需参数数量（optional=False）
            required_count = sum(1 for p in spec.positional_params if not p.optional)
            total_count = len(spec.positional_params)

            if arg_count < required_count:
                self.errors.append(
                    f"{func_name}: Expected at least {required_count} positional arguments, got {arg_count}"
                )
            elif arg_count > total_count:
                self.errors.append(
                    f"{func_name}: Expected at most {total_count} positional arguments, got {arg_count}"
                )

    def _check_param_type(self, arg: ASTNode, arg_type: ParamType,
                         param_spec: ParamSpec, func_name: str, param_index):
        """检查参数类型是否匹配"""
        expected_type = param_spec.param_type

        # ANY类型接受任何类型，但不包括GROUP、STRING和VECTOR
        if expected_type == ParamType.ANY:
            if arg_type == ParamType.GROUP:
                self.errors.append(
                    f"{func_name}: Parameter {param_index} cannot accept GROUP type. "
                    f"GROUP fields can only be used with operators that explicitly accept GROUP parameters"
                )
            elif arg_type == ParamType.STRING:
                self.errors.append(
                    f"{func_name}: Parameter {param_index} cannot accept STRING type. "
                    f"STRING values can only be used with operators that explicitly accept STRING parameters"
                )
            elif arg_type == ParamType.VECTOR:
                self.errors.append(
                    f"{func_name}: Parameter {param_index} cannot accept VECTOR type. "
                    f"VECTOR fields must be converted to MATRIX using vec_* operators first"
                )
            return

        # VECTOR类型必须先通过vec_*操作符转换为MATRIX
        if expected_type == ParamType.MATRIX and arg_type == ParamType.VECTOR:
            # 检查是否是vec_*操作符的调用
            if not (isinstance(arg, FunctionCallNode) and arg.name.startswith('vec_')):
                self.errors.append(
                    f"{func_name}: Parameter {param_index} requires MATRIX type, "
                    f"but got VECTOR. VECTOR fields must be converted using vec_* operators"
                )
            return

        # GROUP类型只能用于group参数
        if expected_type == ParamType.GROUP and arg_type != ParamType.GROUP:
            self.errors.append(
                f"{func_name}: Parameter '{param_spec.name}' requires GROUP type, got {arg_type.value}"
            )
            return

        # 类型匹配检查（允许INT作为FLOAT使用）
        if expected_type != arg_type:
            if not (expected_type == ParamType.FLOAT and arg_type == ParamType.INT):
                if arg_type != ParamType.ANY:  # ANY类型跳过检查
                    self.errors.append(
                        f"{func_name}: Parameter {param_index} type mismatch - "
                        f"expected {expected_type.value}, got {arg_type.value}"
                    )

    def _check_keyword_params_format(self, node: FunctionCallNode, spec: OperatorSpec):
        """检查关键字参数是否正确使用name=value格式"""
        func_name = node.name

        # 检查位置参数中是否误用了关键字参数的名称
        for i, arg in enumerate(node.args):
            # 如果位置参数超过了定义的位置参数数量，且有对应的关键字参数
            if i >= len(spec.positional_params):
                # 检查是否应该使用关键字参数
                if isinstance(arg, IdentifierNode) and arg.name in spec.keyword_params:
                    self.errors.append(
                        f"{func_name}: Parameter '{arg.name}' should be passed as keyword argument"
                    )

    def _check_special_rules(self, node: FunctionCallNode, spec: OperatorSpec, arg_types: List[ParamType]):
        """检查特殊规则"""
        func_name = node.name

        # 特殊规则1: ts_backfill参数使用检查
        if func_name == 'ts_backfill':
            # 检查1a: 不允许使用ignore参数
            if 'ignore' in node.kwargs:
                self.errors.append(f"ts_backfill: 'ignore' parameter is not allowed")

            # 检查1b: 参数使用方式 - 要么用 ts_backfill(x, d)，要么用 ts_backfill(x, lookback=d)
            has_second_positional = len(node.args) >= 2
            has_lookback_keyword = 'lookback' in node.kwargs

            if has_second_positional and has_lookback_keyword:
                self.errors.append(
                    "ts_backfill: Cannot use both positional parameter 'd' and keyword parameter 'lookback'. "
                    "Use either ts_backfill(x, d) or ts_backfill(x, lookback=d)"
                )
            elif not has_second_positional and not has_lookback_keyword:
                self.errors.append(
                    "ts_backfill: Must provide either second positional parameter or 'lookback' keyword parameter. "
                    "Use either ts_backfill(x, d) or ts_backfill(x, lookback=d)"
                )

        # 特殊规则2: bucket的range参数格式检查
        if func_name == 'bucket' and 'range' in node.kwargs:
            range_node = node.kwargs['range']
            if isinstance(range_node, StringNode):
                range_val = range_node.value
                # 检查格式：应该是 "a, b, c" 形式
                parts = range_val.split(',')
                if len(parts) != 3:
                    self.errors.append(f"bucket: range parameter must be in format 'a,b,c'")
                else:
                    try:
                        a, b, c = [float(p.strip()) for p in parts]
                        # 检查c能否被1整除
                        if c <= 0 or 1 % c != 0:
                            # 应该是1可以被c整除
                            if abs(round(1 / c) * c - 1) > 0.001:
                                self.errors.append(f"bucket: range parameter c={c} should divide 1 evenly")
                    except ValueError:
                        self.errors.append(f"bucket: range parameter must contain numeric values")

        # 特殊规则3: rank的rate参数检查
        if func_name == 'rank' and 'rate' in node.kwargs:
            rate_node = node.kwargs['rate']
            if isinstance(rate_node, NumberNode):
                if rate_node.value not in [0, 2]:
                    self.errors.append(f"rank: rate parameter must be 0 or 2 (or omitted)")

        # 特殊规则4: lambda_min < lambda_max 检查
        if func_name in ['ts_target_tvr_decay', 'ts_target_tvr_delta_limit']:
            if 'lambda_min' in node.kwargs and 'lambda_max' in node.kwargs:
                min_node = node.kwargs['lambda_min']
                max_node = node.kwargs['lambda_max']
                if isinstance(min_node, NumberNode) and isinstance(max_node, NumberNode):
                    if min_node.value >= max_node.value:
                        self.errors.append(
                            f"{func_name}: lambda_min must be less than lambda_max"
                        )

    def _check_unused_variables(self):
        """检查未使用的变量"""
        unused = set(self.defined_vars.keys()) - self.used_vars
        for var in unused:
            self.errors.append(f"Variable '{var}' is defined but never used")


# ============================================================================
# 第七部分：主验证函数
# ============================================================================

def validate_expression(expression: str, 
                       csv_path: str = 'EUR_TOP2500_1.csv') -> Tuple[bool, List[str]]:
    """
    验证Alpha表达式
    
    Args:
        expression: Alpha表达式字符串
        csv_path: 数据字段CSV文件路径
    
    Returns:
        (是否通过验证, 错误列表)
    """
    try:
        # 1. 词法分析
        tokenizer = Tokenizer(expression)
        tokens = tokenizer.tokenize()
        
        # 2. 语法分析
        parser = Parser(tokens)
        ast = parser.parse()
        
        # 3. 加载数据上下文
        context = DataContext(csv_path)
        
        # 4. 语义分析
        analyzer = SemanticAnalyzer(context)
        is_valid, errors = analyzer.analyze(ast)
        
        return is_valid, errors
    
    except SyntaxError as e:
        return False, [f"Syntax error: {str(e)}"]
    except Exception as e:
        return False, [f"Validation error: {str(e)}"]


def validate_expression_batch(expressions: List[str],
                             csv_path: str = 'EUR_TOP2500_1.csv') -> List[Tuple[bool, List[str]]]:
    """
    批量验证表达式
    
    Args:
        expressions: 表达式列表
        csv_path: 数据字段CSV文件路径
    
    Returns:
        [(是否通过, 错误列表), ...]
    """
    # 共享数据上下文以提高性能
    context = DataContext(csv_path)
    
    results = []
    for expr in expressions:
        try:
            tokenizer = Tokenizer(expr)
            tokens = tokenizer.tokenize()
            
            parser = Parser(tokens)
            ast = parser.parse()
            
            analyzer = SemanticAnalyzer(context)
            is_valid, errors = analyzer.analyze(ast)
            
            results.append((is_valid, errors))
        except SyntaxError as e:
            results.append((False, [f"Syntax error: {str(e)}"]))
        except Exception as e:
            results.append((False, [f"Validation error: {str(e)}"]))
    
    return results


# ============================================================================
# 测试和调试工具
# ============================================================================

if __name__ == "__main__":
    # 测试用例
    test_cases = [
        # 正确的表达式
        "rank(close)",
        "ts_rank(returns, 10)",
        "add(close, open)",
        "a = rank(close); quantile(a)",
        "vec_avg(anl10_cpxff)",
        
        # 错误的表达式
        "rank()",  # 参数数量错误
        "rank(anl10_cpxff)",  # VECTOR未转换
        "a = rank(close); b = rank(open); quantile(a)",  # 未使用的变量b
        "rank(close);",  # 最终表达式以分号结尾
        "undefined_field",  # 未定义的字段
    ]
    
    print("Expression Validator Test Cases")
    print("=" * 80)
    
    for expr in test_cases:
        print(f"\nExpression: {expr}")
        is_valid, errors = validate_expression(expr)
        if is_valid:
            print("✓ VALID")
        else:
            print("✗ INVALID")
            for error in errors:
                print(f"  - {error}")
    
    print("\n" + "=" * 80)