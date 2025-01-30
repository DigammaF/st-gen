
from __future__ import annotations
from enum import Enum
import string
from sys import argv
from dataclasses import dataclass
from typing import Callable
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from argparse import ArgumentParser

VAR_NAME_CHARS: str = string.ascii_letters + string.digits + ".%"
OPERATORS: tuple[str, ...] = (
	"not", "and", "or", "<=", ">=", "<>", "=", ">", "<",
	"+", "-"
)
OPEN_P: tuple[str] = ("(",)
CLOSE_P: tuple[str] = (")",)
END_INSTR: tuple[str] = (";",)
IF: tuple[str] = ("if",)
END_IF: tuple[str] = ("end_if",)
THEN: tuple[str] = ("then",)
CONTROL_FLOW: tuple[str, ...] = THEN + END_INSTR + IF + END_IF
KEYWORDS: tuple[str, ...] = CONTROL_FLOW + OPERATORS + OPEN_P + CLOSE_P

BINARY_NUMERICAL_FUNCTIONS: dict[str, Callable[[int, int], int]] = {
	"<=": lambda a, b: a <= b,
	">=": lambda a, b: a >= b,
	"<>": lambda a, b: a != b,
	"=": lambda a, b: a == b,
	"<": lambda a, b: a < b,
	">": lambda a, b: a > b,
	"+": lambda a, b: a + b,
	"-": lambda a, b: a - b,
	"*": lambda a, b: a*b
}

class STSyntaxError(Exception): ...

def get_arg_count(operator: str) -> int:
	if operator in ("and", "or", "<=", ">=", "<>", "=", ">", "<", "+", "-"):
		return 2

	if operator in ("not", ):
		return 1

	raise Exception(f"unknown operator {operator}")

@dataclass
class Token:
	type: TokenType
	value: str

class PreprocessedToken(Token):
	""" a token that has already been assigned an AST """
	ast: ASTValue

	def __init__(self, ast: ASTValue):
		self.type = TokenType.PREPROCESSED
		self.value = "/PREPROCESSED/"
		self.ast = ast

class TokenType(Enum):
	OPERATOR = "OPERATOR"
	OPEN_P = "OPEN_P"
	CLOSE_P = "CLOSE_P"
	END_IF = "END_IF"
	IF = "IF"
	END_INSTR = "END_INSTR"
	THEN = "THEN"
	EOF = "EOF"
	VARIABLE = "VARIABLE"
	COMMENT = "COMMENT"
	ASSIGNMENT = "ASSIGNMENT"
	LITERAL = "LITERAL"
	PREPROCESSED = "PREPROCESSED"

def getType(value: str) -> TokenType:
	if value in OPERATORS: return TokenType.OPERATOR
	if value in OPEN_P: return TokenType.OPEN_P
	if value in CLOSE_P: return TokenType.CLOSE_P
	if value in END_IF: return TokenType.END_IF
	if value in END_INSTR: return TokenType.END_INSTR
	if value in IF: return TokenType.IF
	if value in THEN: return TokenType.THEN
	
	raise Exception(f"cannot find type of '{value}'")

@dataclass
class Function:
	arg_count: int

class Args:
	display_all_reads: bool = False

@dataclass
class STInterpreter:
	tokens: list[Token]
	variables: dict[str, int|str]
	console: Console
	args: Args

	@property
	def token(self) -> Token:
		return self.tokens[0]
	
	def advance(self):
		self.tokens.pop(0)

	def expect(self, types: tuple[TokenType, ...]) -> Token:
		token = self.token

		if token.type in types:
			self.advance()
			return token
		
		else:
			raise STSyntaxError(f"expected any of {types}, got {token}")

	def fetch(self, name: str) -> int|str:
		return self.variables.get(name, 0)

	def set(self, name: str, value: int|str):
		self.variables[name] = value

	def starts_with(self, *token_types: TokenType) -> bool:
		if len(self.tokens) < len(token_types):
			return False

		for actual, expected in zip(self.tokens, token_types):
			if actual.type != expected:
				return False
			
		return True

class AST:
	def run(self, interpreter: STInterpreter): ...

@dataclass
class ASTUnknownInstruction(AST):
	line: str

	def run(self, interpreter: STInterpreter):
		interpreter.console.print(f"(?)\t[red]{self.line}[/red]")

@dataclass
class ASTConditionalBody(AST):
	condition: ASTValue
	body: ASTBody

	def run(self, interpreter: STInterpreter):
		self.condition.run(interpreter)

		if self.condition.value:
			self.body.run(interpreter)

@dataclass
class ASTBody(AST):
	children: list[AST]

	def run(self, interpreter: STInterpreter):
		for child in self.children:
			child.run(interpreter)

class ASTValue(AST):
	value: int|str

	def get_write_handle(self) -> str|None:
		return None

@dataclass
class ASTOperator(ASTValue):
	operator: Token
	operands: list[ASTValue]

	def __str__(self) -> str:
		return f" {self.operator.value} ".join(str(op) for op in self.operands)

	def run(self, interpreter: STInterpreter):
		assert self.operator.type == TokenType.OPERATOR

		if self.operator.value == "not":
			[operand,] = self.operands
			operand.run(interpreter)
			self.value = not bool(operand.value)
			return

		if self.operator.value == "and":
			self.value = True

			for operand in self.operands:
				operand.run(interpreter)
				self.value = bool(operand.value)
				if not self.value: break

			return

		if self.operator.value == "or":
			self.value = False

			for operand in self.operands:
				operand.run(interpreter)
				self.value = bool(operand.value)
				if self.value: break

			return

		if self.operator.value in BINARY_NUMERICAL_FUNCTIONS:
			fn = BINARY_NUMERICAL_FUNCTIONS[self.operator.value]
			[a, b] = self.operands
			a.run(interpreter)
			b.run(interpreter)
			if not isinstance(a.value, int): raise STSyntaxError(f"value '{a}' is not compatible with numerical operation {self.operator.value}")
			if not isinstance(b.value, int): raise STSyntaxError(f"value '{b}' is not compatible with numerical operation {self.operator.value}")
			self.value = fn(a.value, b.value)
			return

		raise Exception(f"{self.operator} is not a supported operator")

@dataclass
class ASTSubValue(ASTValue):
	inner: ASTValue

	def __str__(self) -> str:
		return f"({str(self.inner)})"

	def run(self, interpreter: STInterpreter):
		self.inner.run(interpreter)
		self.value = self.inner.value

	def get_write_handle(self) -> str | None:
		return self.inner.get_write_handle()

@dataclass
class ASTVariable(ASTValue):
	token: Token

	def __str__(self) -> str:
		return f"{self.token.value}"

	def run(self, interpreter: STInterpreter):
		assert self.token.type == TokenType.VARIABLE
		important = "t#" in self.token.value
		self.value = interpreter.fetch(self.token.value)

		if important or interpreter.args.display_all_reads:
			color = "yellow" if important else "white"
			interpreter.console.print(f"(R)\t({self.value})\t[{color}]{self.token.value}[/{color}]")

	def get_write_handle(self) -> str | None:
		return self.token.value

@dataclass
class ASTLiteral(ASTValue):
	inner: Token

	def __str__(self) -> str:
		return str(self.inner.value)

	def run(self, interpreter: STInterpreter):
		try:
			self.value = int(self.inner.value)

		except ValueError:
			self.value = self.inner.value

@dataclass
class ASTFunctionCall(ASTValue):
	function: Token
	args: list[ASTValue]

	def run(self, interpreter: STInterpreter):
		color = "red"
		indicator = "?"

		if self.function.value == "set":
			try:
				[variable,] = self.args

			except ValueError:
				raise STSyntaxError(f"set function needs an argument")

			write_handle = variable.get_write_handle()

			if write_handle is None:
				raise STSyntaxError(f"{variable} is not writable")

			interpreter.set(write_handle, 1)
			color = "green"
			indicator = "WRITE"

		if self.function.value == "reset":
			try:
				[variable,] = self.args

			except ValueError:
				raise SyntaxError(f"reset function needs and argument")

			write_handle = variable.get_write_handle()

			if write_handle is None:
				raise STSyntaxError(f"{variable} is not writable")

			interpreter.set(write_handle, 0)
			color = "green"
			indicator = "WRITE"

		if self.function.value == "re":
			try:
				[variable,] = self.args

			except ValueError:
				raise SyntaxError(f"re function needs and argument")
			
			variable.run(interpreter)
			self.value = variable.value
			color = "green"
			indicator = "READ"

		args = ", ".join(str(e) for e in self.args)
		interpreter.console.print(f"({indicator})\t[{color}]{self.function.value} ({args}) [/{color}]")

@dataclass
class ASTComment(AST):
	token: Token

	def run(self, interpreter: STInterpreter):
		interpreter.console.print(f"(COM)\t{self.token.value}")

@dataclass
class ASTVariableAssignment(AST):
	variable: Token
	value: ASTValue

	def run(self, interpreter: STInterpreter):
		self.value.run(interpreter)
		interpreter.set(self.variable.value, self.value.value)
		interpreter.console.print(f"(WRITE)\t[green]{self.value.value} -> {self.variable.value}[/green]")

class Module:
	def run(self, interpreter: STInterpreter): ...

def get_precedence(token: Token) -> int:
	"""Return operator precedence, assuming higher values mean higher precedence."""
	return {
		"+": 100, "-": 100, "*": 200, "/": 200,
		"and": 60, "or": 50, "not": 75,
		"<=": 80, ">=": 80, "<>": 80, "=": 80, ">": 80, "<": 80,

	}.get(token.value, 0)

def apply_shunting_yard(tokens: list[Token]) -> list[Token]:
	output_queue: list[Token] = []
	operator_stack: list[Token] = []
	
	for token in tokens:
		if token.type in (TokenType.VARIABLE, TokenType.LITERAL, TokenType.PREPROCESSED):
			output_queue.append(token)

		elif token.type == TokenType.OPERATOR:
			while (operator_stack and operator_stack[-1].type == TokenType.OPERATOR and
				   (get_precedence(token) <= get_precedence(operator_stack[-1]))):
				output_queue.append(operator_stack.pop())

			operator_stack.append(token)

		elif token.type == TokenType.OPEN_P:
			operator_stack.append(token)

		elif token.type == TokenType.CLOSE_P:
			while operator_stack and operator_stack[-1].type != TokenType.OPEN_P:
				output_queue.append(operator_stack.pop())

			operator_stack.pop()  # Remove '(' from the stack
	
	while operator_stack:
		output_queue.append(operator_stack.pop())
	
	return output_queue

class ValueGatherer(Module):
	value: ASTValue

	def run(self, interpreter: STInterpreter):
		parts: list[Token] = [ ]
		level: int = 0

		while interpreter.token.type in (
			TokenType.VARIABLE, TokenType.OPEN_P, TokenType.CLOSE_P,
			TokenType.OPERATOR, TokenType.LITERAL
		):
			
			if interpreter.starts_with(TokenType.VARIABLE, TokenType.OPEN_P):
				gatherer = FunctionCallGatherer()
				gatherer.run(interpreter)
				ast = gatherer.function_call
				parts.append(PreprocessedToken(ast))
				continue

			if interpreter.token.type == TokenType.OPEN_P: level += 1
			if interpreter.token.type == TokenType.CLOSE_P:
				if level == 0: break
				level -= 1

			parts.append(interpreter.token)
			interpreter.advance()

		polish: list[Token] = apply_shunting_yard(parts)[::-1]
		ast_stack: list[ASTValue] = [ ]

		while polish:
			token = polish.pop()

			if token.type == TokenType.PREPROCESSED:
				assert isinstance(token, PreprocessedToken)
				ast_stack.append(token.ast)
				continue

			if token.type == TokenType.VARIABLE:
				ast_stack.append(ASTVariable(token))
				continue

			if token.type == TokenType.LITERAL:
				ast_stack.append(ASTLiteral(token))
				continue

			if token.type == TokenType.OPERATOR:
				args: list[ASTValue] = [ ]

				for _ in range(get_arg_count(token.value)):
					try:
						ast_arg = ast_stack.pop()

					except IndexError:
						raise STSyntaxError(f"operator {token.value} requires {get_arg_count(token.value)} arguments")

					args.insert(0, ast_arg)

				ast_stack.append(ASTOperator(token, args))
				continue

			raise Exception(f"unsupported token {token}")

		try:
			[value,] = ast_stack

		except ValueError:
			raise STSyntaxError(f"No value supplied where a value is required")

		assert isinstance(value, ASTValue)
		self.value = value

class ConditionalBodyGatherer(Module):
	conditional_body: ASTConditionalBody

	def run(self, interpreter: STInterpreter):
		interpreter.expect((TokenType.IF,))
		condition = ValueGatherer()
		condition.run(interpreter)
		self.conditional_body = ASTConditionalBody(
			condition.value, ASTBody([ ])
		)
		interpreter.expect((TokenType.THEN,))

		while interpreter.token.type != TokenType.END_IF:
			statement = StatementGatherer()
			statement.run(interpreter)
			self.conditional_body.body.children.append(statement.ast)

		interpreter.expect((TokenType.END_IF,))
		interpreter.expect((TokenType.END_INSTR,))

class UnkownInstructionGatherer(Module):
	instruction: ASTUnknownInstruction

	def run(self, interpreter: STInterpreter):
		instr: list[str] = [ ]

		while interpreter.token.type != TokenType.END_INSTR and interpreter.token.type != TokenType.EOF:
			instr.append(interpreter.token.value)
			interpreter.advance()

		interpreter.expect((TokenType.END_INSTR,))
		self.instruction = ASTUnknownInstruction(" ".join(instr) + " ;")

class VariableAssignmentGatherer(Module):
	assignment: ASTVariableAssignment

	def run(self, interpreter: STInterpreter):
		variable = interpreter.token
		interpreter.advance()
		interpreter.expect((TokenType.ASSIGNMENT,))
		gatherer = ValueGatherer()
		gatherer.run(interpreter)
		self.assignment = ASTVariableAssignment(variable, gatherer.value)

class FunctionCallGatherer(Module):
	function_call: ASTFunctionCall

	def run(self, interpreter: STInterpreter):
		function = interpreter.token
		interpreter.advance()
		interpreter.expect((TokenType.OPEN_P,))
		gatherer = ValueGatherer()
		gatherer.run(interpreter)
		interpreter.expect((TokenType.CLOSE_P,))
		self.function_call = ASTFunctionCall(function, [gatherer.value,])

class StatementGatherer(Module):
	ast: AST

	def run(self, interpreter: STInterpreter):
		if interpreter.token.type == TokenType.IF:
			conditional_body = ConditionalBodyGatherer()
			conditional_body.run(interpreter)
			self.ast = conditional_body.conditional_body
			return

		if interpreter.token.type == TokenType.COMMENT:
			self.ast = ASTComment(interpreter.token)
			interpreter.advance()
			return
		
		if interpreter.starts_with(TokenType.VARIABLE, TokenType.ASSIGNMENT):
			gatherer = VariableAssignmentGatherer()
			gatherer.run(interpreter)
			self.ast = gatherer.assignment
			interpreter.expect((TokenType.END_INSTR,))
			return

		if interpreter.starts_with(TokenType.VARIABLE, TokenType.OPEN_P):
			gatherer = FunctionCallGatherer()
			gatherer.run(interpreter)
			self.ast = gatherer.function_call
			interpreter.expect((TokenType.END_INSTR,))
			return
		
		if interpreter.starts_with(TokenType.EOF):
			raise STSyntaxError(f"Expected a statement, but reached EOF. Did you forget an end_if?")

		instruction = UnkownInstructionGatherer()
		instruction.run(interpreter)
		self.ast = instruction.instruction

def tokenize(text: str) -> list[Token]:
	tokens: list[Token] = [ ]

	while text:
		found_keyword = False

		if text[0] == "'":
			text = text[1:]

			try:
				end_index = text.index("'")

			except ValueError:
				raise STSyntaxError(f"string literal not closed")
			
			tokens.append(Token(TokenType.LITERAL, text[:end_index]))
			text = text[end_index + 1:]

		if text.startswith("(*"):
			text = text[2:]

			try:
				end_index = text.index("*)")

			except ValueError:
				raise STSyntaxError(f"comment not closed")
			
			tokens.append(Token(TokenType.COMMENT, text[:end_index]))
			text = text[end_index + 2:]

		if text.startswith(":="):
			tokens.append(Token(TokenType.ASSIGNMENT, ":="))
			text = text[2:]

		for keyword in KEYWORDS:
			if text.startswith(keyword):
				text = text[len(keyword):]
				tokens.append(Token(getType(keyword), keyword.lower()))
				found_keyword = True
				break

		if not found_keyword:
			if text.startswith("%"):
				length = 1

				while text[length - 1] in VAR_NAME_CHARS:
					length += 1

				tokens.append(Token(TokenType.VARIABLE, text[:length-1].lower()))
				text = text[length-1:]

			if text and text[0] in string.ascii_letters + string.digits:
				length = 1

				while text[length - 1] in string.ascii_letters + string.digits + ".#":
					length += 1

				if all(c in string.digits for c in text[:length-1]):
					tokens.append(Token(TokenType.LITERAL, text[:length-1]))

				else:
					tokens.append(Token(TokenType.VARIABLE, text[:length-1].lower()))
				
				text = text[length-1:]

		while text and text[0] in " \n\t":
			text = text[1:]

	return tokens + [ Token(TokenType.EOF, "EOF"), ]

def build_AST(interpreter: STInterpreter) -> ASTBody:
	body = ASTBody([ ])

	while interpreter.token.type != TokenType.EOF:
		statement = StatementGatherer()
		statement.run(interpreter)
		body.children.append(statement.ast)

	return body

def main():
	argument_parser = ArgumentParser(
		prog="st_interpreter",
		description="Runs ST code",
	)
	argument_parser.add_argument("file", help="file containing the code")
	argument_parser.add_argument("-f", "--flags", default="", help="display additional information -> R: raw code, T: tokens, A: AST, F: final interpreter state")
	argument_parser.add_argument("-v", "--verbose", action="store_true", help="display additional operations such as (R) Read operations")
	arguments = argument_parser.parse_args(argv[1:])

	args = Args()
	args.display_all_reads = arguments.verbose

	with open(arguments.file, "r", encoding="utf-8") as file:
		code = file.read() + " "

	console = Console()
	if "R" in arguments.flags: console.print(Panel.fit(code, title="Raw code"))
	tokens = tokenize(code)
	if "T" in arguments.flags: console.print(Panel.fit(Pretty(tokens), title="Tokens"))
	interpreter = STInterpreter(
		tokens, {
			"activation1": 1
		}, console, args,
	)
	ast = build_AST(interpreter)
	if "A" in arguments.flags: console.print(Panel.fit(Pretty(ast), title="AST"))
	console.rule("Code Execution")
	ast.run(interpreter)
	console.rule("End of code execution")
	if "F" in arguments.flags: console.print(Panel.fit(Pretty(interpreter), title="Final State"))

if __name__ == "__main__":
	main()
