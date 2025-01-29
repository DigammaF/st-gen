
from __future__ import annotations
from enum import Enum
import string
from dataclasses import dataclass
from typing import Callable
from rich.console import Console

VAR_NAME_CHARS: str = string.ascii_letters + string.digits + ".%"
OPERATORS: tuple[str, ...] = (
	"not", "and", "or", "<=", ">=", "<>", "=", ">", "<",
	"+", "-", "not"
)
OPEN_P: tuple[str] = ("(",)
CLOSE_P: tuple[str] = (")",)
END_INSTR: tuple[str] = (";",)
IF: tuple[str] = ("if",)
END_IF: tuple[str] = ("end_if",)
THEN: tuple[str] = ("then",)
CONTROL_FLOW: tuple[str, ...] = THEN + END_INSTR + IF + END_IF
KEYWORDS: tuple[str, ...] = CONTROL_FLOW + OPERATORS + OPEN_P + CLOSE_P

class STSyntaxError(Exception): ...

@dataclass
class Token:
	type: TokenType
	value: str

class TokenType(Enum):
	OPERATOR = "OPERATOR"
	OPEN_P = "OPEN_P"
	CLOSE_P = "CLOSE_P"
	END_IF = "END_IF"
	IF = "IF"
	END_INSTR = "END_INSTR"
	THEN = "THEN"
	CONTROL_FLOW = "CONTROL_FLOW"
	EOF = "EOF"
	VARIABLE = "VARIABLE"

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
class STInterpreter:
	tokens: list[Token]
	variables: dict[str, int]
	console: Console
	running: bool

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

	def fetch(self, name: str) -> int:
		return self.variables.get(name, 0)

class AST:
	def run(self, interpreter: STInterpreter): ...

@dataclass
class ASTInstruction(AST):
	line: str

	def run(self, interpreter: STInterpreter):
		interpreter.console.print(self.line)

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

@dataclass
class ASTValue(AST):
	parts: list[Token]
	value: int

	def run(self, interpreter: STInterpreter):
		self.value = eval(" ".join(
			token.value
				if token.type != TokenType.VARIABLE
				else str(interpreter.fetch(token.value)) 
			for token in self.parts
		))

@dataclass
class ASTOperator(AST):
	operator: Token
	operands: list[AST]

	def run(self, interpreter: STInterpreter):
		pass

@dataclass
class ASTSymbolicValue(AST):
	parts: list[AST]
	value: int = 0

	def run(self, interpreter: STInterpreter):
		pass

@dataclass
class ASTVariable(AST):
	token: Token
	value: int = 0

	def run(self, interpreter: STInterpreter):
		pass

@dataclass
class ASTFunctionCall(AST):
	function: Token
	args: list[ASTSymbolicValue]

	def run(self, interpreter: STInterpreter):
		pass

class Module:
	def run(self, interpreter: STInterpreter): ...

class ValueGatherer(Module):
	value: ASTValue

	def run(self, interpreter: STInterpreter):
		self.value = ASTValue([ ], 0)

		while interpreter.token.type in (
			TokenType.VARIABLE, TokenType.OPEN_P, TokenType.CLOSE_P,
			TokenType.OPERATOR
		):
			self.value.parts.append(interpreter.token)
			interpreter.advance()

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

class InstructionGatherer(Module):
	instruction: ASTInstruction

	def run(self, interpreter: STInterpreter):
		instr: list[str] = [ ]

		while interpreter.token.type != TokenType.END_INSTR:
			instr.append(interpreter.token.value)
			interpreter.advance()

		interpreter.expect((TokenType.END_INSTR,))
		self.instruction = ASTInstruction(" ".join(instr) + " ;")

class StatementGatherer(Module):
	ast: AST

	def run(self, interpreter: STInterpreter):
		if interpreter.token.type == TokenType.IF:
			conditional_body = ConditionalBodyGatherer()
			conditional_body.run(interpreter)
			self.ast = conditional_body.conditional_body
			return

		instruction = InstructionGatherer()
		instruction.run(interpreter)
		self.ast = instruction.instruction

def tokenize(text: str) -> list[Token]:
	tokens: list[Token] = [ ]

	while text:
		found_keyword = False

		for keyword in KEYWORDS:
			if text.startswith(keyword):
				text = text[len(keyword):]
				tokens.append(Token(getType(keyword), keyword))
				found_keyword = True
				break

		if not found_keyword:
			if text.startswith("%"):
				length = 1

				while text[length - 1] in VAR_NAME_CHARS:
					length += 1

				tokens.append(Token(TokenType.VARIABLE, text[:length-1]))
				text = text[length-1:]

			if text and text[0] in string.ascii_letters + string.digits:
				length = 1

				while text[length - 1] in string.ascii_letters + string.digits:
					length += 1

				tokens.append(Token(TokenType.VARIABLE, text[:length-1]))
				text = text[length-1:]

		while text and text[0] in " \n":
			text = text[1:]

	return tokens + [ Token(TokenType.EOF, "EOF"), ]

def build_AST(interpreter: STInterpreter) -> ASTBody:
	body = ASTBody([ ])

	while interpreter.token.type != TokenType.EOF:
		statement = StatementGatherer()
		statement.run(interpreter)
		body.children.append(statement.ast)

	return body

TEXT = r"if ( Activation1 >= 0 and ((not %MW100.8) and %MW100.9 and (not %MW100.10) and %MW100.11 and (not %MW100.12) and (not %MW100.13)) ) or ( Activation1 >= 1 and ((not %MW100.14) and %MW100.15 and (not %MW101.0) and %MW101.1 and (not %MW101.2) and (not %MW101.3)) ) or ( Activation1 >= 2 and ((not %MW101.4) and %MW101.5 and (not %MW101.6) and %MW101.7 and (not %MW101.8) and (not %MW101.9)) ) then RESET(Tj1b); SET(Tj2b); end_if;"

def main():
	console = Console()
	console.print(TEXT)
	tokens = tokenize(TEXT)
	console.print(tokens)
	interpreter = STInterpreter(
		tokens, { }, console, True
	)
	ast = build_AST(interpreter)
	console.print(ast)
	ast.run(interpreter)

if __name__ == "__main__":
	main()
