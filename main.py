
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator

from rich.console import Console
from rich.table import Table

CONSOLE = Console(record=True)

TRAINS = ("Train 1", "Train 2", "Train 3")
CALLBACK_ADDRS = ("'{9.1}SYS'", "'{9.2}SYS'", "'{9.3}SYS'")
CALLBACK_MEMORIES = ("102", "2", "3")
CALLBACK_COUNTS = ("1", "1", "1")
ACTIONS = (
	"Allumer/Eteindre tronçon",
	"Déclencher inverseur",
	"Aiguiller droit",
	"Aiguiller bifurqué"
)
ENABLERS = (
	"1 Element (M)",
	"2 Elements (M et N)",
	"3 Elements (M et N et P)"
)
SECTORS = tuple(f"Tn{n}" for n in range(0, 7 + 1)) + tuple(f"T{n}" for n in range(8, 19 + 1))
INVERSERS = tuple(f"Ti{n}" for n in range(0, 7 + 1))
BRANCHES_STRAIGHT = tuple(f"A{n}d" for n in range(0, 8 + 1)) + tuple(f"Tj{n}d" for n in range(0, 2 + 1))
BRANCHES_BRANCH = tuple(f"A{n}b" for n in range(0, 8 + 1)) + tuple(f"Tj{n}b" for n in range(0, 2 + 1))
BRANCHES = tuple(f"A{n}" for n in range(0, 8 + 1)) + tuple(f"Tj{n}" for n in range(0, 2 + 1))
SENSORS = tuple(f"C{n}" for n in range(0, 39 + 1))

# variables
CALLBACK_ADDR = "Address1"
CALLBACK_MEMORY = "Destination1"
CALLBACK_COUNT = "Count1"
TEMPORISATION300 = "ExecuterAction.t>=t#300ms"
TEMPORISATION50 = "ExecuterAction.t>=t#50ms"

TRAIN = "Train1"
ACTION = "Action1"
ENABLER = "Activation1"

@dataclass
class Selector:
	base: int
	offset: int
	size: int

	def __getitem__(self, key: int) -> str:
		assert 0 <= key < self.size
		bit_position = self.offset + key
		base = self.base

		if bit_position > 15:
			bit_position -= 16
			base -= 1

		return f"%MW{base}.{bit_position}"

class Request:
	base = 101
	train = Selector(base, 30, 2)
	action = Selector(base, 28, 2)
	activity = Selector(base, 26, 2)
	enabler = Selector(base, 24, 2)
	A = Selector(base, 18, 6)
	B = Selector(base, 12, 6)
	C = Selector(base, 6, 6)
	sensor = Selector(base, 0, 6)

def equals(variable: Selector, value: Bits) -> str:
	assert variable.size == len(value)
	return "(" + " and ".join(
		variable[n] if value[len(value) - 1 - n] else f"(not {variable[n]})"
		for n in range(len(value))
	) + ")"

class NotEnoughStates(Exception): ...

@dataclass
class Bits:
	_values: tuple[int, ...] # RMB is LSB

	def __str__(self) -> str:
		return "".join(Bits.color(e) for e in self._values)

	def raw_str(self) -> str:
		return "".join(str(e) for e in self._values)

	def __len__(self) -> int:
		return len(self._values)

	@staticmethod
	def color(bit: int) -> str:
		return str(bit) if bit == 0 else f"[blue]{bit}[/blue]"

	@property
	def ST(self) -> str:
		return "2#" + f"".join(str(e) for e in self._values)

	def incremented(self) -> Bits:
		result = list(self._values)

		for n in range(len(result) - 1, 0 -1, -1):
			print(n)
			if result[n]:
				result[n] = 0

			else:
				result[n] = 1
				return Bits(tuple(result))
			
		raise NotEnoughStates

	def __getitem__(self, key: int) -> int:
		return self._values[key]
	
	@staticmethod
	def iter_states(bit_count: int) -> Iterator[Bits]:
		assert bit_count > 0
		n = Bits((0,)*bit_count)

		for _ in range(2**bit_count):
			yield n
			n = n.incremented()

class If:
	_condition: str

	def __init__(self, condition: str) -> None:
		self._condition = condition

	def __enter__(self):
		CONSOLE.print(f"if {self._condition} then")

	def __exit__(self, *_, **__):
		CONSOLE.print("end_if;")

def main():
	CONSOLE.print("""
Une commande de ce type permet d'impacter jusqu'à trois éléments de la maquette
Si l'action est l'allumage/extinction d'un tronçon, le quatrième champ (Q) renseigne
un capteur qui va provoquer l'extinction des tronçons lorsque déclenché.

T: Train
A: Action
E: Activateur
M: Premier élément
N: Deuxième élément
P: Troisième élément
Q: Capteur

0000 1100 0000 0000 0000 0000 0000 0000
TTAA --EE MMMM MMNN NNNN PPPP PPQQ QQQQ
			   
(Les bits '--' sont toujours à 1)
""")

	table = Table("Nom", "Code", title="Trains")
	TRAIN_CODES: list[Bits] = [ ]
	for name, code in zip(TRAINS, Bits.iter_states(2)):
		table.add_row(name, str(code))
		TRAIN_CODES.append(code)

	CONSOLE.print(table)

	table = Table("Nom", "Code", title="Actions")
	ACTION_CODES: list[Bits] = [ ]
	for name, code in zip(ACTIONS, Bits.iter_states(2)):
		table.add_row(name, str(code))
		ACTION_CODES.append(code)
	
	CONSOLE.print(table)

	table = Table("Nom", "Code", title="Activations")
	ENABLER_CODES: list[Bits] = [ ]
	for name, code in zip(ENABLERS, Bits.iter_states(2)):
		table.add_row(name, str(code))
		ENABLER_CODES.append(code)

	CONSOLE.print(table)

	table = Table("Nom", "Code", title="Tronçons")
	SECTOR_CODES: list[Bits] = [ ]
	for name, code in zip(SECTORS, Bits.iter_states(6)):
		table.add_row(name, str(code))
		SECTOR_CODES.append(code)

	CONSOLE.print(table)

	table = Table("Nom", "Code", title="Aiguillages")
	BRANCH_CODES: list[Bits] = [ ]
	for name, code in zip(BRANCHES, Bits.iter_states(6)):
		table.add_row(name, str(code))
		BRANCH_CODES.append(code)

	CONSOLE.print(table)

	table = Table("Nom", "Code", title="Inverseurs")
	INVERSER_CODES: list[Bits] = [ ]
	for name, code in zip(INVERSERS, Bits.iter_states(6)):
		table.add_row(name, str(code))
		INVERSER_CODES.append(code)

	CONSOLE.print(table)

	table = Table("Nom", "Code", title="Capteurs")
	SENSOR_CODES: list[Bits] = [ ]
	for name, code in zip(SENSORS, Bits.iter_states(6)):
		table.add_row(name, str(code))
		SENSOR_CODES.append(code)

	CONSOLE.print(table)

	CONSOLE.rule()
	# ------------------------------------------------------------------------------------------------------

	for train_name, addr, memory, count, train_code in zip(
		TRAINS, CALLBACK_ADDRS, CALLBACK_MEMORIES, CALLBACK_COUNTS, TRAIN_CODES
	):
		with If(equals(Request.train, train_code)):
			CONSOLE.print(f"(* {train_name} *)")
			CONSOLE.print(f"{CALLBACK_ADDR} := {addr}; {CALLBACK_MEMORY} := {memory}; {CALLBACK_COUNT} := {count};")
			CONSOLE.print(f"{TRAIN} := {TRAINS.index(train_name)};")

	for action_name, action_code in zip(ACTIONS, ACTION_CODES):
		with If(equals(Request.action, action_code)):
			CONSOLE.print(f"(* {action_name} *)")
			CONSOLE.print(f"{ACTION} := {ACTIONS.index(action_name)};")

	for enabler_name, enabler_code in zip(ENABLERS, ENABLER_CODES):
		with If(equals(Request.enabler, enabler_code)):
			CONSOLE.print(f"(* {enabler_name} *)")
			CONSOLE.print(f"{ENABLER} := {ENABLERS.index(enabler_name)};")

	# sectors
	with If(f"{ACTION} = 0"):
		CONSOLE.print(f"(* {ACTIONS[0]} *)")
		
		for sector_name, sector_code in zip(SECTORS, SECTOR_CODES):
			a = f"( {ENABLER} >= 0 and {equals(Request.A, sector_code)} )"
			b = f"( {ENABLER} >= 1 and {equals(Request.B, sector_code)} )"
			c = f"( {ENABLER} >= 2 and {equals(Request.C, sector_code)} )"

			with If(" or ".join((a, b, c))):
				CONSOLE.print(f"SET({sector_name});")

	# inverser
	with If(f"{ACTION} = 1"):
		CONSOLE.print(f"(* {ACTIONS[1]} *)")

		for inverser_name, inverser_code in zip(INVERSERS, INVERSER_CODES):
			a = f"( {ENABLER} >= 0 and {equals(Request.A, inverser_code)} )"
			b = f"( {ENABLER} >= 1 and {equals(Request.B, inverser_code)} )"
			c = f"( {ENABLER} >= 2 and {equals(Request.C, inverser_code)} )"

			with If(" or ".join((a, b, c))):
				CONSOLE.print(f"SET({inverser_name});")

	# branch straight
	with If(f"{ACTION} = 2"):
		CONSOLE.print(f"(* {ACTIONS[2]} *)")

		for branch_name, branch_code in zip(BRANCHES_STRAIGHT, BRANCH_CODES):
			a = f"( {ENABLER} >= 0 and {equals(Request.A, branch_code)} )"
			b = f"( {ENABLER} >= 1 and {equals(Request.B, branch_code)} )"
			c = f"( {ENABLER} >= 2 and {equals(Request.C, branch_code)} )"

			with If(" or ".join((a, b, c))):
				CONSOLE.print(f"SET({branch_name});")

	# branch branch
	with If(f"{ACTION} = 3"):
		CONSOLE.print(f"(* {ACTIONS[3]} *)")

		for branch_name, branch_code in zip(BRANCHES_BRANCH, BRANCH_CODES):
			a = f"( {ENABLER} >= 0 and {equals(Request.A, branch_code)} )"
			b = f"( {ENABLER} >= 1 and {equals(Request.B, branch_code)} )"
			c = f"( {ENABLER} >= 2 and {equals(Request.C, branch_code)} )"

			with If(" or ".join((a, b, c))):
				CONSOLE.print(f"SET({branch_name});")

	CONSOLE.rule()
	# ------------------------------------------------------------------------------------------------------
	condition: list[str] = [ ]

	# sector
	disjunction: list[str] = [ ]

	for sensor_name, sensor_code in zip(SENSORS, SENSOR_CODES):
		disjunction.append(f"( {equals(Request.sensor, sensor_code)} and {sensor_name} ) ")

	condition.append(f"( {ACTION} = 0 and ( {" or ".join(disjunction)} ) ) ")

	# inverser
	condition.append(f"( {ACTION} = 1 and {TEMPORISATION50} )")

	# branch
	condition.append(f"( ( {ACTION} = 2 or {ACTION} = 3 ) and {TEMPORISATION300} )")

	CONSOLE.print(" or ".join(condition))
	CONSOLE.rule()
	# ------------------------------------------------------------------------------------------------------

	# sectors
	with If(f"{ACTION} = 0"):
		CONSOLE.print(f"(* {ACTIONS[0]} *)")
		
		for sector_name, sector_code in zip(SECTORS, SECTOR_CODES):
			a = f"( {ENABLER} >= 0 and {equals(Request.A, sector_code)} )"
			b = f"( {ENABLER} >= 1 and {equals(Request.B, sector_code)} )"
			c = f"( {ENABLER} >= 2 and {equals(Request.C, sector_code)} )"

			with If(" or ".join((a, b, c))):
				CONSOLE.print(f"RESET({sector_name});")

	# inverser
	with If(f"{ACTION} = 1"):
		CONSOLE.print(f"(* {ACTIONS[1]} *)")

		for inverser_name, inverser_code in zip(INVERSERS, INVERSER_CODES):
			a = f"( {ENABLER} >= 0 and {equals(Request.A, inverser_code)} )"
			b = f"( {ENABLER} >= 1 and {equals(Request.B, inverser_code)} )"
			c = f"( {ENABLER} >= 2 and {equals(Request.C, inverser_code)} )"

			with If(" or ".join((a, b, c))):
				CONSOLE.print(f"RESET({inverser_name});")

	# branch straight
	with If(f"{ACTION} = 2"):
		CONSOLE.print(f"(* {ACTIONS[2]} *)")

		for branch_name, branch_code in zip(BRANCHES_STRAIGHT, BRANCH_CODES):
			a = f"( {ENABLER} >= 0 and {equals(Request.A, branch_code)} )"
			b = f"( {ENABLER} >= 1 and {equals(Request.B, branch_code)} )"
			c = f"( {ENABLER} >= 2 and {equals(Request.C, branch_code)} )"

			with If(" or ".join((a, b, c))):
				CONSOLE.print(f"RESET({branch_name});")

	# branch branch
	with If(f"{ACTION} = 3"):
		CONSOLE.print(f"(* {ACTIONS[3]} *)")

		for branch_name, branch_code in zip(BRANCHES_BRANCH, BRANCH_CODES):
			a = f"( {ENABLER} >= 0 and {equals(Request.A, branch_code)} )"
			b = f"( {ENABLER} >= 1 and {equals(Request.B, branch_code)} )"
			c = f"( {ENABLER} >= 2 and {equals(Request.C, branch_code)} )"

			with If(" or ".join((a, b, c))):
				CONSOLE.print(f"RESET({branch_name});")

	CONSOLE.rule()
	CONSOLE.print(f"Activity: {Request.activity[0]}")

	for select in (
		Request.train, Request.action, Request.activity,
		Request.enabler, Request.A, Request.B, Request.C,
		Request.sensor
	):
		for n in range(select.size):
			CONSOLE.print(select[n])

	CONSOLE.save_html("doc.html")

if __name__ == "__main__":
	main()
