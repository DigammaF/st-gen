
from rich.console import Console

from main import ACTIONS, BRANCHES, ENABLERS, INVERSERS, SECTORS, SENSORS, TRAINS, Bits

def main():
	console = Console()
	
	for key, action in enumerate(ACTIONS):
		console.print(f"{key}: {action}")

	console.print("secteurs: " + " - ".join(SECTORS))
	console.print("inverseurs: " + " - ".join(INVERSERS))
	console.print("aiguillages: " + " - ".join(BRANCHES))
	console.print("capteurs: " + " - ".join(SENSORS))

	TRAIN_CODES: list[Bits] = [ ]
	for name, code in zip(TRAINS, Bits.iter_states(2)):
		TRAIN_CODES.append(code)

	ACTION_CODES: list[Bits] = [ ]
	for _, code in zip(ACTIONS, Bits.iter_states(2)):
		ACTION_CODES.append(code)

	ENABLER_CODES: list[Bits] = [ ]
	for _, code in zip(ENABLERS, Bits.iter_states(2)):
		ENABLER_CODES.append(code)

	SECTOR_CODES: dict[str, Bits] = { }
	for name, code in zip(SECTORS, Bits.iter_states(6)):
		SECTOR_CODES[name] = code

	BRANCH_CODES: dict[str, Bits] = { }
	for name, code in zip(BRANCHES, Bits.iter_states(6)):
		BRANCH_CODES[name] = code

	INVERSER_CODES: dict[str, Bits] = { }
	for name, code in zip(INVERSERS, Bits.iter_states(6)):
		INVERSER_CODES[name] = code

	SENSOR_CODES: dict[str, Bits] = { }
	for name, code in zip(SENSORS, Bits.iter_states(6)):
		SENSOR_CODES[name] = code

	train_code: Bits = TRAIN_CODES[int(console.input("Train: ")) - 1]
	console.print(train_code.raw_str())

	while True:
		command: list[str] = console.input(": ").split(" ")
		action_code: Bits = ACTION_CODES[int(command.pop(0))]
		enable_code: Bits = ENABLER_CODES[len(command) - 1]
		codes: list[Bits] = [ ]
		
		if command[0] == 0:
			sensor_code = SENSOR_CODES[command.pop()]
			console.print(command)
			console.print(ENABLER_CODES)
			enable_code = ENABLER_CODES[len(command) - 1]
			while command: codes.append(SECTOR_CODES[command.pop(0)])
			codes.append(sensor_code)

		console.print(f"{train_code.raw_str()} {action_code.raw_str()} 11 {enable_code.raw_str()} {''.join(code.raw_str() for code in codes)}")

if __name__ == "__main__":
	main()

