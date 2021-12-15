import sys
from pathlib import Path
import os
import shutil
import fire
import re


def main(target, upper=None, lower=None):
	if not upper and not lower:
		print(f"{upper = }, {lower = }, nothing to do.")
		return
	upper = int(upper) if upper is not None else None
	lower = int(lower) if lower is not None else None

	target = Path(target)
	assert target.exists(), target
	files = target.glob("*.ckpt")
	assert files, target


	for file_ in files:
		try:
			epoch = int(re.match('epoch=(\w+).*', file_.name).group(1))
		except AttributeError as err:
			# err.args += (f"{file_.name = }",)
			print(f"failed for {file_.name = }")
			continue
		if upper and epoch > upper:
			print(f"Removing file {file_.name} because {epoch = } > {upper = }")
			os.remove(file_)
		if lower and epoch < lower:
			print(f"Removing file {file_.name} because {epoch = } < {lower = }")
			os.remove(file_)			

		print(file_)
		print(epoch)


if __name__ == "__main__":
	fire.Fire(main)
