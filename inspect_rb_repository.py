
from pathlib import Path
import ast

ROOT = (
    Path(__file__).resolve().parent
    / "external"
    / "commonroad-rb-planner"
)

FILES = {
    "Main entry point":
        "autonomous_driving/main_run_scenario.py",

    "Rulebook planner":
        "autonomous_driving/rpp_solver/rpp_solver.py",

    "MPC loop":
        "autonomous_driving/rpp_solver/run_rpp_solver_as_mpc.py",

    "Input data":
        "autonomous_driving/rpp_solver/common/data.py",

    "Trajectory sample":
        "autonomous_driving/rpp_solver/common/rpp_sample.py",
}

if not ROOT.is_dir():
    raise FileNotFoundError(
        f"Repository not found: {ROOT}"
    )

print("=" * 60)
print("COMMONROAD RULEBOOK PLANNER INSPECTION")
print("=" * 60)

for title, relative_path in FILES.items():
    path = ROOT / relative_path

    print(f"\n--- {title} ---")
    print("File:", relative_path)

    if not path.exists():
        print("MISSING FILE")
        continue

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            print(f"CLASS: {node.name}")

            for member in node.body:
                if isinstance(
                    member,
                    (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    print(f"    method: {member.name}()")

        elif isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            print(f"FUNCTION: {node.name}()")

print("\n" + "=" * 60)
print("AVAILABLE RULEBOOK FILES")
print("=" * 60)

rulebook_dir = (
    ROOT
    / "autonomous_driving"
    / "configurations"
    / "rulebooks"
)

for file in sorted(rulebook_dir.glob("*.txt")):
    print(file.name)

print("\n" + "=" * 60)
print("AUTONOMOUS DRIVING DEPENDENCIES")
print("=" * 60)

requirements = (
    ROOT
    / "autonomous_driving"
    / "requirements.txt"
)

print(requirements.read_text(encoding="utf-8"))
