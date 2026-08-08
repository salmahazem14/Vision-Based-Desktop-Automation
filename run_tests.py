"""Minimal pytest-free runner (sandbox has no pytest). Discovers test_* funcs."""
import importlib.util, sys, traceback, pathlib
sys.path.insert(0, str(pathlib.Path("src").resolve()))
passed = failed = 0
for tf in sorted(pathlib.Path("tests").glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(tf.stem, tf)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        if name.startswith("test_"):
            fn = getattr(mod, name)
            if callable(fn):
                try:
                    fn(); passed += 1
                    print(f"PASS {tf.stem}::{name}")
                except Exception:
                    failed += 1
                    print(f"FAIL {tf.stem}::{name}")
                    traceback.print_exc()
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
