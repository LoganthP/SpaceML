import os

# Tests never reach the network (no first-launch download job) and start from an empty
# installation unless a test points SPACEML_STARTER_DIR at a pack on purpose.
os.environ.setdefault("SPACEML_BOOTSTRAP", "0")
os.environ.setdefault("SPACEML_STARTER_DIR", os.path.join(os.path.dirname(__file__), "_no_starter_pack_"))
