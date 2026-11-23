PYTHON ?= python3
PREFIX ?= /usr/local
FOCALOOP_PORTABLE_ISA ?= 1

.PHONY: all install-native install clean

all: install-native

install-native:
	FOCALOOP_PORTABLE_ISA=$(FOCALOOP_PORTABLE_ISA) "$(PYTHON)" scripts/build_native.py --prefix "$(PREFIX)"

install: install-native
	"$(PYTHON)" -m pip install . --no-deps --no-build-isolation -vv

clean:
	rm -rf build dist *.egg-info
