# Top level control for managing the dev work

# ----- Project Macro ----- #
UnitTestCategory := AllTests
UnitTestName := MyAlgorithmTest
TestScript := test.py

# ------------------------------------------------------ #
# ------------- DO NOT MODIFY FROM BELOW ----------------#
# ------------------------------------------------------ #
mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_top  := $(dir $(mkfile_path))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

# ----- MACROS -----
MANTIDDIR := $(mkfile_top)/mantid
BUILDDIR  := $(mkfile_top)/build
INTALLDIR := $(mkfile_top)/opt/mantid
HOSTNAME  := $(shell hostname)
BASEOPTS  := -GNinja -DENABLE_MANTIDPLOT=OFF -DCMAKE_INSTALL_PREFIX=$(INTALLDIR) -DCMAKE_BUILD_TYPE=RelWithDebInfo

# ----- GDB -----
TestExecutable := $(BUILDDIR)/bin/AlgorithmsTest

# ----- BUILD OPTIONS -----
# ----- BUILD OPTIONS -----
ifneq (,$(findstring analysis,$(HOSTNAME)))
	# on analysis cluster, need to turn off jemalloc for
	# analysis.sns.gov
	CMKOPTS := $(BASEOPTS) -DUSE_JEMALLOC=OFF
	CMKCMDS := cmake3 $(MANTIDDIR) $(CMKOPTS)
	BLDCMDS := NINJA_STATUS="[%f+%r+%u=%t] ";ninja all $(UnitTestCategory) && ninja install ; true
else
	ifneq (,$(findstring ndav,$(HOSTNAME)))
		# on analysis cluster, need to turn off jemalloc for
		# ndav?.sns.gov
		CMKOPTS := $(BASEOPTS) -DUSE_JEMALLOC=OFF
		CMKCMDS := cmake3 $(MANTIDDIR) $(CMKOPTS)
		BLDCMDS := ninja all $(UnitTestCategory) && ninja install ; true
	else
		CMKOPTS := $(BASEOPTS)
		CMKCMDS := cmake $(MANTIDDIR) $(CMKOPTS)
		BLDCMDS := ninja -j3 all $(UnitTestCategory) && ninja install ; true
	endif
endif

# ----- UNIT TEST -----
UNTCMDS := ctest --output-on-failure -V -R $(UnitTestName)

# ----- Targets -----
.PHONY: test qtest build unittest docs init list clean archive

test: build docs unittest
	@echo "build everything, run unittest and customized testing script"
	$(INTALLDIR)/bin/mantidworkbench -x $(TestScript)

qtest: build
	@echo "quick test, no doc and unittest"
	$(INTALLDIR)/bin/mantidworkbench -x $(TestScript)

build:
	@echo "build mantid"
	@cd $(BUILDDIR); $(BLDCMDS)

unittest:
	@echo "run unittest"
	@cd $(BUILDDIR); $(UNTCMDS)

debugUnittest:
	@echo "run unittest with gdb"
	gdb --args $(TestExecutable) $(UnitTestName)

docs:
	@echo "build html docs"
	@cd $(BUILDDIR); ninja docs-html

# initialize the workproject, only need to be done once
init:
	@echo "deploying on host: ${HOSTNAME}"
	@echo "clone Mantid if not done already"
	@if [ ! -d "$(MANTIDDIR)" ]; then \
		git clone git@github.com:mantidproject/mantid.git; \
	fi
	@echo "switch to ornl-next branch"
	@cd $(MANTIDDIR); git checkout ornl-next
	@echo "make data directory, put testing data here"
	mkdir -p data
	@echo "make figure directory, save all figures here"
	mkdir -p figures
	@echo "config Mantid from scratch"
	mkdir -p ${BUILDDIR}
	mkdir -p ${INTALLDIR}
	@echo "running cmake"
	@cd ${BUILDDIR}; ${CMKCMDS}
	@echo "symbolic the build folder for vscode"
	@cd $(MANTIDDIR); ln -s ${BUILDDIR} .

reconfig:
	@echo "reconfig cmake"
	@cd ${BUILDDIR}; ${CMKCMDS}


# list all possible target in this makefile
list:
	@echo "LIST OF BUILD Targets from ninja"
	@cd $(BUILDDIR); ninja -t targets


# clean all tmp files
clean:
	@echo "Clean up workbench"
	rm  -fvr   *.tmp
	rm  -fvr   tmp_*
	rm  -fvr   build
	rm  -fvr   opt


# clean everything and archive the project
archive: clean
	rm  -fvr  mantid
