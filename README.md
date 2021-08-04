# MantidWorkbench_dev_template
This is a template for setting up a development repo for MantidWorkbench related tasks.

# Folder renaming convention

$Type$TaskNumber_$Abbre

# Steps

- use `make init` to clone mantid and setup cmake
- go into folder `mantid` to setup you branch
- go back to proj folder, use `make build` to kick off building
- edit relevant targets in Makefile
- developing
  - use `make qtest` if a testing script `test.py` is used
  - use `make unittest` if a unit test is used
