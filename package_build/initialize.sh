#!/bin/bash
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Initialization script for building model-card-toolkit release packages.
#
# After this script is run, `python setup.py` commands can be run in the
# `model-card-toolkit/` package.

BASEDIR=$(dirname "$(pwd)/${0#./}")/..

mkdir -p $BASEDIR/dist

for CONFIG_NAME in model-card-toolkit
do
  ln -sf $BASEDIR/setup.py $BASEDIR/package_build/$CONFIG_NAME/
  ln -sf $BASEDIR/dist $BASEDIR/package_build/$CONFIG_NAME/
  ln -sf $BASEDIR/README*.md $BASEDIR/package_build/$CONFIG_NAME/
done
