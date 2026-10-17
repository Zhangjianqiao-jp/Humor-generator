#!/bin/sh
set -eu
exec python "$(dirname "$0")/compare_training_free_interfaces.py" --scheme soft_distribution "$@"
