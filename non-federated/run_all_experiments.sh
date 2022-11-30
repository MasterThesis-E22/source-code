#!/bin/bash
base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then
  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

experiments=("config/cifar10" "config/embryos-lowgpucnn" "config/embryos-mobilenet" "config/embryos-updatedlowgpucnn" "config/embryos-updatedlowgpucnn-classweight" "config/embryos-updatedlowgpucnn-oversampling" "config/mnist")
echo "=============================================================================================="
echo "Starting non-federated experiments"
echo "=============================================================================================="
for experiment in ${experiments[@]}; do
    echo "Starting experiment <$experiment>"
    $base_path/venv/bin/python experiment.py -c $experiment.yaml
    echo "Experiment <$experiment> done"
    echo -e "\n\n\n\n\n"
    echo "=============================================================================================="
    echo -e "\n\n\n\n\n"
done
echo "Finished executing experiments"