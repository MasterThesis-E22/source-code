#!/bin/bash
base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"


echo "Starting experiment FedAVG"
$base_path/../venv/bin/python sync/sync_base.py -c sync/embryos/fedavg_10clients_1e4.yml
echo "Experiment FedAVG done"
echo -e "\n\n\n\n\n"
echo "=============================================================================================="
echo -e "\n\n\n\n\n"

echo "Starting experiment FedASYNC"
$base_path/../venv/bin/python async/async_base.py -c async/embryos/proposal_2.yml
echo "Experiment FedAVG done"

