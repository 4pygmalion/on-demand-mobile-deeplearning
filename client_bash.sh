#!/bin/bash

# Get parameters
echo -e "Bandwidth (/kbps): "
read Bandwidth

echo -e "Direction type (single or bidrection): "
read direction

echo -e "SSH password"
read user_password

# 2. Run optimizer
exit_point=$(python3 ./optimizer.py $Bandwidth $direction)
echo "Exit point estimation finished:" "$exit_point"


runtimes=()
ITER=$(seq 0 50)
for i in $ITER
do
    # 3. Run deep learning on client
    START=$(date +%s.%N)
    python3 ./deep_learning_client.py $exit_point


    # 4. Send data
    sshpass -p $user_password scp -P10022 /mid_data.npy hoheon@localhost:~/mid_data_server.npy
    echo "Send complete"


    # 5. Connect to server
    result=$(sshpass -p $user_password ssh -p10022 hoheon@localhost "python3 ~/deep_learning_server.py $exit_point")


    END=$(date +%s.%N)
    runtime=$(echo "$END - $START" | bc)
    echo "$runtime" >> ./result/runtimes_total.txt
    
done 
