#!/bin/bash

# Function to generate a 10-digit random number
generate_random_number() {
    number=""
    for i in {1..10}; do
        digit=$((RANDOM % 10))
        number="${number}${digit}"
    done
    echo "$number"
}

# Arrays of options for each argument
ps=('l' 'h') # 'h'
ms=('cnn') # 'reslin' 'lstm'
pres=(0 1 2 3 4 5 10 15 20 25 30)
ss=('full') # 

# Generate a 10-digit random number
# ts=$(date +"%m%d%H%M%S")
ts="0823122642"
# ts="0324233831"
# ts="0813184725"
# ts="0827104709"
# ts="0905160507"
echo "Timestamp: $ts"
# ts="0121181130"

# Loop from 1 to 10, incrementing by 1
for (( i=1; i<=5; i++ )); do
    echo "Starting outer loop $i"
    # Loop over each combination of arguments
    # python A_04_experiment_4_ae_2.py -ts "$ts-$i" -dp
    for p in "${ps[@]}"; do
        echo "Starting p loop $p"
        for s in "${ss[@]}"; do
            for m in "${ms[@]}"; do
                echo "Starting model $m"
                # Randomly select a GPU between 0 and 8
                gpu=0
                # post=$((55 - pre))

                # Run the Python script with the current combination of arguments in the background
                python A_05_experiment_4_ae_2_clustering.py -ts "$ts-$i" -p "$p" -s "$s" -m "$m" -gpu "$gpu" -rn "$i" &
                wait
                echo "Finished model $m"
            done
        done
        echo "Finished p loop $p"
    done
    # Wait for all background processes to finish
    echo "Finished outer loop $i"
done
