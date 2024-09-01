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
ps=('l' 'h') # 
ms=('large' 'reslin' 'lstm') # 
pres=(0 1 2 3 4 5 10 15 20 25 30)
ss=('full') # 

# Generate a 10-digit random number
ts=$(date +"%m%d%H%M%S")
# ts="0324233831"
# ts="0813184725"
# ts="0827104709"
echo "Timestamp: $ts"
# ts="0121181130"

# Loop from 1 to 10, incrementing by 1
for (( i=1; i<=5; i++ )); do
    # Loop over each combination of arguments
    python H_21_stage.py -ts "$ts-$i" -dp
    for p in "${ps[@]}"; do
        for s in "${ss[@]}"; do
            for m in "${ms[@]}"; do
                # Randomly select a GPU between 0 and 8
                gpu=$((RANDOM % 9))
                # post=$((55 - pre))

                # Run the Python script with the current combination of arguments in the background
                python H_21_stage.py -ts "$ts-$i" -p "$p" -s "$s" -m "$m" -gpu "$gpu" &
            done
        done
    done
done

# Wait for all background processes to finish
wait
