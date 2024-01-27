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
ps=('f' 'l' 'h')
ms=('large')
ss=('c' 'v')

# Generate a 10-digit random number
random_number=$(generate_random_number)
echo "Timestamp: $random_number"

python H_15_cvcv.py -ts "$random_number" -dp

# Loop over each combination of arguments
for p in "${ps[@]}"; do
    for m in "${ms[@]}"; do
        for s in "${ss[@]}"; do
            # Randomly select a GPU between 0 and 8
            gpu=$((RANDOM % 9))

            # Run the Python script with the current combination of arguments in the background
            python H_15_cvcv.py -ts "$random_number" -p "$p" -m "$m" -s "$s" -gpu "$gpu" &

        done
    done
done

# Wait for all background processes to finish
wait
