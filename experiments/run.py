import os
import sys
import time
import subprocess
from pathlib import Path


# PATHS

ROOT = Path(__file__).resolve().parents[1]

ABLATION_DIR = (
    ROOT
    / "experiments"
)


# SELECT EXPERIMENT
print(
    "\nPCCF DIFFERENTIAL PRIVACY ABLATION STUDY\n"
)


print(
    "\nSelect experiment:"
)

print(
    "1 - With Differential Privacy (E4_FL_DP)"
)

print(
    "0 - Without Differential Privacy (E1_No_DP)"
)


while True:

    choice = input(
        "\nEnter choice [1/0]: "
    ).strip()

    if choice in ("0", "1"):

        break

    print(
        "Invalid choice. "
        "Please enter 1 or 0."
    )


# SET EXPERIMENT ENVIRONMENT

env = os.environ.copy()

env["PCCF_USE_DP"] = choice


if choice == "1":

    experiment = "E4_FL_DP"

else:

    experiment = "E1_No_DP"

print(
    f"Selected Experiment: "
    f"{experiment}"
)

# START SERVER

print(
    "\nStarting Flower server..."
)


server_process = subprocess.Popen(

    [
        sys.executable,
        "server.py"
    ],

    cwd=ABLATION_DIR,

    env=env
)


# Give the server time to start

time.sleep(5)


# START CLIENTS

client_processes = []


for cid in range(3):

    print(
        f"Starting Client {cid + 1}..."
    )

    process = subprocess.Popen(

        [
            sys.executable,
            "client.py",
            str(cid)
        ],

        cwd=ABLATION_DIR,

        env=env
    )

    client_processes.append(
        process
    )


# WAIT FOR CLIENTS

try:

    for process in client_processes:

        process.wait()


    # Wait for the server to finish

    server_process.wait()


except KeyboardInterrupt:

    print(
        "\nStopping experiment..."
    )

    for process in client_processes:

        process.terminate()


    server_process.terminate()

    sys.exit(1)


# CHECK SERVER STATUS

if server_process.returncode != 0:

    print(
        "\nServer terminated with an error."
    )

    sys.exit(
        server_process.returncode
    )


# RUN EVALUATION

print(
    "\nFederated training completed."
)

print(
    "Starting evaluation..."
)


evaluation = subprocess.run(

    [
        sys.executable,
        "evaluate.py"
    ],

    cwd=ABLATION_DIR,

    env=env
)


# FINAL STATUS

if evaluation.returncode == 0:

    print("\n")

    print(
        "EXPERIMENT COMPLETED SUCCESSFULLY"
    )

    print(
        f"Experiment: {experiment}"
    )

else:

    print(
        "\nEvaluation failed."
    )

    sys.exit(
        evaluation.returncode
    )