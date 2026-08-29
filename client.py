import sys
import pickle

import torch
import flwr as fl

from opacus import PrivacyEngine

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from config import (
    USE_DP,
    CLIENT_DIR,
    INITIAL_MODEL,
    BATCH_SIZE,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    FEDPROX_MU,
    DP_NOISE_MULTIPLIER,
    DP_MAX_GRAD_NORM,
    DP_DELTA
)

from model import (
    FusionNetwork,
    get_parameters,
    set_parameters,
    train_local,
    evaluate
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLIENT ID
# ============================================================

cid = int(sys.argv[1])

print(
    f"\nStarting Client {cid + 1}"
)


# ============================================================
# LOAD CLIENT DATA
# ============================================================

client_path = (
    CLIENT_DIR /
    f"client_{cid + 1}.pkl"
)

with open(
    client_path,
    "rb"
) as f:

    data = pickle.load(f)


dataset = TensorDataset(
    data["Cp"],
    data["Cd"],
    data["Cy"],
    data["CS"]
)


# ============================================================
# FLOWER CLIENT
# ============================================================

class Client(fl.client.NumPyClient):

    def __init__(self):

        # ----------------------------------------------------
        # Initial plain model
        # ----------------------------------------------------

        self.model = FusionNetwork().to(
            DEVICE
        )

        self.model.load_state_dict(
            torch.load(
                INITIAL_MODEL,
                map_location=DEVICE
            )
        )

        print(
            "Loaded centralized PCCF model."
        )

        if USE_DP:

            print(
                "Differential Privacy: ENABLED"
            )

            print(
                f"Noise Multiplier: "
                f"{DP_NOISE_MULTIPLIER}"
            )

            print(
                f"Max Grad Norm: "
                f"{DP_MAX_GRAD_NORM}"
            )

            print(
                f"Delta: "
                f"{DP_DELTA}"
            )

        else:

            print(
                "Differential Privacy: DISABLED"
            )


    # ========================================================
    # GET PARAMETERS
    # ========================================================

    def get_parameters(
        self,
        config
    ):

        return get_parameters(
            self.model
        )


    # ========================================================
    # SET PARAMETERS
    # ========================================================

    def set_parameters(
        self,
        parameters
    ):

        set_parameters(
            self.model,
            parameters
        )


    # ========================================================
    # LOCAL TRAINING
    # ========================================================

    def fit(
        self,
        parameters,
        config
    ):

        # ----------------------------------------------------
        # Create a FRESH model for this round
        # ----------------------------------------------------

        self.model = FusionNetwork().to(
            DEVICE
        )

        self.set_parameters(
            parameters
        )

        # IMPORTANT:
        # Opacus requires the model to be in training mode
        # before make_private() is called.

        self.model.train()


        # ----------------------------------------------------
        # Create a fresh DataLoader for this round
        # ----------------------------------------------------

        train_loader = DataLoader(

            dataset,

            batch_size=BATCH_SIZE,

            shuffle=True

        )


        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer = torch.optim.Adam(

            self.model.parameters(),

            lr=LEARNING_RATE

        )


        # ----------------------------------------------------
        # Save global parameters for FedProx
        # ----------------------------------------------------

        global_params = [

            p.detach().clone()

            for p in self.model.parameters()

        ]


        # ----------------------------------------------------
        # DIFFERENTIAL PRIVACY
        # ----------------------------------------------------

        privacy_engine = None

        if USE_DP:

            privacy_engine = PrivacyEngine()

            (
                self.model,
                optimizer,
                train_loader
            ) = privacy_engine.make_private(

                module=self.model,

                optimizer=optimizer,

                data_loader=train_loader,

                noise_multiplier=DP_NOISE_MULTIPLIER,

                max_grad_norm=DP_MAX_GRAD_NORM

            )


        # ----------------------------------------------------
        # LOCAL FEDPROX TRAINING
        # ----------------------------------------------------

        loss = train_local(

            self.model,

            train_loader,

            optimizer,

            DEVICE,

            mu=FEDPROX_MU,

            global_params=global_params,

            epochs=LOCAL_EPOCHS

        )


        # ====================================================
        # DP PRIVACY ACCOUNTING
        # ====================================================

        metrics = {

            "loss": float(loss)

        }


        if USE_DP:

            epsilon = (
                privacy_engine.get_epsilon(
                    delta=DP_DELTA
                )
            )

            metrics["epsilon"] = float(
                epsilon
            )

            metrics["delta"] = float(
                DP_DELTA
            )

            print(
                f"Client {cid + 1} "
                f"Loss: {loss:.6f}"
            )

            print(
                f"Privacy epsilon: "
                f"{epsilon:.4f}"
            )

            print(
                f"Privacy delta: "
                f"{DP_DELTA}"
            )

            # ------------------------------------------------
            # IMPORTANT:
            # Remove Opacus wrapper before sending the
            # model parameters back to Flower.
            # ------------------------------------------------

            if hasattr(
                self.model,
                "_module"
            ):

                self.model = (
                    self.model._module
                )

        else:

            print(
                f"Client {cid + 1} "
                f"Loss: {loss:.6f}"
            )


        # ----------------------------------------------------
        # Make sure returned model is plain PyTorch model
        # ----------------------------------------------------

        self.model = self.model.to(
            DEVICE
        )

        self.model.eval()


        # ----------------------------------------------------
        # Return model parameters
        # ----------------------------------------------------

        return (

            get_parameters(
                self.model
            ),

            len(dataset),

            metrics

        )


    # ========================================================
    # CLIENT EVALUATION
    # ========================================================

    def evaluate(
        self,
        parameters,
        config
    ):

        # ----------------------------------------------------
        # Fresh plain model for evaluation
        # ----------------------------------------------------

        self.model = FusionNetwork().to(
            DEVICE
        )

        self.set_parameters(
            parameters
        )

        eval_loader = DataLoader(

            dataset,

            batch_size=BATCH_SIZE,

            shuffle=False

        )


        # ----------------------------------------------------
        # Calculate reconstruction errors
        # ----------------------------------------------------

        errors = evaluate(

            self.model,

            eval_loader,

            DEVICE

        )


        loss = (
            sum(errors)
            / len(errors)
        )


        return (

            float(loss),

            len(dataset),

            {}

        )


# ============================================================
# START CLIENT
# ============================================================

fl.client.start_client(

    server_address="127.0.0.1:8080",

    client=Client().to_client()

)