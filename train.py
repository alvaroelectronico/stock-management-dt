from torch import optim
from decision_transformer import DecisionTransformer
from decision_transformer_config import DecisionTransformerConfig
from trainer import TrainerConfig
from decision_transformer_trainer import DecisionTransformerTrainer
from decision_transformer_strategies import DTTrainingStrategy
from pathlib import Path

def getProjectDirectory():
        return str(Path(__file__).resolve().parent)

if __name__ == "__main__":

    decisionTransformerConfig = DecisionTransformerConfig(
        # contexto
        hidden_size=192,
        context_feed_feedforward_dim=192*5,
        encoder_number_layers=6,
        encoder_number_heads=12,
        mha_number_heads=12,        

        # dt
        n_layer=6,
        n_head=8,
        n_inner=192*5,

        max_episode_length=20,
        max_context_length=12,
        activation_function="relu"
    )

    model = DecisionTransformer(decisionTransformerConfig)
    stepsPerEpoch = 20000
    #print(f"{sum(t.numel() for t in model.rewardPredictor.parameters()) / 1000 ** 2:.1f}M")

    optimizer = optim.Adam(model.parameters(), lr=0.001,  betas=(0.9, 0.999),  eps=1e-08, weight_decay=0)
    scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    trainStrategy = DTTrainingStrategy(
        dataPath=(getProjectDirectory() +"\\data\\training_data2.pt"),
        trainPercentage=[1],
        augment=True
    )

    trainerConfig = TrainerConfig(
        nBatch=1,
        nVal=400,
        stepsPerEpoch=stepsPerEpoch,
        trainStrategy=trainStrategy,
        optimizer=optimizer,
        lr_scheduler=scheduler
    )

    savePath = (getProjectDirectory() +
                "/training_models/decision_transformer/")
    trainer = DecisionTransformerTrainer(savePath=savePath, name="v4", model=model, trainerConfig=trainerConfig)
    trainer.initTraining()
