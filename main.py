import gymnasium as gym
import ale_py
import torch
import argparse
import os
from tqdm       import tqdm
from dreamer    import Dreamer
from utils      import loadConfig, seedEverything, plotMetrics
from envs       import getEnvProperties, GymPixelsProcessingWrapper, CleanGymWrapper
from utils      import saveLossesToCSV, ensureParentFolders

# Register ALE roms if the current `ale_py` version exposes the helper.
# Some `ale_py` releases do not include `register_v5_envs`; guard the call
# so running on different environments won't crash with AttributeError.
try:
    register = getattr(ale_py, "register_v5_envs", None)
    if callable(register):
        register()
    else:
        # newer/older ale_py may not require explicit registration
        pass
except Exception as _e:
    print(f"Warning: ale_py.register_v5_envs() unavailable or failed: {_e}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optional Weights & Biases (wandb) integration. If `wandb` is installed,
# the script will initialize a run and log training/evaluation metrics.
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False


def main(configFile):
    config = loadConfig(configFile)
    seedEverything(config.seed)

    runName                 = f"{config.environmentName}_{config.runName}"
    checkpointToLoad        = os.path.join(config.folderNames.checkpointsFolder, f"{runName}_{config.checkpointToLoad}")
    metricsFilename         = os.path.join(config.folderNames.metricsFolder,        runName)
    plotFilename            = os.path.join(config.folderNames.plotsFolder,          runName)
    checkpointFilenameBase  = os.path.join(config.folderNames.checkpointsFolder,    runName)
    videoFilenameBase       = os.path.join(config.folderNames.videosFolder,         runName)
    ensureParentFolders(metricsFilename, plotFilename, checkpointFilenameBase, videoFilenameBase)
    # Initialize wandb run if available. Uses `wandbProject` from config if present.
    if _WANDB_AVAILABLE:
        try:
            wandb_project = config.get("wandbProject", "natural-dreamer")
        except Exception:
            wandb_project = "natural-dreamer"
        wandb.init(project=wandb_project, name=runName, config=config)
    
    env             = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(gym.make(config.environmentName), (64, 64))))
    envEvaluation   = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(gym.make(config.environmentName, render_mode="rgb_array"), (64, 64))))
    
    observationShape, isDiscrete, actionSize, actionLow, actionHigh = getEnvProperties(env)
    actionInfo = f"discrete with {actionSize} actions" if isDiscrete else f"continuous, size {actionSize}, range [{actionLow}, {actionHigh}]"
    print(f"envProperties: obs {observationShape}, action space: {actionInfo}")

    dreamer = Dreamer(observationShape, isDiscrete, actionSize, actionLow, actionHigh, device, config.dreamer)
    if config.resume:
        dreamer.loadCheckpoint(checkpointToLoad)

    dreamer.environmentInteraction(env, config.episodesBeforeStart, seed=config.seed)

    iterationsNum = config.gradientSteps // config.replayRatio
    with tqdm(total=config.gradientSteps, unit="step") as pbar:
        for _ in range(iterationsNum):
            for _ in range(config.replayRatio):
                sampledData                         = dreamer.buffer.sample(dreamer.config.batchSize, dreamer.config.batchLength)
                initialStates, worldModelMetrics    = dreamer.worldModelTraining(sampledData)
                behaviorMetrics                     = dreamer.behaviorTraining(initialStates)
                dreamer.totalGradientSteps += 1

                pbar.n = dreamer.totalGradientSteps
                pbar.set_postfix({"totalGradientSteps": dreamer.totalGradientSteps})
                pbar.refresh()

                if dreamer.totalGradientSteps % config.checkpointInterval == 0 and config.saveCheckpoints:
                    suffix = f"{dreamer.totalGradientSteps/1000:.0f}k"
                    dreamer.saveCheckpoint(f"{checkpointFilenameBase}_{suffix}")
                    evaluationScore = dreamer.environmentInteraction(envEvaluation, config.numEvaluationEpisodes, seed=config.seed, evaluation=True, saveVideo=True, filename=f"{videoFilenameBase}_{suffix}")
                    print(f"Saved Checkpoint and Video at {suffix:>6} gradient steps. Evaluation score: {evaluationScore:>8.2f}")
                    # Log evaluation score to wandb if available
                    if _WANDB_AVAILABLE:
                        wandb.log({"evaluation/score": evaluationScore}, step=dreamer.totalGradientSteps)

            mostRecentScore = dreamer.environmentInteraction(env, config.numInteractionEpisodes, seed=config.seed)
            # Save metrics and optionally log to wandb. `worldModelMetrics` and `behaviorMetrics`
            # correspond to the most recent gradient update; combine them with run counters.
            if config.saveMetrics:
                metricsBase = {"envSteps": dreamer.totalEnvSteps, "gradientSteps": dreamer.totalGradientSteps, "totalReward" : mostRecentScore}
                combined = metricsBase | worldModelMetrics | behaviorMetrics
                saveLossesToCSV(metricsFilename, combined)
                plotMetrics(f"{metricsFilename}", savePath=f"{plotFilename}", title=f"{config.environmentName}")
                if _WANDB_AVAILABLE:
                    # Log all metrics for this step to wandb
                    wandb.log(combined, step=dreamer.totalGradientSteps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="SpaceInvadersNoFrameskip-v4.yml")
    main(parser.parse_args().config)
