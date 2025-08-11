import os
import gym
import torch
import argparse
from gym.wrappers import AtariPreprocessing, frame_stack  # type: ignore

from agents import BCAgent
from policies import CNNPolicy
from dataset import AtariTemporalDifference
from infrastructure.trainer import OfflineRLTrainer


def main(args: argparse.Namespace) -> None:
    env = gym.make(
        args.env_name,
        frameskip=args.frame_skip,
        repeat_action_probability=args.sticky_actions
    )
    env = AtariPreprocessing(env, frame_skip=1, scale_obs=args.scale)
    policy = CNNPolicy(env.action_space.n, args.device)  # type: ignore
    agent = BCAgent(
        ema=args.ema,
        policy=policy,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        model_path=args.model_path,
        save_optimizer=args.save_optimizer,
        resume_training=args.resume_training,
        device=args.device
    )
    dataset = AtariTemporalDifference(
        args.dataset_path,
        part=args.data_seed,
        subset=args.data_subset,
        load_n_obs=args.load_n_obs,
    )
    trainer = OfflineRLTrainer(
        env=env,
        agent=agent,
        dataset=dataset,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        eval_freq=args.eval_freq,
        device=args.device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Pong-v0", help="Name of the Atari environment")
    parser.add_argument("--method", type=str, default="BC", help="Method to use for training - {BC, BVE}")
    parser.add_argument("--frame_skip", type=int, default=4, help="Frame skip for the Atari environment")
    parser.add_argument("--sticky_actions", type=float, default=0.25, help="Sticky actions probability")
    parser.add_argument("--scale", action="store_true", help="Scale observations to [0, 1]")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the dataset")
    parser.add_argument("--data_seed", type=int, default=1, help="Experimentation seed used for experimentation while collecting data - {1,2,3}")
    parser.add_argument("--data_subset", type=str, default="all", help="Subset of the data to use - {initial, final, all}")
    parser.add_argument("--ema", type=float, default=0.9999, help="Exponential moving average")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for the optimizer")
    parser.add_argument("--num_steps", type=int, default=100_000, help="Number of steps to train")
    parser.add_argument("--eval_freq", type=int, default=5_000, help="Number of steps between evaluations")
    parser.add_argument("--model_path", type=str, default="models", help="Path to save the model")
    parser.add_argument("--save_optimizer", action="store_true", help="Whether to save the optimizer state")
    parser.add_argument("--resume_training", action="store_true", help="Whether to resume training from a saved model")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()
    args.device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    assert args.env_name.endswith("-v0"), "Environment name must end with '-v0'"
    env_name = args.env_name[:-3]
    args.dataset_path = os.path.join(args.data_path, env_name)
    args.load_n_obs = not args.method == "BC"
    main(args)
