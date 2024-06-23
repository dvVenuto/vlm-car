import argparse
import verification_pipeline

parser = argparse.ArgumentParser(description='Verification Pipeline Args')
parser.add_argument('--env_name', default="PushJoints-v3",
                    help='Mujoco Gym environment to verify')
parser.add_argument('--expert_traj_dir', default="/home",
                    help='The directory of the expert trajectories of the env')
parser.add_argument('--random_traj_dir', default="/home",
                    help='The directory of the random trajectories of the env')
parser.add_argument('--n_tasks', type=int, default=2,
                    help='The number of tasks used in VLM-CaR')
args = parser.parse_args()

verifier = verification_pipeline(args.env_name, args.expert_traj_dir, args.random_traj_dir, args.n_tasks)
checked = verifier.verify()

print("Verification was attempted and the result was: " + str(checked))
