import numpy 
import glob
import os

from GPT_scripts.doorkey import DoorKey8x8
from GPT_scripts.pandas import PandaGym
from GPT_scripts.opendoor import OpenDoor

class VerificationPipeline():

    def __init__(self, env, expert_traj_dir, random_traj_dir, n_tasks = 2):
        self.expert_traj_dir = expert_traj_dir
        self.random_traj_dir = random_traj_dir
        self.env=env
        self.n_tasks =n_tasks

        if str(self.env) == "PushJoints-v3" or str(self.env) == "SlideJoints-v3":
            self.checker = PandaGym()
        elif str(self.env) == "DoorKey6x6-v0" or str(self.env) == "DoorKey8x8-v0":
            self.checker = DoorKey8x8()
        elif str(self.env) == "Unlock-v0" or str(self.env) == "UnlockPickup-v0":
            self.checker == OpenDoor()
        else:
            print("Not a valid env")
            quit()

    def verify(self):
        
        #Verify random trajectories
        path = self.random_traj_dir + "/"

        lst = os.listdir(path)
        n_random_trajs = len(lst)

        random_completed_trajs = 0
        for i in range(n_random_trajs):
            completed_tasks = 0
            
            path = self.random_traj_dir + "/traj_" + str(i) + "/*"
            print(path)
            frames = glob.glob(path)
            frames = [x for x in frames]

            for frame in frames:
                completed = self.checker.check_and_progress(str(frame))
                print(completed)
                if completed != 0:
                    completed_tasks += 1
            if completed_tasks >= self.n_tasks:
                random_completed_trajs += 1

        if random_completed_trajs / n_random_trajs > 0.1:
            return False
        
        #Verify expert trajectories
        path = self.expert_traj_dir + "/"

        lst = os.listdir(path)
        n_expert_trajs = len(lst)

        expert_completed_trajs = 0
        for i in range(n_expert_trajs):
            completed_tasks = 0
            
            path = self.expert_traj_dir + "/traj_" + str(i) + "/*"
            frames = glob.glob(path)
            frames = [x for x in frames]

            for frame in frames:
                completed = self.checker.check_and_progress( str(frame))
                if completed != 0:
                    completed_tasks += 1
            if completed_tasks >= self.n_tasks:
                expert_completed_trajs += 1

        if expert_completed_trajs / n_expert_trajs == 1.0:
            return True
        else:
            return False
        

        
