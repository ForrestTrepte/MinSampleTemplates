#!/usr/bin/env python3

import datetime
import json
import os
import pathlib
import random
import sys
import time
import numpy as np
from typing import Dict, Union
from scipy.stats import truncnorm

from dotenv import load_dotenv, set_key
from microsoft_bonsai_api.simulator.client import BonsaiClient, BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import (
    SimulatorInterface,
    SimulatorState,
    SimulatorSessionResponse,
)
from azure.core.exceptions import HttpResponseError
from functools import partial

from sim import cartpole

default_config = {}


class TemplateSimulatorSession:
    def __init__(self, env_name: str = "BonsaiPythonSim"):
        self.simulator = cartpole.CartPole()
        self.count_view = False
        self.env_name = env_name

    def get_state(self) -> Dict[str, float]:
        """Extract current states from the simulator

        Returns
        -------
        Dict[str, float]
            Returns float of current values from the simulator
        """
        state = self.simulator.state.copy()
        return state

    def halted(self) -> bool:
        """Halt current episode. Note, this should only return True if the simulator has reached an unexpected state.

        Returns
        -------
        bool
            Whether to terminate current episode
        """
        return False

    def episode_start(self, config: Dict = None) -> None:
        self.config = config
        self.simulator.reset(config)

    def episode_step(self, action: Dict):
        self.simulator.step(action)


def env_setup(env_file: str = ".env"):
    """Helper function to setup connection with Project Bonsai

    Returns
    -------
    Tuple
        workspace, and access_key
    """

    load_dotenv(dotenv_path=env_file, verbose=True, override=True)
    workspace = os.getenv("SIM_WORKSPACE")
    access_key = os.getenv("SIM_ACCESS_KEY")

    env_file_exists = os.path.exists(env_file)
    if not env_file_exists:
        open(env_file, "a").close()

    if not all([env_file_exists, workspace]):
        workspace = input("Please enter your workspace id: ")
        set_key(env_file, "SIM_WORKSPACE", workspace)
    if not all([env_file_exists, access_key]):
        access_key = input("Please enter your access key: ")
        set_key(env_file, "SIM_ACCESS_KEY", access_key)

    load_dotenv(dotenv_path=env_file, verbose=True, override=True)
    workspace = os.getenv("SIM_WORKSPACE")
    access_key = os.getenv("SIM_ACCESS_KEY")

    return workspace, access_key


def main(
    config_setup: bool = False,
    sim_speed: int = 0,
    sim_speed_variance: int = 0,
    env_file: Union[str, bool] = ".env",
    workspace: str = None,
    accesskey: str = None,
):
    # check if workspace or access-key passed in CLI
    use_cli_args = all([workspace, accesskey])

    # use dotenv file if provided
    use_dotenv = env_file or config_setup

    # check for accesskey and workspace id in system variables
    # Three scenarios
    # 1. workspace and accesskey provided by CLI args
    # 2. dotenv provided
    # 3. system variables
    # do 1 if provided, use 2 if provided; ow use 3; if no sys vars or dotenv, fail

    if use_cli_args:
        # BonsaiClientConfig will retrieve as environment variables
        os.environ["SIM_WORKSPACE"] = workspace
        os.environ["SIM_ACCESS_KEY"] = accesskey
    elif use_dotenv:
        if not env_file:
            env_file = ".env"
        print(
            f"No system variables for workspace-id or access-key found, checking in env-file at {env_file}"
        )
        workspace, accesskey = env_setup(env_file)
        load_dotenv(env_file, verbose=True, override=True)
    else:
        try:
            workspace = os.environ["SIM_WORKSPACE"]
            accesskey = os.environ["SIM_ACCESS_KEY"]
        except:
            raise IndexError(
                f"Workspace or access key not set or found. Use --config-setup for help setting up."
            )

    # Grab standardized way to interact with sim API
    sim = TemplateSimulatorSession()

    # Configure client to interact with Bonsai service
    config_client = BonsaiClientConfig()
    client = BonsaiClient(config_client)

    # # Load json file as simulator integration config type file
    with open("cartpole_description.json") as file:
        interface = json.load(file)

    # Create simulator session and init sequence id
    registration_info = SimulatorInterface(
        name=sim.env_name,
        timeout=interface["timeout"],
        simulator_context=config_client.simulator_context,
        description=interface["description"],
    )

    def CreateSession(
        registration_info: SimulatorInterface, config_client: BonsaiClientConfig
    ):
        """Creates a new Simulator Session and returns new session, sequenceId
        """

        try:
            print(
                "config: {}, {}".format(config_client.server, config_client.workspace)
            )
            registered_session: SimulatorSessionResponse = client.session.create(
                workspace_name=config_client.workspace, body=registration_info
            )
            print("Registered simulator. {}".format(registered_session.session_id))

            return registered_session, 1
        except HttpResponseError as ex:
            print(
                "HttpResponseError in Registering session: StatusCode: {}, Error: {}, Exception: {}".format(
                    ex.status_code, ex.error.message, ex
                )
            )
            raise ex
        except Exception as ex:
            print(
                "UnExpected error: {}, Most likely, it's some network connectivity issue, make sure you are able to reach bonsai platform from your network.".format(
                    ex
                )
            )
            raise ex

    registered_session, sequence_id = CreateSession(registration_info, config_client)
    episode = 0
    iteration = 0

    try:
        while True:
            # Advance by the new state depending on the event type
            # TODO: it's risky not doing doing `get_state` without first initializing the sim
            sim_state = SimulatorState(
                sequence_id=sequence_id, state=sim.get_state(), halted=sim.halted(),
            )
            try:
                event = client.session.advance(
                    workspace_name=config_client.workspace,
                    session_id=registered_session.session_id,
                    body=sim_state,
                )
                sequence_id = event.sequence_id
                print(
                    "[{}] Last Event: {}".format(time.strftime("%H:%M:%S"), event.type)
                )
            except HttpResponseError as ex:
                print(
                    "HttpResponseError in Advance: StatusCode: {}, Error: {}, Exception: {}".format(
                        ex.status_code, ex.error.message, ex
                    )
                )
                # This can happen in network connectivity issue, though SDK has retry logic, but even after that request may fail,
                # if your network has some issue, or sim session at platform is going away..
                # So let's re-register sim-session and get a new session and continue iterating. :-)
                registered_session, sequence_id = CreateSession(
                    registration_info, config_client
                )
                continue
            except Exception as err:
                print("Unexpected error in Advance: {}".format(err))
                # Ideally this shouldn't happen, but for very long-running sims It can happen with various reasons, let's re-register sim & Move on.
                # If possible try to notify Bonsai team to see, if this is platform issue and can be fixed.
                registered_session, sequence_id = CreateSession(
                    registration_info, config_client
                )
                continue

            # Event loop
            if event.type == "Idle":
                time.sleep(event.idle.callback_time)
                print("Idling...")
            elif event.type == "EpisodeStart":
                print(event.episode_start.config)
                sim.episode_start(event.episode_start.config)
                episode += 1
            elif event.type == "EpisodeStep":
                iteration += 1
                delay = 0.0
                if sim_speed > 0:
                    if (
                        sim_speed_variance > 0
                    ):  # stochastic delay, truncated normal distribution
                        mu = sim_speed
                        sigma = sim_speed_variance
                        lower = np.max(
                            [0, sim_speed - 3 * sim_speed_variance]
                        )  # truncating at min +/- 3*variance
                        upper = sim_speed + 3 * sim_speed_variance
                        delay = truncnorm.rvs(
                            (lower - mu) / sigma,
                            (upper - mu) / sigma,
                            loc=mu,
                            scale=sigma,
                        )
                        print("stochastic sim delay: {}s".format(delay))
                        time.sleep(delay)
                    else:  # fixed delay
                        delay = sim_speed
                        print("sim delay: {}s".format(delay))
                        time.sleep(delay)
                sim.episode_step(event.episode_step.action)
            elif event.type == "EpisodeFinish":
                print("Episode Finishing...")
                iteration = 0
            elif event.type == "Unregister":
                print(
                    "Simulator Session unregistered by platform because '{}', Registering again!".format(
                        event.unregister.details
                    )
                )
                registered_session, sequence_id = CreateSession(
                    registration_info, config_client
                )
                continue
            else:
                pass
    except KeyboardInterrupt:
        # Gracefully unregister with keyboard interrupt
        client.session.delete(
            workspace_name=config_client.workspace,
            session_id=registered_session.session_id,
        )
        print("Unregistered simulator.")
    except Exception as err:
        # Gracefully unregister for any other exceptions
        client.session.delete(
            workspace_name=config_client.workspace,
            session_id=registered_session.session_id,
        )
        print("Unregistered simulator because: {}".format(err))


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Bonsai and Simulator Integration...")
    parser.add_argument(
        "--config-setup",
        action="store_true",
        default=False,
        help="Use a local environment file to setup access keys and workspace ids",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        metavar="ENVIRONMENT FILE",
        help="path to your environment file",
        default=None,
    )
    parser.add_argument(
        "--workspace",
        type=str,
        metavar="WORKSPACE ID",
        help="your workspace id",
        default=None,
    )
    parser.add_argument(
        "--accesskey",
        type=str,
        metavar="Your Bonsai workspace access-key",
        help="your bonsai workspace access key",
        default=None,
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--test-exported",
        type=int,
        const=5000,  # if arg is passed with no PORT, use this
        nargs="?",
        metavar="PORT",
        help="Run simulator with an exported brain running on localhost:PORT (default 5000)",
    )

    parser.add_argument(
        "--iteration-limit",
        type=int,
        metavar="EPISODE_ITERATIONS",
        help="Episode iteration limit when running local test.",
        default=200,
    )

    parser.add_argument(
        "--sim-speed",
        type=int,
        metavar="SIM_SPEED",
        help="additional emulated sim speed wait in seconds, default: adds 0s",
        default=0,
    )

    parser.add_argument(
        "--sim-speed-variance",
        type=int,
        metavar="SIM_SPEED_VARIANCE",
        help="emulates stochastic sim speed, adds uniform variance to --sim-speed",
        default=0,
    )

    args = parser.parse_args()

    main(
        config_setup=args.config_setup,
        sim_speed=args.sim_speed,
        sim_speed_variance=args.sim_speed_variance,
        env_file=args.env_file,
        workspace=args.workspace,
        accesskey=args.accesskey,
    )
