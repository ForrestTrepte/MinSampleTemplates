#!/usr/bin/env python3

import os
import time
from microsoft_bonsai_api.simulator.client import BonsaiClient, BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import (
    SimulatorInterface,
    SimulatorState,
    SimulatorSessionResponse,
)
from azure.core.exceptions import HttpResponseError
from sim import Sim

def main():
    workspace = os.getenv("SIM_WORKSPACE")
    accesskey = os.getenv("SIM_ACCESS_KEY")

    simulator = Sim()

    config_client = BonsaiClientConfig()
    client = BonsaiClient(config_client)

    registration_info = SimulatorInterface(
        name="BonsaiPythonSim",
        timeout=60,
        simulator_context=config_client.simulator_context,
        description=None,
    )

    try:
        print("config: {}, {}".format(config_client.server, config_client.workspace))
        registered_session: SimulatorSessionResponse = client.session.create(workspace_name=config_client.workspace, body=registration_info)
        print("Registered simulator. {}".format(registered_session.session_id))
    except HttpResponseError as ex:
        print("HttpResponseError in Registering session: StatusCode: {}, Error: {}, Exception: {}".format(ex.status_code, ex.error.message, ex))
        raise ex
    except Exception as ex:
        print("UnExpected error: {}, Most likely, it's some network connectivity issue, make sure you are able to reach bonsai platform from your network.".format(ex))
        raise ex

    sequence_id = 1
    episode = 0
    iteration = 0

    try:
        while True:
            sim_state = SimulatorState(sequence_id=sequence_id, state=simulator.state.copy(), halted=simulator.halted)
            try:
                event = client.session.advance(
                    workspace_name=config_client.workspace,
                    session_id=registered_session.session_id,
                    body=sim_state,
                )
                sequence_id = event.sequence_id
                print("[{}] Last Event: {}".format(time.strftime("%H:%M:%S"), event.type))
            except HttpResponseError as ex:
                print("HttpResponseError in Advance: StatusCode: {}, Error: {}, Exception: {}".format(ex.status_code, ex.error.message, ex))
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
                simulator.reset(event.episode_start.config)
                episode += 1
            elif event.type == "EpisodeStep":
                iteration += 1
                delay = 0.0
                simulator.step(event.episode_step.action)
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
    main()
