import random
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

NUM_TRAFFIC_FLOWS = 350

# traffic = t.Traffic(
#     flows=[
#         t.Flow(
#             route=t.RandomRoute(),
#             rate=60 * 2,
#             actors={
#                 t.TrafficActor(
#                     name="car",
#                     vehicle_type=random.choices(
#                         [
#                             "passenger",
#                             "bus",
#                             "coach",
#                             "truck",
#                             "trailer",
#                         ],
#                         weights=[5, 1, 1, 1, 1],
#                         k=1,
#                     )[0],
#                 ): 1
#             },
#         )
#         for _ in range(NUM_TRAFFIC_FLOWS)
#     ]
# )
traffic = t.Traffic(
    flows=[
        # t.Flow(
        #     route=t.Route(begin=("-60477583#0", 0, 3), end=("-4212250#3", 0, "max")),
        #     rate=60 * 60,
        #     actors={t.TrafficActor(name="car", vehicle_type="passenger"): 1},
        # ),
        # t.Flow(
        #     route=t.Route(begin=("27616550#2", 0, 3), end=("-4212250#3", 0, "max")),
        #     rate=160 * 160,
        #     actors={t.TrafficActor(name="car", vehicle_type="passenger"): 1},
        # ),
        t.Flow(
            route=t.Route(begin=("4212249#1", 0, 3), end=("242973006#2", 0, "max")),
            rate=30 * 30,
            actors={t.TrafficActor(name="car", vehicle_type="passenger"): 1},
        ),
        t.Flow(
            route=t.Route(begin=("27616496", 0, 3), end=("242973006#2", 0, "max")),
            rate=30 * 30,
            actors={t.TrafficActor(name="car", vehicle_type="passenger"): 1},
        ),
        
    ]
)

open_agent_actor = t.SocialAgentActor(
    name="open-agent", agent_locator="open_agent:open_agent-v0"
)

laner_actor = t.SocialAgentActor(
    name="keep-lane-agent",
    agent_locator="zoo.policies:keep-lane-agent-v0",
)
ego_missions = [
    t.Mission(route=t.Route(begin=("244093699#1", 0, 10), end=("242973006#0", 0, "max")))
]


travelling_bubbles = [
    t.Bubble(
        zone=t.PositionalZone(pos=(50, 0), size=(10, 50)),
        margin=5,
        actor=open_agent_actor,
        follow_actor_id=t.Bubble.to_actor_id(laner_actor, mission_group="all"),
        follow_offset=(-7, 10),
    )
]

static_bubbles = [
    t.Bubble(
        zone=t.MapZone((id_, 0, 10), 200, 1),
        margin=5,
        actor=laner_actor,
    )
    for id_ in ["21675239", "126742590#1", "-77720372", "-263506114#6", "-33002812#1"]
] + [
    t.Bubble(
        zone=t.PositionalZone(pos=(1012.19, 1084.20), size=(30, 30)),
        margin=5,
        actor=open_agent_actor,
    )
]

gen_scenario(
    t.Scenario(
        traffic={"basic": traffic},
        # social_agent_missions={
        #     "all": ([laner_actor, open_agent_actor], [t.Mission(route=t.RandomRoute())])
        # },
        # bubbles=[*travelling_bubbles, *static_bubbles],
        ego_missions=ego_missions,
    ),
    output_dir=Path(__file__).parent,
)
